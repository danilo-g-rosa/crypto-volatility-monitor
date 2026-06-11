"""
Kalman Filter Engine
====================

Motor de filtragem e previsão baseado no Filtro de Kalman linear para
séries de preços de criptomoedas. Implementa um modelo de velocidade
constante (constant velocity) em 2D para denoising e extrapolação
de curto prazo.

Modelo de Estado (Espaço de Estados Linear-Gaussiano):
    Estado x_t = [preço, tendência]^T

    Transição de estado:
        x_t = F · x_{t-1} + w_t,   w_t ~ N(0, Q)

        F = [[1, 1],    →  preço_t = preço_{t-1} + tendência_{t-1}
             [0, 1]]    →  tendência_t = tendência_{t-1}

    Observação:
        z_t = H · x_t + v_t,   v_t ~ N(0, R)

        H = [1, 0]     →  observamos apenas o preço (com ruído)

Calibração Automática:
    - R (variância de medição): estimada pela variância dos resíduos entre
      preço observado e uma média móvel suave.
    - Q (variância de processo): derivada da variância das diferenças de
      segunda ordem (aceleração) dos preços, refletindo mudanças de
      tendência não modeladas.

Equações do Filtro de Kalman (fase de predição e atualização):

    Predição:
        x̂_{t|t-1} = F · x̂_{t-1|t-1}
        P_{t|t-1}  = F · P_{t-1|t-1} · F^T + Q

    Atualização:
        y_t = z_t - H · x̂_{t|t-1}          (inovação)
        S_t = H · P_{t|t-1} · H^T + R       (covariância da inovação)
        K_t = P_{t|t-1} · H^T · S_t^{-1}    (ganho de Kalman)
        x̂_{t|t} = x̂_{t|t-1} + K_t · y_t
        P_{t|t} = (I - K_t · H) · P_{t|t-1}
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class KalmanFilterEngine:
    """
    Motor de Filtro de Kalman para denoising e previsão de preços de
    criptomoedas.

    Utiliza um modelo de velocidade constante (estado = [preço, tendência])
    para filtrar ruído de observação e extrapolar previsões de curto prazo
    com bandas de incerteza derivadas da matriz de covariância P.
    """

    def __init__(self) -> None:
        """Inicializa o motor sem estado persistente."""
        logger.info("KalmanFilterEngine inicializado.")

    # ------------------------------------------------------------------
    # Calibração de ruído
    # ------------------------------------------------------------------

    def _calibrate_noise(self, prices: np.ndarray) -> tuple:
        """
        Calibra as matrizes de ruído de processo (Q) e de medição (R)
        a partir dos dados históricos de preços.

        Estratégia:
            - R (ruído de medição): variância dos resíduos entre o preço
              observado e uma média móvel de janela 5 (proxy do preço
              verdadeiro). Representa o quanto a observação é ruidosa.
            - Q (ruído de processo): a componente de preço usa a variância
              das diferenças de primeira ordem (volatilidade do retorno
              absoluto); a componente de tendência usa a variância das
              diferenças de segunda ordem (aceleração), refletindo mudanças
              inesperadas na tendência.

        Parâmetros:
            prices: array 1D de preços de fechamento.

        Retorna:
            Tupla (Q, R) onde:
                Q: matriz 2x2 de covariância do processo.
                R: escalar, variância de medição.
        """
        n = len(prices)

        # --- R: variância de medição ---
        window = min(5, n // 2)
        if window >= 2:
            ma = pd.Series(prices).rolling(window=window, min_periods=1).mean().values
            residuals = prices - ma
            R = float(np.var(residuals, ddof=1))
        else:
            R = float(np.var(prices, ddof=1))

        R = max(R, 1e-8)

        # --- Q: variância de processo ---
        diff1 = np.diff(prices)
        q_price = float(np.var(diff1, ddof=1)) if len(diff1) > 1 else R * 0.1

        diff2 = np.diff(prices, n=2)
        q_trend = float(np.var(diff2, ddof=1)) if len(diff2) > 1 else q_price * 0.1

        q_price = max(q_price, 1e-8)
        q_trend = max(q_trend, 1e-8)

        Q = np.array([
            [q_price, 0.0],
            [0.0, q_trend],
        ])

        logger.info(
            "Ruído calibrado: R=%.4f, Q_price=%.4f, Q_trend=%.4f",
            R, q_price, q_trend,
        )

        return Q, R

    # ------------------------------------------------------------------
    # Filtro de Kalman: passada forward
    # ------------------------------------------------------------------

    def _run_forward_pass(
        self,
        prices: np.ndarray,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: float,
    ) -> tuple:
        """
        Executa a passada forward (filtragem) do Filtro de Kalman sobre
        toda a série histórica de preços.

        Para cada observação z_t:
            1. Predição: propaga estado e covariância.
            2. Atualização: incorpora a observação via ganho de Kalman.

        Parâmetros:
            prices: array 1D de preços observados.
            F: matriz de transição de estado (2x2).
            H: vetor de observação (1x2).
            Q: matriz de covariância do processo (2x2).
            R: variância de medição (escalar).

        Retorna:
            Tupla (filtered_states, P_history, K_history) onde:
                filtered_states: array (n, 2) com [preço, tendência] filtrados.
                P_history: lista de matrizes P (2x2), uma por observação.
                K_history: lista de ganhos de Kalman (escalar K[0]).
        """
        n = len(prices)

        # Inicialização do estado: [preço_0, tendência_0]
        x = np.array([prices[0], 0.0])  # tendência inicial = 0

        # Inicialização da covariância: incerteza alta
        P = np.eye(2) * R * 10.0

        filtered_states = np.zeros((n, 2))
        P_history = []
        K_history = []

        for t in range(n):
            # ---- Etapa de Predição ----
            if t > 0:
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
            else:
                # Na primeira observação, não fazemos predição
                x_pred = x.copy()
                P_pred = P.copy()

            # ---- Etapa de Atualização ----
            z = prices[t]
            y = z - H @ x_pred                          # Inovação
            S = H @ P_pred @ H.T + R                    # Covariância da inovação
            S_inv = 1.0 / float(S)
            K = (P_pred @ H.T) * S_inv                  # Ganho de Kalman (2x1)

            x = x_pred + K.flatten() * float(y)
            P = (np.eye(2) - np.outer(K.flatten(), H)) @ P_pred

            # Armazenar resultados
            filtered_states[t] = x
            P_history.append(P.copy())
            K_history.append(float(K[0]))                # Ganho do componente preço

        return filtered_states, P_history, K_history

    # ------------------------------------------------------------------
    # Previsão (extrapolação)
    # ------------------------------------------------------------------

    def _run_forecast(
        self,
        x_last: np.ndarray,
        P_last: np.ndarray,
        F: np.ndarray,
        Q: np.ndarray,
        steps: int,
    ) -> tuple:
        """
        Extrapola o estado filtrado para `steps` períodos à frente
        usando apenas a equação de predição (sem novas observações).

        A incerteza cresce a cada passo, pois P acumula Q sem correção.

        Parâmetros:
            x_last: último estado filtrado [preço, tendência].
            P_last: última matriz de covariância P.
            F: matriz de transição de estado.
            Q: matriz de covariância do processo.
            steps: número de passos de previsão.

        Retorna:
            Tupla (forecast_prices, forecast_stds) onde:
                forecast_prices: array de preços previstos.
                forecast_stds: array de desvios-padrão da previsão.
        """
        x = x_last.copy()
        P = P_last.copy()

        forecast_prices = np.zeros(steps)
        forecast_stds = np.zeros(steps)

        for t in range(steps):
            x = F @ x
            P = F @ P @ F.T + Q

            forecast_prices[t] = x[0]
            # Desvio-padrão da componente preço = sqrt(P[0,0])
            forecast_stds[t] = np.sqrt(max(P[0, 0], 0.0))

        return forecast_prices, forecast_stds

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def run_filter(self, df: pd.DataFrame, steps: int = 7) -> dict:
        """
        Executa o Filtro de Kalman completo: filtragem histórica + previsão.

        Etapas:
            1. Validação dos dados de entrada.
            2. Calibração automática de Q (ruído de processo) e R (ruído
               de medição) a partir da série de preços.
            3. Passada forward: filtragem de toda a série histórica,
               produzindo estimativas denoised e ganhos de Kalman.
            4. Extrapolação: previsão de `steps` passos à frente com
               bandas de incerteza crescentes.

        Parâmetros:
            df: DataFrame com coluna obrigatória ['close'] e opcional
                ['timestamp']. Mínimo de 10 observações.
            steps: número de passos de previsão (default: 7).

        Retorna:
            Dicionário com:
                - 'filtered_prices': lista de preços filtrados (mesmo tamanho
                  que a entrada), representando a série denoised.
                - 'filtered_upper': banda superior filtrada (+1.96·σ).
                - 'filtered_lower': banda inferior filtrada (-1.96·σ).
                - 'forecast_prices': lista de preços previstos (length=steps).
                - 'forecast_upper': banda superior da previsão.
                - 'forecast_lower': banda inferior da previsão.
                - 'forecast_timestamps': timestamps das previsões.
                - 'kalman_gain_history': lista do ganho de Kalman ao longo
                  do tempo (componente de preço), útil para diagnóstico.
                - 'noise_ratio': razão R/Q_price. Valores altos indicam
                  observações muito ruidosas relativas ao processo.

        Raises:
            ValueError: se os dados de entrada forem insuficientes ou inválidos.
        """
        # ---- Validação ----
        if df is None or df.empty or len(df) < 10:
            raise ValueError(
                "Dados insuficientes para o Filtro de Kalman. "
                "São necessárias pelo menos 10 observações."
            )

        if "close" not in df.columns:
            raise ValueError(
                "Coluna 'close' é obrigatória no DataFrame de entrada."
            )

        logger.info(
            "Iniciando Filtro de Kalman: %d observações, %d passos de previsão.",
            len(df), steps,
        )

        prices = df["close"].dropna().values.astype(float)

        if len(prices) < 10:
            raise ValueError(
                "Menos de 10 preços válidos (não-nulos) disponíveis."
            )

        # ---- Definição das matrizes do modelo ----
        # Transição: modelo de velocidade constante
        F = np.array([
            [1.0, 1.0],
            [0.0, 1.0],
        ])

        # Observação: observamos apenas o preço
        H = np.array([1.0, 0.0])

        # ---- Calibração ----
        Q, R = self._calibrate_noise(prices)

        # ---- Passada forward (filtragem) ----
        filtered_states, P_history, K_history = self._run_forward_pass(
            prices, F, H, Q, R
        )

        filtered_prices = filtered_states[:, 0]
        filtered_stds = np.array([np.sqrt(max(P[0, 0], 0.0)) for P in P_history])

        z_score_95 = 1.96
        filtered_upper = (filtered_prices + z_score_95 * filtered_stds).tolist()
        filtered_lower = (filtered_prices - z_score_95 * filtered_stds).tolist()

        # ---- Previsão ----
        x_last = filtered_states[-1]
        P_last = P_history[-1]

        forecast_prices, forecast_stds = self._run_forecast(
            x_last, P_last, F, Q, steps
        )

        forecast_upper = (forecast_prices + z_score_95 * forecast_stds).tolist()
        forecast_lower = (forecast_prices - z_score_95 * forecast_stds).tolist()

        # ---- Timestamps da previsão ----
        forecast_timestamps = self._generate_future_timestamps(df, steps)

        # ---- Razão de ruído R/Q ----
        q_price = Q[0, 0]
        noise_ratio = float(R / q_price) if q_price > 0 else float("inf")

        result = {
            "filtered_prices": filtered_prices.tolist(),
            "filtered_upper": filtered_upper,
            "filtered_lower": filtered_lower,
            "forecast_prices": forecast_prices.tolist(),
            "forecast_upper": forecast_upper,
            "forecast_lower": forecast_lower,
            "forecast_timestamps": forecast_timestamps,
            "kalman_gain_history": K_history,
            "noise_ratio": noise_ratio,
        }

        logger.info(
            "Filtro de Kalman concluído. Razão de ruído R/Q=%.4f, "
            "ganho final=%.4f",
            noise_ratio, K_history[-1] if K_history else 0.0,
        )

        return result

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def _generate_future_timestamps(
        self, df: pd.DataFrame, steps: int
    ) -> List[str]:
        """
        Gera timestamps futuros com base no intervalo observado nos dados.

        Se a coluna 'timestamp' existir e for conversível a datetime,
        extrapola usando o intervalo mediano entre observações. Caso
        contrário, retorna índices inteiros sequenciais como fallback.

        Parâmetros:
            df: DataFrame original com dados históricos.
            steps: número de passos futuros.

        Retorna:
            Lista de strings representando os timestamps futuros.
        """
        if "timestamp" in df.columns:
            try:
                ts = pd.to_datetime(df["timestamp"])
                diffs = ts.diff().dropna()
                if len(diffs) > 0:
                    median_delta = diffs.median()
                    last_ts = ts.iloc[-1]
                    future_ts = [
                        (last_ts + median_delta * (i + 1)).isoformat()
                        for i in range(steps)
                    ]
                    return future_ts
            except Exception as e:
                logger.warning(
                    "Erro ao processar timestamps: %s. Usando fallback.", str(e)
                )

        # Fallback: índices inteiros
        last_idx = len(df)
        return [str(last_idx + i) for i in range(steps)]
