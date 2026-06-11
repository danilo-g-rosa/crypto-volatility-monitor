"""
Motor de Lotka-Volterra para Dinâmica de Compradores/Vendedores.

Implementa as equações de Lotka-Volterra (predador-presa) adaptadas para
modelar a interação entre compradores (presa) e vendedores (predador) no
mercado de criptomoedas.

Fundamento Matemático:
    Equações de Lotka-Volterra:
        dx/dt = α·x - β·x·y    (compradores: crescem naturalmente,
                                 consumidos por vendedores)
        dy/dt = δ·x·y - γ·y    (vendedores: crescem ao consumir compradores,
                                 decaem naturalmente)

    onde:
        x = população de compradores (demanda)
        y = população de vendedores (oferta)
        α = taxa de crescimento natural dos compradores
            (calibrada pelo momentum de retornos positivos)
        β = taxa de interação compradores→vendedores
            (calibrada pela frequência de reversões de retorno)
        δ = taxa de resposta dos vendedores à demanda
        γ = taxa de decaimento natural dos vendedores

    Pontos de equilíbrio:
        x* = γ/δ  (equilíbrio de compradores)
        y* = α/β  (equilíbrio de vendedores)

    As órbitas no espaço de fases são curvas fechadas ao redor de (x*, y*),
    representando ciclos de dominância compradores↔vendedores.

    Período de oscilação (aproximação linearizada):
        T ≈ 2π / √(α·γ)

    Resolução numérica via Runge-Kutta de 4ª ordem (RK4).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LotkaVolterraEngine:
    """
    Motor de simulação de Lotka-Volterra para dinâmica comprador/vendedor.

    Modela compradores como presas e vendedores como predadores,
    capturando os ciclos naturais de dominância de oferta e demanda
    no mercado de criptomoedas.
    """

    def __init__(self) -> None:
        """Inicializa o motor Lotka-Volterra."""
        np.random.seed(42)
        logger.info("LotkaVolterraEngine inicializado.")

    def _compute_returns(self, close: np.ndarray) -> np.ndarray:
        """
        Calcula retornos logarítmicos a partir dos preços de fechamento.

        Parâmetros:
            close: Array de preços de fechamento.

        Retorna:
            Array de retornos logarítmicos.
        """
        close = np.clip(close, 1e-10, None)
        return np.diff(np.log(close))

    def _calibrate_alpha(self, returns: np.ndarray) -> float:
        """
        Calibra α (taxa de crescimento dos compradores) pelo momentum
        de retornos positivos.

        α é proporcional à fração e magnitude média dos retornos positivos
        recentes, representando a tendência natural de crescimento da demanda.

        Cálculo:
            α = fração_positivos × magnitude_média_positivos × fator_escala

        Parâmetros:
            returns: Array de retornos logarítmicos.

        Retorna:
            α calibrado, limitado ao intervalo [0.05, 2.0].
        """
        positive_returns = returns[returns > 0]

        if len(positive_returns) == 0:
            logger.warning("Nenhum retorno positivo. Usando α mínimo.")
            return 0.05

        fraction_positive = len(positive_returns) / len(returns)
        mean_positive = float(np.mean(np.abs(positive_returns)))
        vol = float(np.std(returns)) if np.std(returns) > 1e-12 else 0.01

        # Normalizar magnitude pela volatilidade
        normalized_magnitude = mean_positive / vol

        # α combina frequência e magnitude dos retornos positivos
        alpha = fraction_positive * normalized_magnitude * 1.5
        alpha = float(np.clip(alpha, 0.05, 2.0))

        logger.debug(
            f"Calibração α: frac_pos={fraction_positive:.3f}, "
            f"mag_norm={normalized_magnitude:.3f}, α={alpha:.4f}"
        )
        return alpha

    def _calibrate_beta(self, returns: np.ndarray) -> float:
        """
        Calibra β (taxa de interação) pela frequência de reversões de retorno.

        Uma reversão ocorre quando o sinal do retorno muda de um candle
        para o seguinte (positivo→negativo ou vice-versa). Alta frequência
        de reversões indica interação forte entre compradores e vendedores.

        Cálculo:
            β = nº_reversões / (nº_retornos - 1) × fator_escala

        Parâmetros:
            returns: Array de retornos logarítmicos.

        Retorna:
            β calibrado, limitado ao intervalo [0.05, 2.0].
        """
        if len(returns) < 2:
            return 0.5

        signs = np.sign(returns)
        # Contar reversões (mudanças de sinal)
        reversals = np.sum(signs[:-1] != signs[1:])
        reversal_rate = reversals / (len(returns) - 1)

        # β proporcional à taxa de reversão
        beta = reversal_rate * 2.0
        beta = float(np.clip(beta, 0.05, 2.0))

        logger.debug(
            f"Calibração β: {reversals} reversões em {len(returns) - 1} pares, "
            f"taxa={reversal_rate:.3f}, β={beta:.4f}"
        )
        return beta

    def _calibrate_delta(self, returns: np.ndarray) -> float:
        """
        Calibra δ (taxa de resposta dos vendedores) pela intensidade com
        que retornos negativos seguem retornos positivos.

        δ mede quão rapidamente os vendedores respondem à atividade dos
        compradores. Se retornos negativos grandes seguem retornos positivos
        frequentemente, δ é alto.

        Parâmetros:
            returns: Array de retornos logarítmicos.

        Retorna:
            δ calibrado, limitado ao intervalo [0.02, 1.5].
        """
        if len(returns) < 3:
            return 0.3

        # Encontrar transições positivo → negativo
        sell_responses: List[float] = []
        for i in range(1, len(returns)):
            if returns[i - 1] > 0 and returns[i] < 0:
                sell_responses.append(abs(returns[i]))

        if not sell_responses:
            logger.debug("Sem transições pos→neg. Usando δ padrão.")
            return 0.3

        vol = float(np.std(returns)) if np.std(returns) > 1e-12 else 0.01
        mean_response = float(np.mean(sell_responses))
        response_ratio = mean_response / vol

        # Frequência dessas transições
        transition_freq = len(sell_responses) / (len(returns) - 1)

        delta = response_ratio * transition_freq * 3.0
        delta = float(np.clip(delta, 0.02, 1.5))

        logger.debug(
            f"Calibração δ: {len(sell_responses)} transições pos→neg, "
            f"resp_ratio={response_ratio:.3f}, δ={delta:.4f}"
        )
        return delta

    def _calibrate_gamma(self, returns: np.ndarray) -> float:
        """
        Calibra γ (taxa de decaimento dos vendedores) pela tendência de
        retornos negativos se esgotarem.

        γ mede quão rapidamente os vendedores perdem força. Se sequências
        de retornos negativos são curtas, γ é alto (vendedores se esgotam
        rapidamente).

        Parâmetros:
            returns: Array de retornos logarítmicos.

        Retorna:
            γ calibrado, limitado ao intervalo [0.05, 2.0].
        """
        if len(returns) < 3:
            return 0.5

        # Comprimento de sequências de retornos negativos
        neg_run_lengths: List[int] = []
        current_run = 0

        for r in returns:
            if r < 0:
                current_run += 1
            else:
                if current_run > 0:
                    neg_run_lengths.append(current_run)
                current_run = 0

        if current_run > 0:
            neg_run_lengths.append(current_run)

        if not neg_run_lengths:
            logger.debug("Sem sequências negativas. Usando γ alto.")
            return 1.5

        mean_run = float(np.mean(neg_run_lengths))

        # γ inversamente proporcional à duração das sequências negativas
        gamma = 1.0 / max(mean_run, 0.5)
        gamma = float(np.clip(gamma, 0.05, 2.0))

        logger.debug(
            f"Calibração γ: {len(neg_run_lengths)} sequências negativas, "
            f"duração média={mean_run:.2f}, γ={gamma:.4f}"
        )
        return gamma

    def _lotka_volterra_derivatives(
        self,
        x: float,
        y: float,
        alpha: float,
        beta: float,
        delta: float,
        gamma: float
    ) -> Tuple[float, float]:
        """
        Calcula as derivadas do sistema Lotka-Volterra.

        dx/dt = α·x - β·x·y
        dy/dt = δ·x·y - γ·y

        Parâmetros:
            x: População de compradores.
            y: População de vendedores.
            alpha, beta, delta, gamma: Parâmetros do modelo.

        Retorna:
            Tupla (dx/dt, dy/dt).
        """
        dx = alpha * x - beta * x * y
        dy = delta * x * y - gamma * y
        return dx, dy

    def _rk4_step(
        self,
        x: float,
        y: float,
        dt: float,
        alpha: float,
        beta: float,
        delta: float,
        gamma: float
    ) -> Tuple[float, float]:
        """
        Executa um passo do método Runge-Kutta de 4ª ordem (RK4).

        O RK4 avalia a derivada em quatro pontos intermediários e computa
        uma média ponderada para obter precisão de 4ª ordem:

            k1 = f(t_n, y_n)
            k2 = f(t_n + dt/2, y_n + dt/2·k1)
            k3 = f(t_n + dt/2, y_n + dt/2·k2)
            k4 = f(t_n + dt, y_n + dt·k3)
            y_{n+1} = y_n + dt/6·(k1 + 2·k2 + 2·k3 + k4)

        Parâmetros:
            x, y: Populações atuais.
            dt: Passo temporal.
            alpha, beta, delta, gamma: Parâmetros do modelo.

        Retorna:
            Tupla (x_new, y_new) com populações atualizadas.
        """
        # k1
        dx1, dy1 = self._lotka_volterra_derivatives(x, y, alpha, beta, delta, gamma)

        # k2
        x2 = x + 0.5 * dt * dx1
        y2 = y + 0.5 * dt * dy1
        dx2, dy2 = self._lotka_volterra_derivatives(x2, y2, alpha, beta, delta, gamma)

        # k3
        x3 = x + 0.5 * dt * dx2
        y3 = y + 0.5 * dt * dy2
        dx3, dy3 = self._lotka_volterra_derivatives(x3, y3, alpha, beta, delta, gamma)

        # k4
        x4 = x + dt * dx3
        y4 = y + dt * dy3
        dx4, dy4 = self._lotka_volterra_derivatives(x4, y4, alpha, beta, delta, gamma)

        # Atualização RK4
        x_new = x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        y_new = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)

        # Populações não podem ser negativas
        x_new = max(x_new, 1e-10)
        y_new = max(y_new, 1e-10)

        return x_new, y_new

    def _estimate_cycle_period(
        self, buyers: List[float], dt: float
    ) -> Optional[float]:
        """
        Estima o período de oscilação a partir da série temporal de compradores.

        Utiliza a detecção de picos (máximos locais) na série de compradores
        e calcula a distância média entre picos consecutivos.

        Parâmetros:
            buyers: Lista de populações de compradores ao longo do tempo.
            dt: Passo temporal.

        Retorna:
            Período estimado em unidades de tempo, ou None se não oscilar.
        """
        if len(buyers) < 5:
            return None

        arr = np.array(buyers)

        # Detectar máximos locais
        peaks: List[int] = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks.append(i)

        if len(peaks) < 2:
            return None

        # Distâncias entre picos consecutivos
        peak_distances = np.diff(peaks)
        mean_period = float(np.mean(peak_distances)) * dt

        if mean_period < dt * 2:
            return None  # Período muito curto, provavelmente ruído

        logger.debug(
            f"Período estimado: {mean_period:.2f} "
            f"({len(peaks)} picos detectados)"
        )
        return mean_period

    def run_simulation(
        self,
        df: pd.DataFrame,
        projection_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Executa a simulação de Lotka-Volterra para dinâmica comprador/vendedor.

        Etapas:
        1. Calcula retornos e determina condições iniciais (x₀, y₀)
        2. Calibra parâmetros (α, β, δ, γ) a partir dos dados de mercado
        3. Resolve as equações de Lotka-Volterra via RK4
        4. Calcula pontos de equilíbrio e estima período de oscilação
        5. Determina dominância atual (compradores vs vendedores)

        Parâmetros:
            df: DataFrame com coluna 'close' (preços de fechamento).
            projection_steps: Número de passos futuros a projetar.

        Retorna:
            Dicionário com trajetórias de compradores/vendedores,
            parâmetros calibrados, equilíbrios, dominância e retrato de fase.

        Levanta:
            ValueError: Se os dados forem insuficientes.
        """
        # --- Validação de entrada ---
        if df is None or df.empty or len(df) < 10:
            raise ValueError(
                "Dados insuficientes para simulação de Lotka-Volterra. "
                "São necessários pelo menos 10 registros com coluna 'close'."
            )

        if 'close' not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'close'.")

        close = df['close'].dropna().values.astype(float)
        if len(close) < 10:
            raise ValueError("Menos de 10 preços válidos após remover NaN.")

        logger.info(
            f"Iniciando simulação Lotka-Volterra: {len(close)} candles, "
            f"{projection_steps} passos de projeção."
        )

        # --- Calcular retornos ---
        returns = self._compute_returns(close)

        if len(returns) < 5:
            raise ValueError("Retornos insuficientes para calibração (mínimo 5).")

        # --- Condições iniciais ---
        # Usar as últimas 30 observações (ou todas se < 30) para estimar x₀, y₀
        window = min(30, len(returns))
        recent_returns = returns[-window:]

        # x₀ = proporção de retornos positivos (compradores)
        x0 = float(np.sum(recent_returns > 0)) / len(recent_returns)
        x0 = max(x0, 0.05)  # Mínimo de 5%

        # y₀ = proporção de retornos negativos (vendedores)
        y0 = float(np.sum(recent_returns < 0)) / len(recent_returns)
        y0 = max(y0, 0.05)  # Mínimo de 5%

        logger.debug(f"Condições iniciais: x₀={x0:.4f}, y₀={y0:.4f}")

        # --- Calibração dos parâmetros ---
        alpha = self._calibrate_alpha(returns)
        beta = self._calibrate_beta(returns)
        delta = self._calibrate_delta(returns)
        gamma = self._calibrate_gamma(returns)

        # Pontos de equilíbrio teóricos
        equilibrium_buyers = gamma / delta if delta > 1e-10 else float('inf')
        equilibrium_sellers = alpha / beta if beta > 1e-10 else float('inf')

        logger.info(
            f"Parâmetros calibrados: α={alpha:.4f}, β={beta:.4f}, "
            f"δ={delta:.4f}, γ={gamma:.4f}"
        )
        logger.info(
            f"Equilíbrios teóricos: x*={equilibrium_buyers:.4f}, "
            f"y*={equilibrium_sellers:.4f}"
        )

        # --- Resolver via RK4 ---
        dt = 0.1  # Passo temporal fino para estabilidade numérica
        substeps = 10  # Sub-passos por passo de projeção

        steps_list: List[int] = [0]
        buyers_list: List[float] = [x0]
        sellers_list: List[float] = [y0]
        phase_x: List[float] = [x0]
        phase_y: List[float] = [y0]

        x, y = x0, y0

        for step in range(1, projection_steps + 1):
            for _ in range(substeps):
                x, y = self._rk4_step(x, y, dt, alpha, beta, delta, gamma)

                # Limitar valores extremos para estabilidade
                x = min(x, 10.0)
                y = min(y, 10.0)

                phase_x.append(x)
                phase_y.append(y)

            steps_list.append(step)
            buyers_list.append(x)
            sellers_list.append(y)

        # --- Análise dos resultados ---
        # Dominância atual
        current_x = buyers_list[-1]
        current_y = sellers_list[-1]
        current_dominance = "COMPRADORES" if current_x > current_y else "VENDEDORES"

        # Período de oscilação
        cycle_period = self._estimate_cycle_period(buyers_list, dt * substeps)

        # Período teórico (aproximação linearizada ao redor do equilíbrio)
        if alpha * gamma > 0:
            theoretical_period = 2.0 * np.pi / np.sqrt(alpha * gamma)
            logger.debug(f"Período teórico (linearizado): {theoretical_period:.4f}")

        result = {
            "steps": steps_list,
            "buyers": buyers_list,
            "sellers": sellers_list,
            "alpha": alpha,
            "beta": beta,
            "delta": delta,
            "gamma": gamma,
            "equilibrium_buyers": float(equilibrium_buyers),
            "equilibrium_sellers": float(equilibrium_sellers),
            "current_dominance": current_dominance,
            "cycle_period": cycle_period,
            "phase_portrait_x": phase_x,
            "phase_portrait_y": phase_y,
        }

        logger.info(
            f"Simulação Lotka-Volterra concluída. "
            f"Dominância: {current_dominance}, "
            f"x_final={current_x:.4f}, y_final={current_y:.4f}, "
            f"período={cycle_period}"
        )

        return result
