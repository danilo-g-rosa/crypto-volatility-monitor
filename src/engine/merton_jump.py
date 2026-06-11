"""
Merton Jump-Diffusion Engine
=============================

Motor de simulação baseado no modelo de Merton (1976) para saltos em preços
de criptomoedas. Estende o Movimento Browniano Geométrico (GBM) clássico com
um Processo de Poisson Composto, permitindo capturar movimentos bruscos
(jumps) que ocorrem com frequência em mercados de alta volatilidade.

Modelo Matemático:
    dS/S = (μ - λ·k)·dt + σ·dW + J·dN

Onde:
    - μ  : drift (retorno médio anualizado)
    - σ  : volatilidade difusiva (desvio-padrão dos retornos log)
    - λ  : intensidade do processo de Poisson (frequência de saltos)
    - k  : E[e^J - 1], compensador de drift para manter a martingale
    - dW : incremento Browniano padrão ~ N(0, dt)
    - dN : incremento Poisson ~ Poisson(λ·dt)
    - J  : tamanho do salto ~ N(μ_J, σ_J²)

Calibração:
    - λ é estimado a partir da frequência de anomalias na coluna 'anomaly'
    - μ_J e σ_J são estimados a partir dos retornos extremos (|r| > 2σ)
    - μ e σ são estimados dos retornos logarítmicos históricos
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class MertonJumpEngine:
    """
    Motor de simulação Merton Jump-Diffusion para preços de criptomoedas.

    Combina o GBM clássico com saltos aleatórios calibrados a partir de
    dados históricos de anomalias e retornos extremos, produzindo cenários
    de preço futuros mais realistas do que um GBM puro.
    """

    def __init__(self) -> None:
        """Inicializa o motor sem estado persistente."""
        logger.info("MertonJumpEngine inicializado.")

    # ------------------------------------------------------------------
    # Métodos auxiliares de calibração
    # ------------------------------------------------------------------

    def _compute_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcula os retornos logarítmicos da série de preços.

        Parâmetros:
            prices: array de preços de fechamento.

        Retorna:
            Array de retornos log: ln(S_t / S_{t-1}).
        """
        prices = prices.astype(float)
        returns = np.diff(np.log(prices))
        return returns

    def _calibrate_diffusion(self, log_returns: np.ndarray) -> tuple:
        """
        Calibra os parâmetros difusivos μ (drift) e σ (volatilidade)
        a partir dos retornos logarítmicos.

        Parâmetros:
            log_returns: array de retornos logarítmicos.

        Retorna:
            Tupla (mu, sigma) com drift e volatilidade por período.
        """
        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns, ddof=1))
        # Proteção contra sigma zero
        sigma = max(sigma, 1e-8)
        logger.debug("Parâmetros difusivos calibrados: mu=%.6f, sigma=%.6f", mu, sigma)
        return mu, sigma

    def _calibrate_jumps(
        self,
        log_returns: np.ndarray,
        anomaly_flags: Optional[np.ndarray],
        sigma: float,
    ) -> tuple:
        """
        Calibra os parâmetros de salto (λ, μ_J, σ_J) do processo de Poisson
        composto.

        Estratégia:
            1. λ (intensidade) é derivado da fração de observações marcadas
               como anomalia. Se a coluna 'anomaly' não existir ou estiver
               vazia, usa-se a frequência de retornos extremos como fallback.
            2. μ_J e σ_J são estimados dos retornos cujo valor absoluto
               excede 2 desvios-padrão (retornos extremos).

        Parâmetros:
            log_returns: array de retornos logarítmicos.
            anomaly_flags: array booleano de anomalias (pode ser None).
            sigma: volatilidade difusiva calibrada.

        Retorna:
            Tupla (lambda_jumps, mu_jump, sigma_jump).
        """
        n = len(log_returns)

        # --- Calibração de λ ---
        if anomaly_flags is not None and len(anomaly_flags) > 0:
            # anomaly_flags tem tamanho len(df), log_returns tem len(df)-1
            # Alinhamos usando os últimos n elementos
            aligned = anomaly_flags[-n:] if len(anomaly_flags) > n else anomaly_flags
            anomaly_count = np.sum(aligned.astype(bool))
            lambda_jumps = float(anomaly_count / max(n, 1))
            logger.debug(
                "λ calibrado via anomalias: %d anomalias em %d obs → λ=%.4f",
                anomaly_count, n, lambda_jumps,
            )
        else:
            # Fallback: frequência de retornos extremos
            extreme_mask = np.abs(log_returns) > 2.0 * sigma
            lambda_jumps = float(np.sum(extreme_mask) / max(n, 1))
            logger.warning(
                "Coluna 'anomaly' ausente ou vazia. λ calibrado via retornos "
                "extremos: λ=%.4f", lambda_jumps,
            )

        # Piso mínimo para evitar λ = 0 (nenhum salto jamais)
        lambda_jumps = max(lambda_jumps, 0.01)

        # --- Calibração de μ_J e σ_J ---
        extreme_mask = np.abs(log_returns) > 2.0 * sigma
        extreme_returns = log_returns[extreme_mask]

        if len(extreme_returns) >= 3:
            mu_jump = float(np.mean(extreme_returns))
            sigma_jump = float(np.std(extreme_returns, ddof=1))
        else:
            # Fallback: usa estatísticas da cauda inteira escaladas
            mu_jump = 0.0
            sigma_jump = float(sigma * 2.0)
            logger.warning(
                "Poucos retornos extremos (%d). Usando fallback: μ_J=0, σ_J=%.4f",
                len(extreme_returns), sigma_jump,
            )

        sigma_jump = max(sigma_jump, 1e-8)

        logger.info(
            "Parâmetros de salto calibrados: λ=%.4f, μ_J=%.6f, σ_J=%.6f",
            lambda_jumps, mu_jump, sigma_jump,
        )
        return lambda_jumps, mu_jump, sigma_jump

    # ------------------------------------------------------------------
    # Simulação Monte Carlo
    # ------------------------------------------------------------------

    def _simulate_paths(
        self,
        s0: float,
        mu: float,
        sigma: float,
        lambda_jumps: float,
        mu_jump: float,
        sigma_jump: float,
        steps: int,
        num_paths: int,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Gera trajetórias de preço via Monte Carlo usando o modelo de
        Merton Jump-Diffusion.

        Para cada passo temporal:
            S_{t+1} = S_t · exp[(μ - 0.5·σ² - λ·k)·dt + σ·√dt·Z + J·N]

        Onde:
            Z ~ N(0,1)  (componente difusivo)
            N ~ Poisson(λ·dt)  (número de saltos no intervalo)
            J ~ N(μ_J, σ_J²) para cada salto individual
            k = exp(μ_J + 0.5·σ_J²) - 1  (compensador)

        Parâmetros:
            s0: preço inicial.
            mu: drift por período.
            sigma: volatilidade por período.
            lambda_jumps: intensidade de Poisson (saltos/período).
            mu_jump: média do tamanho do salto (log).
            sigma_jump: desvio-padrão do tamanho do salto (log).
            steps: número de passos à frente.
            num_paths: número de trajetórias simuladas.
            dt: fração de tempo por passo (default = 1.0).

        Retorna:
            Matriz (num_paths, steps+1) com as trajetórias de preço,
            incluindo o preço inicial na coluna 0.
        """
        np.random.seed(42)

        # Compensador de drift: k = E[e^J - 1]
        k = np.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1.0

        # Drift ajustado
        drift = (mu - 0.5 * sigma ** 2 - lambda_jumps * k) * dt

        # Matriz de saída
        paths = np.zeros((num_paths, steps + 1))
        paths[:, 0] = s0

        for t in range(1, steps + 1):
            # Componente difusivo
            z = np.random.standard_normal(num_paths)
            diffusion = sigma * np.sqrt(dt) * z

            # Componente de salto
            n_jumps = np.random.poisson(lambda_jumps * dt, num_paths)
            jump_sizes = np.zeros(num_paths)
            for i in range(num_paths):
                if n_jumps[i] > 0:
                    jumps_i = np.random.normal(mu_jump, sigma_jump, n_jumps[i])
                    jump_sizes[i] = np.sum(jumps_i)

            # Evolução log-preço
            log_return = drift + diffusion + jump_sizes
            paths[:, t] = paths[:, t - 1] * np.exp(log_return)

        return paths

    def _simulate_gbm_paths(
        self,
        s0: float,
        mu: float,
        sigma: float,
        steps: int,
        num_paths: int,
        dt: float = 1.0,
    ) -> np.ndarray:
        """
        Gera trajetórias de preço usando GBM puro (sem saltos) para
        comparação visual com o modelo de Merton.

        S_{t+1} = S_t · exp[(μ - 0.5·σ²)·dt + σ·√dt·Z]

        Parâmetros:
            s0: preço inicial.
            mu: drift por período.
            sigma: volatilidade por período.
            steps: número de passos à frente.
            num_paths: número de trajetórias simuladas.
            dt: fração de tempo por passo.

        Retorna:
            Matriz (num_paths, steps+1) com trajetórias GBM puras.
        """
        np.random.seed(42)

        drift = (mu - 0.5 * sigma ** 2) * dt
        paths = np.zeros((num_paths, steps + 1))
        paths[:, 0] = s0

        for t in range(1, steps + 1):
            z = np.random.standard_normal(num_paths)
            paths[:, t] = paths[:, t - 1] * np.exp(drift + sigma * np.sqrt(dt) * z)

        return paths

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        df: pd.DataFrame,
        steps: int = 15,
        num_paths: int = 150,
    ) -> dict:
        """
        Executa a simulação completa de Merton Jump-Diffusion.

        Etapas:
            1. Validação e preparação dos dados de entrada.
            2. Calibração dos parâmetros difusivos (μ, σ) e de salto (λ, μ_J, σ_J).
            3. Simulação Monte Carlo com saltos (Merton) e sem saltos (GBM).
            4. Cálculo de percentis (5%, 50%, 95%) e métricas resumo.

        Parâmetros:
            df: DataFrame com colunas obrigatórias ['close'] e opcionais
                ['timestamp', 'anomaly']. Mínimo de 10 observações.
            steps: número de passos futuros a simular (default: 15).
            num_paths: número de trajetórias Monte Carlo (default: 150).

        Retorna:
            Dicionário com:
                - 'timestamps': lista de timestamps futuros projetados.
                - 'paths': matriz de trajetórias simuladas (lista de listas).
                - 'percentile_5': percentil 5% por passo (lista).
                - 'percentile_50': mediana por passo (lista).
                - 'percentile_95': percentil 95% por passo (lista).
                - 'gbm_only_median': mediana do GBM puro (lista).
                - 'lambda_jumps': frequência de saltos calibrada (float).
                - 'mu_jump': média do tamanho de salto calibrada (float).
                - 'sigma_jump': desvio-padrão do salto calibrado (float).
                - 'avg_jumps_per_path': média de saltos por trajetória (float).

        Raises:
            ValueError: se os dados de entrada forem insuficientes ou inválidos.
        """
        # ---- Validação ----
        if df is None or df.empty or len(df) < 10:
            raise ValueError(
                "Dados insuficientes para simulação Merton Jump-Diffusion. "
                "São necessárias pelo menos 10 observações."
            )

        if "close" not in df.columns:
            raise ValueError(
                "Coluna 'close' é obrigatória no DataFrame de entrada."
            )

        logger.info(
            "Iniciando simulação Merton Jump-Diffusion: %d obs, %d passos, %d trajetórias.",
            len(df), steps, num_paths,
        )

        prices = df["close"].dropna().values.astype(float)

        if len(prices) < 10:
            raise ValueError(
                "Menos de 10 preços válidos (não-nulos) disponíveis."
            )

        # ---- Calibração ----
        log_returns = self._compute_log_returns(prices)
        mu, sigma = self._calibrate_diffusion(log_returns)

        anomaly_flags = None
        if "anomaly" in df.columns:
            anomaly_flags = df["anomaly"].dropna().values

        lambda_jumps, mu_jump, sigma_jump = self._calibrate_jumps(
            log_returns, anomaly_flags, sigma
        )

        # ---- Simulação Merton (com saltos) ----
        s0 = float(prices[-1])
        merton_paths = self._simulate_paths(
            s0=s0,
            mu=mu,
            sigma=sigma,
            lambda_jumps=lambda_jumps,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
            steps=steps,
            num_paths=num_paths,
        )

        # ---- Simulação GBM pura (sem saltos) ----
        gbm_paths = self._simulate_gbm_paths(
            s0=s0,
            mu=mu,
            sigma=sigma,
            steps=steps,
            num_paths=num_paths,
        )

        # ---- Percentis (excluindo a coluna 0 = preço atual) ----
        future_merton = merton_paths[:, 1:]  # (num_paths, steps)
        future_gbm = gbm_paths[:, 1:]

        percentile_5 = np.percentile(future_merton, 5, axis=0).tolist()
        percentile_50 = np.percentile(future_merton, 50, axis=0).tolist()
        percentile_95 = np.percentile(future_merton, 95, axis=0).tolist()
        gbm_only_median = np.percentile(future_gbm, 50, axis=0).tolist()

        # ---- Timestamps futuros ----
        timestamps = self._generate_future_timestamps(df, steps)

        # ---- Média de saltos por trajetória (estimativa analítica) ----
        avg_jumps_per_path = float(lambda_jumps * steps)

        # ---- Preparar matriz de trajetórias como lista de listas ----
        paths_as_lists = merton_paths.tolist()

        result = {
            "timestamps": timestamps,
            "paths": paths_as_lists,
            "percentile_5": percentile_5,
            "percentile_50": percentile_50,
            "percentile_95": percentile_95,
            "gbm_only_median": gbm_only_median,
            "lambda_jumps": lambda_jumps,
            "mu_jump": mu_jump,
            "sigma_jump": sigma_jump,
            "avg_jumps_per_path": avg_jumps_per_path,
        }

        logger.info(
            "Simulação concluída. λ=%.4f, μ_J=%.6f, σ_J=%.6f, "
            "saltos médios/traj=%.2f",
            lambda_jumps, mu_jump, sigma_jump, avg_jumps_per_path,
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
        extrapola com o intervalo mediano. Caso contrário, gera índices
        inteiros sequenciais como fallback.

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
