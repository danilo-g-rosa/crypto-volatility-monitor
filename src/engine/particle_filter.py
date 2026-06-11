"""
Particle Filter (Sequential Monte Carlo) para estimação de volatilidade latente.

Este módulo implementa um filtro de partículas Bootstrap para estimar o regime
de volatilidade latente de ativos criptográficos. O modelo assume que a
log-volatilidade segue um processo autorregressivo de primeira ordem (AR(1))
e que os retornos observados são normalmente distribuídos condicionados à
volatilidade latente.

Modelo Matemático:
    Estado latente (transição):
        ln(σ_t) = φ · ln(σ_{t-1}) + η_t,  onde η_t ~ N(0, σ_η²)

    Observação:
        r_t ~ N(0, σ_t²)

    onde:
        - σ_t é a volatilidade latente no tempo t
        - φ é o parâmetro de persistência (calibrado via autocorrelação dos retornos absolutos)
        - σ_η é o desvio padrão do ruído de transição (calibrado via variância da log-volatilidade)
        - r_t são os log-retornos observados

    Reamostragem sistemática é ativada quando o tamanho efetivo da amostra (ESS)
    cai abaixo de N/2, onde N é o número de partículas.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ParticleFilterEngine:
    """
    Motor de Filtro de Partículas Bootstrap para estimação de regimes de volatilidade.

    O filtro de partículas (Sequential Monte Carlo) aproxima a distribuição posterior
    da volatilidade latente usando um conjunto de partículas ponderadas. Cada partícula
    representa um possível estado de log-volatilidade, e seus pesos são atualizados
    a cada passo temporal com base na verossimilhança das observações.

    Atributos:
        phi (float): Parâmetro de persistência do processo AR(1) de log-volatilidade.
        sigma_eta (float): Desvio padrão do ruído de transição de estado.
    """

    def __init__(self) -> None:
        """Inicializa o motor do filtro de partículas."""
        self.phi: float = 0.98
        self.sigma_eta: float = 0.1

    def _compute_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcula os log-retornos a partir de uma série de preços.

        Fórmula:
            r_t = ln(P_t / P_{t-1})

        Parâmetros:
            prices (np.ndarray): Vetor de preços de fechamento.

        Retorna:
            np.ndarray: Vetor de log-retornos (comprimento = len(prices) - 1).
        """
        prices = prices.astype(np.float64)
        # Evitar log(0) ou log(negativo) substituindo por valor anterior
        prices = np.where(prices <= 0, np.nan, prices)
        prices = pd.Series(prices).ffill().bfill().values
        log_returns = np.diff(np.log(prices))
        # Substituir NaN/Inf residuais por 0
        log_returns = np.where(np.isfinite(log_returns), log_returns, 0.0)
        return log_returns

    def _calibrate_parameters(self, log_returns: np.ndarray) -> Tuple[float, float]:
        """
        Calibra os parâmetros φ e σ_η a partir dos dados observados.

        Método de calibração:
            - φ (persistência): estimado pela autocorrelação de lag-1 dos retornos
              absolutos, que serve como proxy para a persistência da volatilidade.
              Limitado ao intervalo [0.9, 0.999] para garantir estacionariedade.

            - σ_η (ruído de transição): estimado pelo desvio padrão das diferenças
              do logaritmo dos retornos absolutos, representando a inovação na
              log-volatilidade. Limitado ao intervalo [0.01, 0.5].

        Parâmetros:
            log_returns (np.ndarray): Vetor de log-retornos.

        Retorna:
            Tuple[float, float]: (φ, σ_η) calibrados.
        """
        abs_returns = np.abs(log_returns)
        abs_returns = np.where(abs_returns < 1e-15, 1e-15, abs_returns)

        # Estimar φ via autocorrelação de lag-1 dos retornos absolutos
        if len(abs_returns) > 2:
            mean_abs = np.mean(abs_returns)
            centered = abs_returns - mean_abs
            var = np.var(abs_returns)
            if var > 1e-20:
                autocov = np.mean(centered[:-1] * centered[1:])
                phi = np.clip(autocov / var, 0.9, 0.999)
            else:
                phi = 0.98
        else:
            phi = 0.98

        # Estimar σ_η via variância das diferenças de ln(|r_t|)
        log_abs = np.log(abs_returns)
        if len(log_abs) > 2:
            diff_log_abs = np.diff(log_abs)
            diff_log_abs = diff_log_abs[np.isfinite(diff_log_abs)]
            if len(diff_log_abs) > 1:
                sigma_eta = np.clip(np.std(diff_log_abs), 0.01, 0.5)
            else:
                sigma_eta = 0.1
        else:
            sigma_eta = 0.1

        logger.info(f"Parâmetros calibrados: φ={phi:.4f}, σ_η={sigma_eta:.4f}")
        return phi, sigma_eta

    def _systematic_resample(self, weights: np.ndarray, num_particles: int) -> np.ndarray:
        """
        Executa reamostragem sistemática das partículas.

        A reamostragem sistemática é mais eficiente que a multinomial pois
        utiliza um único número aleatório uniforme para selecionar todas
        as partículas, reduzindo a variância do processo de reamostragem.

        Algoritmo:
            1. Calcular a função de distribuição acumulada (CDF) dos pesos.
            2. Gerar um ponto de partida aleatório u_0 ~ U(0, 1/N).
            3. Para cada partícula i, o ponto de amostragem é u_i = u_0 + i/N.
            4. Selecionar a partícula cujo intervalo da CDF contém u_i.

        Parâmetros:
            weights (np.ndarray): Pesos normalizados das partículas.
            num_particles (int): Número de partículas.

        Retorna:
            np.ndarray: Índices das partículas reamostradas.
        """
        cdf = np.cumsum(weights)
        # Garantir que CDF termine em exatamente 1.0
        cdf[-1] = 1.0

        u0 = np.random.uniform(0, 1.0 / num_particles)
        u = u0 + np.arange(num_particles) / num_particles

        indices = np.searchsorted(cdf, u)
        # Clamp para evitar índice fora dos limites
        indices = np.clip(indices, 0, num_particles - 1)
        return indices

    def _compute_ess(self, weights: np.ndarray) -> float:
        """
        Calcula o Tamanho Efetivo da Amostra (Effective Sample Size - ESS).

        Fórmula:
            ESS = 1 / Σ(w_i²)

        O ESS mede a degeneração dos pesos. Quando todos os pesos são iguais
        (1/N), ESS = N. Quando um peso domina, ESS → 1. A reamostragem é
        necessária quando ESS < N/2 para evitar degeneração severa.

        Parâmetros:
            weights (np.ndarray): Pesos normalizados das partículas.

        Retorna:
            float: O tamanho efetivo da amostra.
        """
        sum_sq = np.sum(weights ** 2)
        if sum_sq < 1e-30:
            return 0.0
        return 1.0 / sum_sq

    def run_filter(self, df: pd.DataFrame, num_particles: int = 500) -> dict:
        """
        Executa o Filtro de Partículas Bootstrap para estimação de volatilidade latente.

        O algoritmo procede da seguinte forma para cada passo temporal t:
            1. **Propagação**: Cada partícula é propagada pelo modelo de transição:
               ln(σ_t^(i)) = φ · ln(σ_{t-1}^(i)) + η_t^(i), η_t^(i) ~ N(0, σ_η²)

            2. **Atualização de pesos**: Os pesos são atualizados pela verossimilhança
               da observação dado o estado de cada partícula:
               w_t^(i) ∝ w_{t-1}^(i) · p(r_t | σ_t^(i))
               onde p(r_t | σ_t) = N(r_t | 0, σ_t²)

            3. **Normalização**: Os pesos são normalizados para somar 1.

            4. **Reamostragem**: Se ESS < N/2, aplica-se reamostragem sistemática
               para redistribuir as partículas em regiões de alta probabilidade.

            5. **Estimação**: A média posterior da volatilidade é:
               E[σ_t | r_{1:t}] ≈ Σ w_t^(i) · σ_t^(i)

        Parâmetros:
            df (pd.DataFrame): DataFrame com colunas obrigatórias: 'close' e 'timestamp'.
                               Deve conter ao menos 10 linhas.
            num_particles (int): Número de partículas a utilizar (padrão: 500).
                                 Mais partículas = melhor aproximação, mas maior custo.

        Retorna:
            dict: Dicionário com as seguintes chaves:
                - 'timestamps': lista de timestamps correspondentes aos retornos
                - 'volatility_estimate': lista com a média posterior da volatilidade
                - 'volatility_upper': lista com o percentil 95 das partículas
                - 'volatility_lower': lista com o percentil 5 das partículas
                - 'regime_probability': lista com P(regime de alta vol) em cada passo
                - 'effective_sample_size': lista com o ESS ao longo do tempo
                - 'current_regime': str ('HIGH_VOL' ou 'LOW_VOL')
                - 'current_regime_prob': float com a probabilidade do regime atual

        Levanta:
            ValueError: Se o DataFrame for nulo, vazio, ou com menos de 10 linhas.
        """
        # ── Validação de entrada ──────────────────────────────────────────
        if df is None or df.empty or len(df) < 10:
            raise ValueError(
                "Dados insuficientes para o filtro de partículas. "
                "São necessárias ao menos 10 observações."
            )

        if "close" not in df.columns:
            raise ValueError("Coluna 'close' não encontrada no DataFrame.")

        np.random.seed(42)

        # ── Preparar dados ────────────────────────────────────────────────
        prices = df["close"].values.astype(np.float64)
        log_returns = self._compute_log_returns(prices)
        T = len(log_returns)

        if T < 5:
            raise ValueError(
                "Dados insuficientes após cálculo de retornos. "
                f"Apenas {T} retornos disponíveis (mínimo: 5)."
            )

        # Timestamps correspondentes aos retornos (deslocamento de 1)
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].iloc[1:].tolist()
        else:
            timestamps = list(range(T))

        logger.info(
            f"Iniciando filtro de partículas: {T} observações, "
            f"{num_particles} partículas."
        )

        # ── Calibrar parâmetros ───────────────────────────────────────────
        self.phi, self.sigma_eta = self._calibrate_parameters(log_returns)

        # ── Calcular limiar para regime de alta volatilidade ──────────────
        # Mediana da volatilidade histórica (|retornos|) como limiar
        hist_vol = np.abs(log_returns)
        vol_median = np.median(hist_vol)
        logger.info(f"Limiar de regime (mediana da vol histórica): {vol_median:.6f}")

        # ── Inicializar partículas ────────────────────────────────────────
        # Inicializar log-volatilidade com base na volatilidade empírica inicial
        initial_vol = np.std(log_returns[:min(20, T)])
        if initial_vol < 1e-15:
            initial_vol = 1e-4

        initial_log_vol = np.log(initial_vol)
        # Dispersão inicial das partículas ao redor da estimativa empírica
        particles = initial_log_vol + self.sigma_eta * np.random.randn(num_particles)
        weights = np.ones(num_particles) / num_particles

        # ── Estruturas para armazenar resultados ──────────────────────────
        volatility_estimate: List[float] = []
        volatility_upper: List[float] = []
        volatility_lower: List[float] = []
        regime_probability: List[float] = []
        ess_history: List[float] = []

        # ── Loop principal do filtro ──────────────────────────────────────
        for t in range(T):
            r_t = log_returns[t]

            # 1. Propagação (transição de estado)
            noise = self.sigma_eta * np.random.randn(num_particles)
            particles = self.phi * particles + noise

            # Converter log-volatilidade para volatilidade
            # Clamp para evitar overflow numérico: exp(x) com x em [-20, 5]
            clamped = np.clip(particles, -20.0, 5.0)
            vol_particles = np.exp(clamped)

            # 2. Atualização de pesos via verossimilhança
            # p(r_t | σ_t) = (1/(σ_t√(2π))) · exp(-r_t² / (2σ_t²))
            # Em log: -ln(σ_t) - 0.5·ln(2π) - r_t²/(2σ_t²)
            # Usar σ_t com piso para evitar divisão por zero
            vol_safe = np.maximum(vol_particles, 1e-15)
            log_likelihood = (
                -np.log(vol_safe)
                - 0.5 * np.log(2.0 * np.pi)
                - 0.5 * (r_t ** 2) / (vol_safe ** 2)
            )

            # Estabilizar numericamente subtraindo o máximo
            log_likelihood -= np.max(log_likelihood)
            likelihood = np.exp(log_likelihood)

            # Atualizar pesos
            raw_weights = weights * likelihood
            weight_sum = np.sum(raw_weights)

            if weight_sum < 1e-30:
                # Degeneração total: resetar pesos uniformes
                logger.warning(
                    f"Passo {t}: degeneração total dos pesos. Resetando."
                )
                weights = np.ones(num_particles) / num_particles
            else:
                weights = raw_weights / weight_sum

            # 3. Calcular estimativas posteriores
            # Média ponderada da volatilidade
            mean_vol = np.sum(weights * vol_particles)

            # Percentis via reamostragem ponderada
            sorted_indices = np.argsort(vol_particles)
            sorted_vol = vol_particles[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cum_weights = np.cumsum(sorted_weights)

            idx_05 = np.searchsorted(cum_weights, 0.05)
            idx_95 = np.searchsorted(cum_weights, 0.95)
            idx_05 = min(idx_05, num_particles - 1)
            idx_95 = min(idx_95, num_particles - 1)

            lower = sorted_vol[idx_05]
            upper = sorted_vol[idx_95]

            # Probabilidade de regime de alta volatilidade
            # P(σ_t > mediana_histórica) = Σ w_i · I(σ_t^(i) > mediana)
            high_vol_mask = vol_particles > vol_median
            p_high = np.sum(weights * high_vol_mask)

            volatility_estimate.append(float(mean_vol))
            volatility_upper.append(float(upper))
            volatility_lower.append(float(lower))
            regime_probability.append(float(p_high))

            # 4. Calcular ESS e reamostrar se necessário
            ess = self._compute_ess(weights)
            ess_history.append(float(ess))

            if ess < num_particles / 2.0:
                indices = self._systematic_resample(weights, num_particles)
                particles = particles[indices]
                weights = np.ones(num_particles) / num_particles

        # ── Determinar regime atual ───────────────────────────────────────
        current_regime_prob = regime_probability[-1] if regime_probability else 0.5
        current_regime = "HIGH_VOL" if current_regime_prob > 0.5 else "LOW_VOL"

        logger.info(
            f"Filtro concluído. Regime atual: {current_regime} "
            f"(P={current_regime_prob:.3f}). "
            f"Volatilidade média final: {volatility_estimate[-1]:.6f}"
        )

        return {
            "timestamps": timestamps,
            "volatility_estimate": volatility_estimate,
            "volatility_upper": volatility_upper,
            "volatility_lower": volatility_lower,
            "regime_probability": regime_probability,
            "effective_sample_size": ess_history,
            "current_regime": current_regime,
            "current_regime_prob": float(current_regime_prob),
        }
