"""
Teoria de Valores Extremos (EVT) + Modelo de Mistura Gaussiana (GMM)
para análise de risco de cauda e classificação de regimes de volatilidade.

Este módulo combina duas abordagens complementares:

1. **EVT — Distribuição Generalizada de Pareto (GPD)**:
   Modela as caudas da distribuição dos retornos para estimar métricas de
   risco extremo como Value at Risk (VaR) e Expected Shortfall (CVaR).

   Modelo Matemático:
       Dado um limiar u (ex.: percentil 90 de |r_t|), os excessos y = |r_t| - u
       seguem uma distribuição GPD:

           G(y; ξ, σ) = 1 - (1 + ξ·y/σ)^{-1/ξ}    se ξ ≠ 0
           G(y; ξ, σ) = 1 - exp(-y/σ)               se ξ = 0

       onde:
           - ξ (shape): controla o peso da cauda (ξ > 0 → cauda pesada)
           - σ (scale): parâmetro de escala

       VaR e CVaR são calculados via:
           VaR_p = u + (σ/ξ) · [((n/N_u) · (1-p))^{-ξ} - 1]
           CVaR_p = VaR_p / (1-ξ) + (σ - ξ·u) / (1-ξ)

2. **GMM — Gaussian Mixture Model**:
   Classifica os retornos em regimes de volatilidade usando uma mistura
   de distribuições gaussianas, onde cada componente representa um
   regime distinto (baixa, média ou alta volatilidade).

   Modelo:
       p(r_t) = Σ_k π_k · N(r_t | μ_k, σ_k²)

   Os parâmetros são estimados via algoritmo EM (Expectation-Maximization)
   utilizando sklearn.mixture.GaussianMixture.
"""

import pandas as pd
import numpy as np
import logging
from scipy import optimize
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class EVTGMMEngine:
    """
    Motor de análise combinando Teoria de Valores Extremos (EVT) com
    Modelo de Mistura Gaussiana (GMM).

    A componente EVT fornece estimativas robustas de risco de cauda
    (VaR, CVaR) baseadas na Distribuição Generalizada de Pareto, enquanto
    a componente GMM classifica o mercado em regimes de volatilidade
    distintos para auxiliar na tomada de decisão.
    """

    def __init__(self) -> None:
        """Inicializa o motor EVT+GMM."""
        pass

    def _compute_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcula os log-retornos a partir de uma série de preços.

        Fórmula:
            r_t = ln(P_t / P_{t-1})

        Parâmetros:
            prices (np.ndarray): Vetor de preços de fechamento.

        Retorna:
            np.ndarray: Vetor de log-retornos.
        """
        prices = prices.astype(np.float64)
        prices = np.where(prices <= 0, np.nan, prices)
        prices = pd.Series(prices).ffill().bfill().values
        log_returns = np.diff(np.log(prices))
        log_returns = np.where(np.isfinite(log_returns), log_returns, 0.0)
        return log_returns

    # ──────────────────────────────────────────────────────────────────────
    #  EVT — Distribuição Generalizada de Pareto
    # ──────────────────────────────────────────────────────────────────────

    def _gpd_neg_log_likelihood(
        self, params: np.ndarray, exceedances: np.ndarray
    ) -> float:
        """
        Calcula a log-verossimilhança negativa da GPD para otimização.

        A log-verossimilhança da GPD é:
            Se ξ ≠ 0:
                ℓ(ξ, σ) = -n·ln(σ) - (1 + 1/ξ) · Σ ln(1 + ξ·y_i/σ)

            Se ξ ≈ 0 (caso exponencial):
                ℓ(σ) = -n·ln(σ) - (1/σ) · Σ y_i

        Parâmetros:
            params (np.ndarray): [ξ, σ] — parâmetros shape e scale.
            exceedances (np.ndarray): Vetor de excessos acima do limiar.

        Retorna:
            float: Negativo da log-verossimilhança (para minimização).
        """
        xi, sigma = params

        if sigma <= 0:
            return 1e10

        n = len(exceedances)

        if abs(xi) < 1e-8:
            # Caso exponencial (ξ → 0)
            nll = n * np.log(sigma) + np.sum(exceedances) / sigma
        else:
            # Caso geral
            z = 1.0 + xi * exceedances / sigma
            if np.any(z <= 0):
                return 1e10
            nll = n * np.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(z))

        if not np.isfinite(nll):
            return 1e10

        return nll

    def _fit_gpd(
        self, exceedances: np.ndarray
    ) -> Tuple[float, float]:
        """
        Ajusta a Distribuição Generalizada de Pareto (GPD) via Máxima
        Verossimilhança (MLE).

        Utiliza o método de Momentos Ponderados por Probabilidade (PWM) para
        obter estimativas iniciais, seguido de otimização numérica com
        scipy.optimize.minimize (método Nelder-Mead).

        Parâmetros:
            exceedances (np.ndarray): Excessos acima do limiar (y_i = |r_i| - u).

        Retorna:
            Tuple[float, float]: (ξ, σ) — parâmetros shape e scale estimados.
        """
        y = np.sort(exceedances)
        n = len(y)

        if n < 5:
            logger.warning(
                "Poucos excessos para ajuste GPD robusto. "
                "Usando estimativa de momentos simples."
            )
            mean_y = np.mean(y)
            var_y = np.var(y)
            if var_y > 0:
                xi_init = 0.5 * ((mean_y ** 2) / var_y - 1.0)
                sigma_init = 0.5 * mean_y * ((mean_y ** 2) / var_y + 1.0)
            else:
                xi_init = 0.1
                sigma_init = mean_y if mean_y > 0 else 0.01
            return float(np.clip(xi_init, -0.5, 2.0)), float(max(sigma_init, 1e-8))

        # Estimativas iniciais via método dos momentos
        mean_y = np.mean(y)
        var_y = np.var(y)

        if var_y > 1e-20 and mean_y > 1e-15:
            xi_init = 0.5 * ((mean_y ** 2) / var_y - 1.0)
            sigma_init = 0.5 * mean_y * ((mean_y ** 2) / var_y + 1.0)
        else:
            xi_init = 0.1
            sigma_init = mean_y if mean_y > 0 else 0.01

        xi_init = float(np.clip(xi_init, -0.5, 2.0))
        sigma_init = float(max(sigma_init, 1e-8))

        # Otimização via Nelder-Mead
        try:
            result = optimize.minimize(
                self._gpd_neg_log_likelihood,
                x0=[xi_init, sigma_init],
                args=(y,),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
            )

            if result.success or result.fun < self._gpd_neg_log_likelihood(
                np.array([xi_init, sigma_init]), y
            ):
                xi_fit, sigma_fit = result.x
                # Validar parâmetros
                if sigma_fit <= 0:
                    sigma_fit = sigma_init
                xi_fit = float(np.clip(xi_fit, -0.5, 2.0))
                sigma_fit = float(max(sigma_fit, 1e-8))
                logger.info(
                    f"GPD ajustada: ξ={xi_fit:.4f}, σ={sigma_fit:.6f} "
                    f"(convergiu={result.success})"
                )
                return xi_fit, sigma_fit
            else:
                logger.warning(
                    "Otimização GPD não melhorou. Usando estimativas iniciais."
                )
                return xi_init, sigma_init
        except Exception as e:
            logger.error(f"Erro na otimização GPD: {e}. Usando estimativas iniciais.")
            return xi_init, sigma_init

    def _compute_var_cvar(
        self,
        xi: float,
        sigma: float,
        threshold: float,
        n_total: int,
        n_exceed: int,
        confidence: float,
    ) -> Tuple[float, float]:
        """
        Calcula Value at Risk (VaR) e Conditional VaR (CVaR/Expected Shortfall)
        usando os parâmetros da GPD ajustada.

        Fórmulas:
            VaR_p = u + (σ/ξ) · [((n/N_u) · (1-p))^{-ξ} - 1]     se ξ ≠ 0
            VaR_p = u + σ · ln((n/N_u) · (1-p))                    se ξ ≈ 0

            CVaR_p = VaR_p / (1-ξ) + (σ - ξ·u) / (1-ξ)           se ξ < 1

        onde:
            - u é o limiar
            - n é o número total de observações
            - N_u é o número de excessos
            - p é o nível de confiança (ex.: 0.95)

        Parâmetros:
            xi (float): Parâmetro de forma (shape) da GPD.
            sigma (float): Parâmetro de escala (scale) da GPD.
            threshold (float): Limiar usado para definir excessos.
            n_total (int): Número total de observações.
            n_exceed (int): Número de observações acima do limiar.
            confidence (float): Nível de confiança (ex.: 0.95 ou 0.99).

        Retorna:
            Tuple[float, float]: (VaR, CVaR) no nível de confiança especificado.
        """
        if n_exceed <= 0:
            logger.warning("Nenhum excesso encontrado. Retornando VaR/CVaR = limiar.")
            return threshold, threshold

        p = confidence
        # Taxa de excesso
        rate = n_exceed / n_total

        if abs(xi) < 1e-8:
            # Caso exponencial
            var = threshold + sigma * np.log(rate / (1.0 - p))
            cvar = var + sigma
        else:
            # Caso geral
            base = (rate / (1.0 - p))
            if base > 0:
                var = threshold + (sigma / xi) * (base ** xi - 1.0)
            else:
                var = threshold

            if xi < 1.0:
                cvar = var / (1.0 - xi) + (sigma - xi * threshold) / (1.0 - xi)
            else:
                # ξ >= 1 implica média infinita; usar aproximação
                cvar = var * 1.5
                logger.warning(
                    f"ξ={xi:.3f} >= 1 indica cauda extremamente pesada. "
                    "CVaR aproximado."
                )

        # Garantir que CVaR >= VaR (por definição)
        cvar = max(cvar, var)

        return float(var), float(cvar)

    # ──────────────────────────────────────────────────────────────────────
    #  GMM — Modelo de Mistura Gaussiana
    # ──────────────────────────────────────────────────────────────────────

    def _fit_gmm(
        self, returns: np.ndarray, n_components: int = 3
    ) -> Tuple[GaussianMixture, np.ndarray]:
        """
        Ajusta um Modelo de Mistura Gaussiana (GMM) aos log-retornos.

        O GMM modela a distribuição dos retornos como uma mistura de K
        distribuições gaussianas:
            p(r) = Σ_{k=1}^{K} π_k · N(r | μ_k, σ_k²)

        onde π_k são os pesos da mistura, μ_k as médias e σ_k² as variâncias.
        Os parâmetros são estimados via algoritmo EM (Expectation-Maximization).

        Parâmetros:
            returns (np.ndarray): Vetor de log-retornos.
            n_components (int): Número de componentes da mistura (2 ou 3).

        Retorna:
            Tuple[GaussianMixture, np.ndarray]: O modelo ajustado e os rótulos
            de regime para cada observação.
        """
        X = returns.reshape(-1, 1)

        # Tentar ajustar com n_components; fallback para 2 se necessário
        for n_comp in [n_components, 2]:
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    max_iter=300,
                    n_init=5,
                    random_state=42,
                    tol=1e-6,
                )
                gmm.fit(X)

                if gmm.converged_:
                    logger.info(
                        f"GMM com {n_comp} componentes convergiu em "
                        f"{gmm.n_iter_} iterações."
                    )
                else:
                    logger.warning(
                        f"GMM com {n_comp} componentes não convergiu "
                        f"após {gmm.n_iter_} iterações."
                    )

                labels = gmm.predict(X)
                return gmm, labels

            except Exception as e:
                logger.warning(
                    f"Falha ao ajustar GMM com {n_comp} componentes: {e}"
                )
                if n_comp == n_components:
                    continue
                else:
                    raise

        # Fallback impossível de alcançar, mas por segurança
        raise RuntimeError("Falha ao ajustar GMM com qualquer configuração.")

    def _assign_regime_labels(
        self,
        gmm: GaussianMixture,
        labels: np.ndarray,
    ) -> Tuple[List[str], dict]:
        """
        Atribui rótulos de regime de volatilidade a cada componente do GMM.

        Os componentes são ordenados pela variância (desvio padrão) de cada
        componente. O componente com menor variância recebe 'Baixa Volatilidade',
        o intermediário 'Média Volatilidade' e o de maior variância
        'Alta Volatilidade'. Para 2 componentes, apenas 'Baixa' e 'Alta'.

        Parâmetros:
            gmm (GaussianMixture): Modelo GMM ajustado.
            labels (np.ndarray): Rótulos numéricos de componente por observação.

        Retorna:
            Tuple[List[str], dict]: Lista de rótulos por observação e mapeamento
            componente → rótulo.
        """
        n_comp = gmm.n_components
        means = gmm.means_.flatten()
        # Variâncias (covariance_type='full' → covariâncias shape (n, 1, 1))
        variances = gmm.covariances_.flatten()
        stds = np.sqrt(variances)

        # Ordenar componentes por desvio padrão (volatilidade)
        order = np.argsort(stds)

        if n_comp == 2:
            regime_names = ["Baixa Volatilidade", "Alta Volatilidade"]
        else:
            regime_names = [
                "Baixa Volatilidade",
                "Média Volatilidade",
                "Alta Volatilidade",
            ]

        # Mapear índice de componente original → rótulo
        component_to_label = {}
        for rank, comp_idx in enumerate(order):
            component_to_label[comp_idx] = regime_names[rank]

        # Atribuir rótulo a cada observação
        regime_labels = [component_to_label[int(l)] for l in labels]

        logger.info(
            f"Regimes atribuídos — Mapeamento: "
            + ", ".join(
                f"Comp{k}(μ={means[k]:.5f}, σ={stds[k]:.5f})→{v}"
                for k, v in sorted(component_to_label.items())
            )
        )

        return regime_labels, component_to_label

    def run_analysis(self, df: pd.DataFrame, confidence: float = 0.95) -> dict:
        """
        Executa a análise combinada EVT + GMM.

        Etapas:
            1. Calcular log-retornos a partir dos preços de fechamento.
            2. **EVT**: Definir limiar (percentil 90 de |retornos|), extrair
               excessos, ajustar GPD via MLE, calcular VaR e CVaR para os
               níveis de confiança 95% e 99%.
            3. **GMM**: Ajustar mistura de 3 gaussianas aos retornos, classificar
               cada observação em um regime de volatilidade, determinar o regime
               atual com base nos retornos mais recentes.

        Parâmetros:
            df (pd.DataFrame): DataFrame com coluna 'close' obrigatória.
                               Deve conter ao menos 10 linhas.
            confidence (float): Nível de confiança base para VaR/CVaR
                                (padrão: 0.95).

        Retorna:
            dict: Dicionário com as seguintes chaves:
                EVT:
                    - 'var_95', 'cvar_95': VaR e CVaR a 95%
                    - 'var_99', 'cvar_99': VaR e CVaR a 99%
                    - 'gpd_shape': parâmetro ξ da GPD
                    - 'gpd_scale': parâmetro σ da GPD
                    - 'tail_threshold': limiar u utilizado
                    - 'n_exceedances': número de excessos

                GMM:
                    - 'gmm_means': lista de médias por componente
                    - 'gmm_stds': lista de desvios padrão por componente
                    - 'gmm_weights': lista de pesos por componente
                    - 'gmm_regime_labels': rótulo de regime por observação
                    - 'current_regime': regime atual (str em português)
                    - 'current_regime_prob': probabilidade do regime atual

                Geral:
                    - 'returns': lista de retornos usados na análise

        Levanta:
            ValueError: Se o DataFrame for nulo, vazio, ou com menos de 10 linhas.
        """
        # ── Validação de entrada ──────────────────────────────────────────
        if df is None or df.empty or len(df) < 10:
            raise ValueError(
                "Dados insuficientes para análise EVT+GMM. "
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
                f"Dados insuficientes após cálculo de retornos ({T}). "
                "Mínimo necessário: 5."
            )

        abs_returns = np.abs(log_returns)

        logger.info(
            f"Iniciando análise EVT+GMM com {T} retornos. "
            f"Nível de confiança base: {confidence:.0%}"
        )

        # ══════════════════════════════════════════════════════════════════
        #  PARTE 1: EVT — Distribuição Generalizada de Pareto
        # ══════════════════════════════════════════════════════════════════

        # Definir limiar como percentil 90 dos retornos absolutos
        threshold = float(np.percentile(abs_returns, 90))
        if threshold < 1e-15:
            # Fallback: usar percentil 80 ou mediana
            threshold = float(np.percentile(abs_returns, 80))
            if threshold < 1e-15:
                threshold = float(np.median(abs_returns)) + 1e-10
            logger.warning(
                f"Limiar percentil 90 era ~0. Ajustado para {threshold:.8f}"
            )

        # Extrair excessos
        exceed_mask = abs_returns > threshold
        exceedances = abs_returns[exceed_mask] - threshold
        n_exceedances = len(exceedances)

        logger.info(
            f"EVT: limiar u={threshold:.6f}, "
            f"{n_exceedances}/{T} excessos ({100*n_exceedances/T:.1f}%)"
        )

        # Ajustar GPD
        if n_exceedances >= 3:
            gpd_shape, gpd_scale = self._fit_gpd(exceedances)
        else:
            logger.warning(
                f"Apenas {n_exceedances} excessos. "
                "Usando estimativa exponencial (ξ=0)."
            )
            gpd_shape = 0.0
            gpd_scale = float(np.mean(abs_returns)) if np.mean(abs_returns) > 0 else 0.01

        # Calcular VaR e CVaR em 95% e 99%
        var_95, cvar_95 = self._compute_var_cvar(
            gpd_shape, gpd_scale, threshold, T, n_exceedances, 0.95
        )
        var_99, cvar_99 = self._compute_var_cvar(
            gpd_shape, gpd_scale, threshold, T, n_exceedances, 0.99
        )

        logger.info(
            f"EVT resultados — VaR95={var_95:.6f}, CVaR95={cvar_95:.6f}, "
            f"VaR99={var_99:.6f}, CVaR99={cvar_99:.6f}"
        )

        # ══════════════════════════════════════════════════════════════════
        #  PARTE 2: GMM — Modelo de Mistura Gaussiana
        # ══════════════════════════════════════════════════════════════════

        # Determinar número de componentes com base na quantidade de dados
        n_components = 3 if T >= 50 else 2

        gmm, labels = self._fit_gmm(log_returns, n_components=n_components)

        # Extrair parâmetros do GMM
        gmm_means = gmm.means_.flatten().tolist()
        gmm_stds = np.sqrt(gmm.covariances_.flatten()).tolist()
        gmm_weights = gmm.weights_.tolist()

        # Atribuir rótulos de regime
        regime_labels, component_map = self._assign_regime_labels(gmm, labels)

        # Determinar regime atual: usar os últimos N retornos (janela recente)
        recent_window = min(10, T)
        recent_returns = log_returns[-recent_window:].reshape(-1, 1)
        recent_probs = gmm.predict_proba(recent_returns)

        # Média das probabilidades posteriores na janela recente
        avg_probs = np.mean(recent_probs, axis=0)

        # Componente mais provável nos dados recentes
        current_component = int(np.argmax(avg_probs))
        current_regime = component_map.get(
            current_component, "Média Volatilidade"
        )
        current_regime_prob = float(avg_probs[current_component])

        logger.info(
            f"GMM regime atual: '{current_regime}' "
            f"(P={current_regime_prob:.3f}), "
            f"componentes: {n_components}, "
            f"pesos: {[f'{w:.3f}' for w in gmm_weights]}"
        )

        # ── Montar resultado ──────────────────────────────────────────────
        return {
            # EVT
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "var_99": float(var_99),
            "cvar_99": float(cvar_99),
            "gpd_shape": float(gpd_shape),
            "gpd_scale": float(gpd_scale),
            "tail_threshold": float(threshold),
            "n_exceedances": int(n_exceedances),
            # GMM
            "gmm_means": gmm_means,
            "gmm_stds": gmm_stds,
            "gmm_weights": gmm_weights,
            "gmm_regime_labels": regime_labels,
            "current_regime": current_regime,
            "current_regime_prob": float(current_regime_prob),
            # Geral
            "returns": log_returns.tolist(),
        }
