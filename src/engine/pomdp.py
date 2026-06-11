"""
pomdp.py - Motor POMDP (Processo de Decisão de Markov Parcialmente Observável)
para inferência de estado de mercado e recomendação de ação.

Matemática
----------
Um POMDP é definido pela tupla (S, A, O, T, Ω, R, b₀) onde:

  S = {BULL, BEAR, NEUTRAL}       — estados ocultos do mercado
  A = {COMPRAR, VENDER, MANTER}   — ações possíveis
  O = (o_ret, o_vol, o_zscore)    — observação composta discretizada
  T(s'|s) = matriz de transição    — calibrada empiricamente
  Ω(o|s)  = modelo de observação   — distribuição empírica das observações dado o estado
  R(s, a) = modelo de recompensa   — recompensa esperada por ação em cada estado
  b₀      = crença inicial uniforme [1/3, 1/3, 1/3]

Atualização de crença (Algoritmo Forward):
  b'(s') ∝ Ω(o|s') · Σ_s T(s'|s) · b(s)
  Normaliza-se para que Σ_s' b'(s') = 1.

A recomendação final é baseada na crença terminal:
  - P(BULL) > 0.5  → COMPRAR
  - P(BEAR) > 0.5  → VENDER
  - caso contrário  → MANTER
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
STATES: List[str] = ["BULL", "BEAR", "NEUTRAL"]
ACTIONS: List[str] = ["COMPRAR", "VENDER", "MANTER"]

RETURN_CATEGORIES: List[str] = ["HIGH_POS", "LOW_POS", "FLAT", "LOW_NEG", "HIGH_NEG"]
VOLUME_CATEGORIES: List[str] = ["HIGH", "NORMAL", "LOW"]
ZSCORE_CATEGORIES: List[str] = ["EXTREME_POS", "POS", "FLAT", "NEG", "EXTREME_NEG"]


class POMDPEngine:
    """
    Motor de inferência baseado em POMDP simplificado para o mercado de
    criptomoedas.

    O modelo trata o verdadeiro regime de mercado (BULL / BEAR / NEUTRAL)
    como um estado oculto e utiliza observações discretizadas de retorno,
    variação de volume e z-score para atualizar iterativamente um vetor
    de crença via o algoritmo forward.

    Parâmetros calibrados empiricamente a partir dos dados fornecidos.
    """

    # Modelo de recompensa R(s, a): recompensa esperada de tomar ação *a*
    # quando o verdadeiro estado é *s*.  Valores em unidades abstratas.
    _REWARD_TABLE: Dict[str, Dict[str, float]] = {
        "BULL":    {"COMPRAR":  1.0, "VENDER": -1.0, "MANTER":  0.3},
        "BEAR":    {"COMPRAR": -1.0, "VENDER":  1.0, "MANTER":  0.1},
        "NEUTRAL": {"COMPRAR":  0.0, "VENDER":  0.0, "MANTER":  0.5},
    }

    # ------------------------------------------------------------------ #
    #  Discretização de observações                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _discretize_return(ret: float) -> str:
        """
        Discretiza o retorno log em cinco categorias.

        Limiares (calibrados para cripto, volatilidade elevada):
          |ret| > 2 %  → HIGH_POS / HIGH_NEG
          |ret| > 0.5% → LOW_POS  / LOW_NEG
          caso contrário → FLAT
        """
        if ret > 0.02:
            return "HIGH_POS"
        elif ret > 0.005:
            return "LOW_POS"
        elif ret > -0.005:
            return "FLAT"
        elif ret > -0.02:
            return "LOW_NEG"
        else:
            return "HIGH_NEG"

    @staticmethod
    def _discretize_volume(vol_change: float) -> str:
        """
        Discretiza a variação percentual de volume em três categorias.

        Limiares:
          vol_change > 0.5  → HIGH  (aumento > 50 %)
          vol_change < -0.3 → LOW   (queda > 30 %)
          caso contrário    → NORMAL
        """
        if vol_change > 0.5:
            return "HIGH"
        elif vol_change < -0.3:
            return "LOW"
        else:
            return "NORMAL"

    @staticmethod
    def _discretize_zscore(z: float) -> str:
        """
        Discretiza o z-score em cinco categorias.

        Limiares:
          |z| > 2.0 → EXTREME_POS / EXTREME_NEG
          |z| > 0.5 → POS / NEG
          caso contrário → FLAT
        """
        if z > 2.0:
            return "EXTREME_POS"
        elif z > 0.5:
            return "POS"
        elif z > -0.5:
            return "FLAT"
        elif z > -2.0:
            return "NEG"
        else:
            return "EXTREME_NEG"

    # ------------------------------------------------------------------ #
    #  Classificação do regime "verdadeiro" para calibração               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_regime(returns: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Classifica cada ponto temporal em um regime de mercado usando uma
        média móvel dos retornos.

        Lógica:
          - média_móvel > +0.003  → 0 (BULL)
          - média_móvel < -0.003  → 1 (BEAR)
          - caso contrário        → 2 (NEUTRAL)

        Retorna um array de inteiros com shape (N,).
        """
        if len(returns) < window:
            window = max(1, len(returns) // 2)

        # Média móvel simples usando convolução
        kernel = np.ones(window) / window
        smoothed = np.convolve(returns, kernel, mode="same")

        regimes = np.full(len(returns), 2, dtype=int)  # default NEUTRAL
        regimes[smoothed > 0.003] = 0   # BULL
        regimes[smoothed < -0.003] = 1  # BEAR
        return regimes

    # ------------------------------------------------------------------ #
    #  Calibração do modelo de transição T(s'|s)                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_transition_matrix(regimes: np.ndarray,
                                    n_states: int = 3,
                                    smoothing: float = 1.0) -> np.ndarray:
        """
        Estima a matriz de transição T(s'|s) via contagem de bigramas
        com suavização de Laplace.

        T[i, j] = (count(i→j) + α) / (count(i→*) + α·|S|)

        Parâmetros
        ----------
        regimes : array de inteiros (0=BULL, 1=BEAR, 2=NEUTRAL)
        n_states : número de estados
        smoothing : parâmetro α de suavização de Laplace

        Retorna
        -------
        T : np.ndarray shape (n_states, n_states), linhas somam 1
        """
        counts = np.full((n_states, n_states), smoothing)
        for i in range(len(regimes) - 1):
            s_from = regimes[i]
            s_to = regimes[i + 1]
            if 0 <= s_from < n_states and 0 <= s_to < n_states:
                counts[s_from, s_to] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        T = counts / row_sums
        return T

    # ------------------------------------------------------------------ #
    #  Calibração do modelo de observação Ω(o|s)                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_observation_model(
        regimes: np.ndarray,
        obs_ret: List[str],
        obs_vol: List[str],
        obs_z: List[str],
        smoothing: float = 1.0,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Estima Ω(o|s) — a distribuição empírica de cada componente de
        observação condicionada ao estado oculto, com suavização de Laplace.

        Retorna um dicionário aninhado:
          obs_model[state][obs_type][category] = probabilidade

        onde obs_type ∈ {'return', 'volume', 'zscore'}.
        """
        category_sets = {
            "return": RETURN_CATEGORIES,
            "volume": VOLUME_CATEGORIES,
            "zscore": ZSCORE_CATEGORIES,
        }

        obs_data = {
            "return": obs_ret,
            "volume": obs_vol,
            "zscore": obs_z,
        }

        obs_model: Dict[str, Dict[str, Dict[str, float]]] = {}

        for s_idx, state in enumerate(STATES):
            obs_model[state] = {}
            mask = regimes == s_idx

            for obs_type, categories in category_sets.items():
                counts = {cat: smoothing for cat in categories}
                obs_values = obs_data[obs_type]

                for i, is_state in enumerate(mask):
                    if is_state and i < len(obs_values):
                        cat = obs_values[i]
                        if cat in counts:
                            counts[cat] += 1.0

                total = sum(counts.values())
                total = total if total > 0 else 1.0
                obs_model[state][obs_type] = {
                    cat: c / total for cat, c in counts.items()
                }

        return obs_model

    # ------------------------------------------------------------------ #
    #  Probabilidade de observação composta P(o|s)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _observation_probability(
        obs_model: Dict[str, Dict[str, Dict[str, float]]],
        state: str,
        o_ret: str,
        o_vol: str,
        o_z: str,
    ) -> float:
        """
        Calcula a probabilidade conjunta da observação composta (o_ret,
        o_vol, o_z) dado o estado, assumindo independência condicional:

          P(o|s) = P(o_ret|s) · P(o_vol|s) · P(o_z|s)

        Utiliza um piso mínimo de 1e-12 para estabilidade numérica.
        """
        p_ret = obs_model[state]["return"].get(o_ret, 1e-6)
        p_vol = obs_model[state]["volume"].get(o_vol, 1e-6)
        p_z = obs_model[state]["zscore"].get(o_z, 1e-6)
        return max(p_ret * p_vol * p_z, 1e-12)

    # ------------------------------------------------------------------ #
    #  Atualização de crença (Forward step)                              #
    # ------------------------------------------------------------------ #

    def _belief_update(
        self,
        belief: np.ndarray,
        transition: np.ndarray,
        obs_model: Dict[str, Dict[str, Dict[str, float]]],
        o_ret: str,
        o_vol: str,
        o_z: str,
    ) -> np.ndarray:
        """
        Executa um passo do algoritmo forward para atualizar a crença.

        b'(s') ∝ Ω(o|s') · Σ_s T(s'|s) · b(s)

        Parâmetros
        ----------
        belief : vetor de crença atual, shape (3,)
        transition : matriz de transição T, shape (3, 3)
        obs_model : modelo de observação calibrado
        o_ret, o_vol, o_z : observações discretizadas atuais

        Retorna
        -------
        new_belief : vetor de crença atualizado e normalizado, shape (3,)
        """
        n = len(STATES)
        new_belief = np.zeros(n)

        for s_prime_idx in range(n):
            # Predição: Σ_s T(s'|s) · b(s)
            predicted = 0.0
            for s_idx in range(n):
                predicted += transition[s_idx, s_prime_idx] * belief[s_idx]

            # Atualização com observação
            obs_prob = self._observation_probability(
                obs_model, STATES[s_prime_idx], o_ret, o_vol, o_z
            )
            new_belief[s_prime_idx] = obs_prob * predicted

        # Normalização
        total = new_belief.sum()
        if total > 0:
            new_belief /= total
        else:
            # Fallback para crença uniforme se degenerado
            new_belief = np.ones(n) / n
            logger.warning("Crença degenerada detectada; revertendo para uniforme.")

        return new_belief

    # ------------------------------------------------------------------ #
    #  Recomendação de ação                                              #
    # ------------------------------------------------------------------ #

    def _recommend_action(
        self, belief: np.ndarray
    ) -> Tuple[str, float, str, Dict[str, float]]:
        """
        Determina a ação recomendada com base no vetor de crença final.

        Regras:
          - P(BULL) > 0.5  → COMPRAR
          - P(BEAR) > 0.5  → VENDER
          - caso contrário  → MANTER

        Também calcula o valor esperado de cada ação:
          V(a) = Σ_s b(s) · R(s, a)

        Retorna
        -------
        action : str — ação recomendada
        confidence : float — probabilidade máxima do vetor de crença
        dominant_state : str — estado com maior probabilidade
        reward_estimates : dict — valor esperado de cada ação
        """
        # Estado dominante
        dominant_idx = int(np.argmax(belief))
        dominant_state = STATES[dominant_idx]
        confidence = float(belief[dominant_idx])

        # Ação por regra de limiar
        if belief[0] > 0.5:
            action = "COMPRAR"
        elif belief[1] > 0.5:
            action = "VENDER"
        else:
            action = "MANTER"

        # Valor esperado de cada ação
        reward_estimates: Dict[str, float] = {}
        for a in ACTIONS:
            v = 0.0
            for s_idx, state in enumerate(STATES):
                v += belief[s_idx] * self._REWARD_TABLE[state][a]
            reward_estimates[a] = round(float(v), 6)

        return action, confidence, dominant_state, reward_estimates

    # ------------------------------------------------------------------ #
    #  Método principal                                                   #
    # ------------------------------------------------------------------ #

    def run_analysis(self, df: pd.DataFrame) -> dict:
        """
        Executa a análise POMDP completa sobre o DataFrame de mercado.

        Etapas
        ------
        1. Validação e preparação dos dados.
        2. Cálculo de retornos logarítmicos e variação de volume.
        3. Discretização das observações (retorno, volume, z-score).
        4. Classificação de regimes históricos para calibração.
        5. Estimação empírica da matriz de transição T(s'|s).
        6. Estimação empírica do modelo de observação Ω(o|s).
        7. Execução do algoritmo forward para atualização de crença.
        8. Recomendação de ação baseada na crença terminal.

        Parâmetros
        ----------
        df : pd.DataFrame
            Deve conter pelo menos as colunas 'close' e 'volume'.
            Opcionalmente 'timestamp' e 'z_score'.

        Retorna
        -------
        dict com as chaves:
            - belief_history : lista de dicts {BULL, BEAR, NEUTRAL}
            - current_belief : dict {BULL, BEAR, NEUTRAL}
            - timestamps : lista de timestamps
            - recommended_action : str
            - action_confidence : float
            - dominant_state : str
            - state_history : lista de str
            - transition_matrix : dict (3×3)
            - reward_estimates : dict {COMPRAR, VENDER, MANTER}
        """
        # -------- 1. Validação --------
        if df is None or df.empty:
            raise ValueError(
                "Dados insuficientes: DataFrame é None ou vazio."
            )

        required_cols = {"close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Colunas obrigatórias ausentes: {missing}. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        if len(df) < 10:
            raise ValueError(
                f"Dados insuficientes: necessário pelo menos 10 linhas, "
                f"recebido {len(df)}."
            )

        logger.info(
            "Iniciando análise POMDP com %d pontos de dados.", len(df)
        )

        # Garantir reprodutibilidade
        np.random.seed(42)

        # -------- 2. Preparação dos dados --------
        close = df["close"].astype(float).values

        # Retornos logarítmicos (primeiro valor = 0)
        log_returns = np.zeros(len(close))
        log_returns[1:] = np.diff(np.log(np.where(close > 0, close, 1e-10)))

        # Variação percentual de volume
        if "volume" in df.columns:
            volume = df["volume"].astype(float).values
            vol_change = np.zeros(len(volume))
            with np.errstate(divide="ignore", invalid="ignore"):
                shifted = np.roll(volume, 1)
                shifted[0] = volume[0] if volume[0] != 0 else 1.0
                safe_shifted = np.where(shifted == 0, 1.0, shifted)
                vol_change[1:] = (volume[1:] - shifted[1:]) / safe_shifted[1:]
        else:
            logger.warning(
                "Coluna 'volume' ausente; usando variação de volume = 0."
            )
            vol_change = np.zeros(len(close))

        # Z-Score
        if "z_score" in df.columns:
            z_scores = df["z_score"].astype(float).fillna(0.0).values
        else:
            # Calcular z-score a partir dos retornos
            logger.info(
                "Coluna 'z_score' ausente; calculando a partir dos retornos."
            )
            mean_ret = np.mean(log_returns)
            std_ret = np.std(log_returns)
            if std_ret > 0:
                z_scores = (log_returns - mean_ret) / std_ret
            else:
                z_scores = np.zeros(len(log_returns))

        # Timestamps
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].tolist()
        else:
            timestamps = list(range(len(df)))

        # -------- 3. Discretização --------
        obs_ret = [self._discretize_return(r) for r in log_returns]
        obs_vol = [self._discretize_volume(v) for v in vol_change]
        obs_z = [self._discretize_zscore(z) for z in z_scores]

        logger.debug(
            "Distribuição de retornos discretizados: %s",
            {cat: obs_ret.count(cat) for cat in RETURN_CATEGORIES},
        )

        # -------- 4. Classificação de regimes --------
        window = min(10, max(2, len(log_returns) // 10))
        regimes = self._classify_regime(log_returns, window=window)

        regime_counts = {
            STATES[i]: int(np.sum(regimes == i)) for i in range(3)
        }
        logger.info("Regimes detectados: %s", regime_counts)

        # -------- 5. Matriz de transição --------
        T = self._estimate_transition_matrix(regimes, n_states=3, smoothing=1.0)
        logger.info(
            "Matriz de transição estimada:\n%s",
            np.array2string(T, precision=4),
        )

        # -------- 6. Modelo de observação --------
        obs_model = self._estimate_observation_model(
            regimes, obs_ret, obs_vol, obs_z, smoothing=1.0
        )

        # -------- 7. Algoritmo Forward --------
        belief = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        belief_history: List[Dict[str, float]] = []
        state_history: List[str] = []

        for t in range(len(df)):
            belief = self._belief_update(
                belief, T, obs_model, obs_ret[t], obs_vol[t], obs_z[t]
            )

            belief_dict = {
                STATES[i]: round(float(belief[i]), 6) for i in range(3)
            }
            belief_history.append(belief_dict)

            dominant_idx = int(np.argmax(belief))
            state_history.append(STATES[dominant_idx])

        # -------- 8. Recomendação --------
        action, confidence, dominant_state, reward_estimates = (
            self._recommend_action(belief)
        )

        logger.info(
            "Análise POMDP concluída — Ação: %s (confiança: %.4f), "
            "Estado dominante: %s",
            action,
            confidence,
            dominant_state,
        )

        # -------- Montagem do resultado --------
        transition_dict: Dict[str, Dict[str, float]] = {}
        for i, s_from in enumerate(STATES):
            transition_dict[s_from] = {
                s_to: round(float(T[i, j]), 6)
                for j, s_to in enumerate(STATES)
            }

        return {
            "belief_history": belief_history,
            "current_belief": belief_history[-1] if belief_history else {
                s: round(1.0 / 3.0, 6) for s in STATES
            },
            "timestamps": timestamps,
            "recommended_action": action,
            "action_confidence": round(confidence, 6),
            "dominant_state": dominant_state,
            "state_history": state_history,
            "transition_matrix": transition_dict,
            "reward_estimates": reward_estimates,
        }
