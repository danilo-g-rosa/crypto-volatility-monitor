"""
Motor de Modelo SIR para Propagação de Pânico/Euforia no Mercado.

Implementa o modelo epidemiológico SIR (Suscetível-Infectado-Recuperado)
adaptado para modelar a propagação de pânico ou euforia entre participantes
do mercado de criptomoedas.

Fundamento Matemático:
    Equações diferenciais do modelo SIR:
        dS/dt = -β·S·I/N
        dI/dt =  β·S·I/N - γ·I
        dR/dt =  γ·I

    onde:
        S = participantes neutros (suscetíveis ao contágio)
        I = participantes em modo de pânico/euforia (infectados)
        R = participantes que se adaptaram/acalmaram (recuperados)
        N = S + I + R = população total (constante)
        β = taxa de contágio, calibrada pela velocidade de clustering de anomalias
        γ = taxa de recuperação, calibrada pela duração média dos clusters

    Número básico de reprodução:
        R₀ = β/γ
        R₀ > 1 → epidemia se espalha (pânico contagioso)
        R₀ < 1 → epidemia se extingue (mercado se acalma)

    Limiar de imunidade de rebanho:
        p_c = 1 - 1/R₀

    Resolução via método de Euler explícito com passo dt = 1.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SIRMarketEngine:
    """
    Motor de simulação do modelo SIR para propagação de pânico no mercado.

    Adapta o modelo epidemiológico clássico SIR para capturar como o pânico
    (ou euforia) se espalha entre participantes do mercado, usando anomalias
    detectadas nos dados de mercado como proxy para 'infecção'.
    """

    def __init__(self) -> None:
        """Inicializa o motor SIR."""
        np.random.seed(42)
        logger.info("SIRMarketEngine inicializado.")

    def _detect_anomaly_clusters(
        self, anomalies: np.ndarray
    ) -> List[List[int]]:
        """
        Detecta clusters (sequências contíguas) de anomalias nos dados.

        Um cluster é definido como uma sequência de candles consecutivos
        marcados como anomalia (valor 1 ou True). Clusters separados por
        no máximo 1 candle normal são mesclados para capturar propagação
        com pequenos intervalos.

        Parâmetros:
            anomalies: Array binário (0/1) indicando anomalias.

        Retorna:
            Lista de clusters, onde cada cluster é uma lista de índices.
        """
        clusters: List[List[int]] = []
        current_cluster: List[int] = []

        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                current_cluster.append(i)
            else:
                if current_cluster:
                    # Permitir gap de 1 candle entre anomalias no mesmo cluster
                    if i + 1 < len(anomalies) and anomalies[i + 1]:
                        current_cluster.append(i)  # Incluir o gap
                    else:
                        clusters.append(current_cluster)
                        current_cluster = []

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _calibrate_beta(
        self, anomalies: np.ndarray, total_points: int
    ) -> float:
        """
        Calibra a taxa de contágio β a partir da velocidade de clustering
        de anomalias.

        β é proporcional à frequência com que anomalias aparecem em
        sequência. Se anomalias tendem a se agrupar, β é alto (contágio
        rápido). Se aparecem isoladamente, β é baixo.

        Cálculo:
            β = (nº de anomalias em clusters / nº total de anomalias) ×
                (nº total de anomalias / nº total de pontos) × fator_escala

        Parâmetros:
            anomalies: Array binário de anomalias.
            total_points: Número total de pontos de dados.

        Retorna:
            β calibrado, limitado ao intervalo [0.05, 2.0].
        """
        total_anomalies = int(np.sum(anomalies))

        if total_anomalies < 2:
            logger.info("Poucas anomalias detectadas. Usando β mínimo.")
            return 0.05

        clusters = self._detect_anomaly_clusters(anomalies)

        # Anomalias em clusters (clusters com tamanho > 1)
        clustered_count = sum(len(c) for c in clusters if len(c) > 1)
        clustering_ratio = clustered_count / max(total_anomalies, 1)

        # Frequência de anomalias
        anomaly_frequency = total_anomalies / max(total_points, 1)

        # β: combina clustering com frequência
        # Fator de escala 5.0 para colocar β numa faixa útil
        beta = 5.0 * clustering_ratio * anomaly_frequency
        beta = max(beta, anomaly_frequency * 2.0)  # Piso baseado na frequência
        beta = float(np.clip(beta, 0.05, 2.0))

        logger.debug(
            f"Calibração β: {total_anomalies} anomalias, "
            f"{len(clusters)} clusters, clustering_ratio={clustering_ratio:.3f}, "
            f"β={beta:.4f}"
        )
        return beta

    def _calibrate_gamma(self, anomalies: np.ndarray) -> float:
        """
        Calibra a taxa de recuperação γ a partir da duração média dos
        clusters de anomalias.

        γ = 1 / duração_média_dos_clusters

        Clusters longos → γ baixo (recuperação lenta, pânico persistente)
        Clusters curtos → γ alto (recuperação rápida)

        Parâmetros:
            anomalies: Array binário de anomalias.

        Retorna:
            γ calibrado, limitado ao intervalo [0.01, 1.0].
        """
        clusters = self._detect_anomaly_clusters(anomalies)

        if not clusters:
            logger.info("Nenhum cluster detectado. Usando γ padrão.")
            return 0.3

        durations = [len(c) for c in clusters]
        mean_duration = np.mean(durations)

        if mean_duration < 1e-6:
            gamma = 1.0
        else:
            gamma = 1.0 / mean_duration

        gamma = float(np.clip(gamma, 0.01, 1.0))

        logger.debug(
            f"Calibração γ: {len(clusters)} clusters, "
            f"duração média={mean_duration:.2f}, γ={gamma:.4f}"
        )
        return gamma

    def _estimate_initial_infected(
        self, anomalies: np.ndarray, window: int = 10
    ) -> float:
        """
        Estima a proporção inicial de 'infectados' (I₀) a partir das
        anomalias recentes.

        Usa as últimas `window` observações para determinar a fração
        atual de participantes em modo de pânico/euforia.

        Parâmetros:
            anomalies: Array binário de anomalias.
            window: Janela de observações recentes.

        Retorna:
            Proporção de infectados iniciais, com mínimo de 0.01.
        """
        recent = anomalies[-window:] if len(anomalies) >= window else anomalies
        I0 = float(np.mean(recent))
        I0 = max(I0, 0.01)  # Garantir pelo menos 1% de infectados iniciais

        logger.debug(f"I₀ estimado: {I0:.4f} ({int(I0 * 100)}% das últimas {window} obs.)")
        return I0

    def _determine_phase(
        self,
        I_current: float,
        I_history: List[float],
        peak_infection: float
    ) -> str:
        """
        Determina a fase atual da 'epidemia' de pânico/euforia.

        Fases:
            INÍCIO     - Infecção baixa e crescente (I < 10% do pico)
            ACELERAÇÃO - Infecção crescente significativa
            PICO       - Próximo ou no máximo de infecção
            DECLÍNIO   - Infecção decrescente
            ESTÁVEL    - Infecção muito baixa / epidemia extinta

        Parâmetros:
            I_current: Valor atual de I.
            I_history: Histórico completo de I.
            peak_infection: Valor máximo de I.

        Retorna:
            String com a fase atual.
        """
        if peak_infection < 0.02:
            return "ESTÁVEL"

        ratio_to_peak = I_current / max(peak_infection, 1e-10)

        # Verificar tendência recente
        if len(I_history) >= 3:
            recent_trend = I_history[-1] - I_history[-3]
        else:
            recent_trend = 0.0

        if ratio_to_peak > 0.9 and abs(recent_trend) < 0.01:
            return "PICO"
        elif recent_trend > 0.005:
            if ratio_to_peak < 0.3:
                return "INÍCIO"
            else:
                return "ACELERAÇÃO"
        elif recent_trend < -0.005:
            return "DECLÍNIO"
        else:
            if I_current < 0.02:
                return "ESTÁVEL"
            elif ratio_to_peak > 0.7:
                return "PICO"
            else:
                return "DECLÍNIO"

    def run_simulation(
        self,
        df: pd.DataFrame,
        projection_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Executa a simulação do modelo SIR para propagação de pânico no mercado.

        Etapas:
        1. Extrai anomalias do DataFrame (coluna 'anomaly')
        2. Calibra β (contágio) pela velocidade de clustering de anomalias
        3. Calibra γ (recuperação) pela duração dos clusters
        4. Estima I₀ pelas anomalias recentes
        5. Resolve as equações SIR via Euler explícito
        6. Projeta a dinâmica para `projection_steps` passos à frente
        7. Classifica a fase atual da propagação

        Parâmetros:
            df: DataFrame com coluna 'anomaly' (binária 0/1).
                Opcionalmente 'close' para dados adicionais.
            projection_steps: Número de passos futuros a projetar.

        Retorna:
            Dicionário com trajetórias S/I/R, parâmetros calibrados,
            R₀, pico de infecção e fase atual.

        Levanta:
            ValueError: Se os dados forem insuficientes.
        """
        # --- Validação de entrada ---
        if df is None or df.empty or len(df) < 10:
            raise ValueError(
                "Dados insuficientes para simulação SIR. "
                "São necessários pelo menos 10 registros."
            )

        # Extrair ou gerar anomalias
        if 'anomaly' in df.columns:
            anomalies = df['anomaly'].fillna(0).values.astype(int)
        elif 'z_score' in df.columns:
            # Fallback: usar z_score > 2 como proxy de anomalia
            z_scores = df['z_score'].fillna(0).values.astype(float)
            anomalies = (np.abs(z_scores) > 2.0).astype(int)
            logger.info("Coluna 'anomaly' ausente. Usando |z_score| > 2 como proxy.")
        elif 'close' in df.columns:
            # Fallback: anomalias baseadas em retornos extremos
            close = df['close'].dropna().values.astype(float)
            close = np.clip(close, 1e-10, None)
            returns = np.diff(np.log(close))
            vol = np.std(returns) if len(returns) > 1 else 0.01
            anomalies = np.zeros(len(df), dtype=int)
            if len(returns) > 0 and vol > 1e-12:
                extreme = (np.abs(returns) > 2.0 * vol).astype(int)
                anomalies[1:len(extreme) + 1] = extreme
            logger.info(
                "Colunas 'anomaly' e 'z_score' ausentes. "
                "Usando retornos extremos como proxy."
            )
        else:
            raise ValueError(
                "DataFrame deve conter ao menos uma das colunas: "
                "'anomaly', 'z_score', ou 'close'."
            )

        N = len(df)  # População total

        logger.info(
            f"Iniciando simulação SIR: N={N}, "
            f"{int(np.sum(anomalies))} anomalias detectadas, "
            f"{projection_steps} passos de projeção."
        )

        # --- Calibração dos parâmetros ---
        beta = self._calibrate_beta(anomalies, N)
        gamma = self._calibrate_gamma(anomalies)

        # R₀ (número básico de reprodução)
        R0 = beta / gamma if gamma > 1e-10 else float('inf')

        # Limiar de imunidade de rebanho
        if R0 > 1.0:
            herd_immunity = 1.0 - 1.0 / R0
        else:
            herd_immunity = 0.0

        logger.info(
            f"Parâmetros SIR calibrados: β={beta:.4f}, γ={gamma:.4f}, "
            f"R₀={R0:.4f}, limiar_imunidade={herd_immunity:.4f}"
        )

        # --- Condições iniciais (proporções) ---
        I0 = self._estimate_initial_infected(anomalies)
        S0 = 1.0 - I0
        R_init = 0.0  # Ninguém recuperado no início

        # --- Resolver equações SIR via Euler explícito ---
        dt = 1.0  # Passo temporal unitário
        steps_list: List[int] = [0]
        S_list: List[float] = [S0]
        I_list: List[float] = [I0]
        R_list: List[float] = [R_init]

        S, I, R_val = S0, I0, R_init

        for step in range(1, projection_steps + 1):
            # Equações SIR (Euler explícito)
            dS = -beta * S * I * dt
            dI = (beta * S * I - gamma * I) * dt
            dR = gamma * I * dt

            S = S + dS
            I = I + dI
            R_val = R_val + dR

            # Garantir que as proporções permaneçam fisicamente válidas
            S = float(np.clip(S, 0.0, 1.0))
            I = float(np.clip(I, 0.0, 1.0))
            R_val = float(np.clip(R_val, 0.0, 1.0))

            # Renormalizar para que S + I + R = 1
            total = S + I + R_val
            if total > 1e-10:
                S /= total
                I /= total
                R_val /= total

            steps_list.append(step)
            S_list.append(S)
            I_list.append(I)
            R_list.append(R_val)

        # --- Análise dos resultados ---
        peak_infection = float(max(I_list))
        peak_step = int(np.argmax(I_list))

        # Fase atual (baseada nos primeiros pontos da projeção)
        current_phase = self._determine_phase(I_list[0], I_list[:5], peak_infection)

        result = {
            "steps": steps_list,
            "S": S_list,
            "I": I_list,
            "R": R_list,
            "beta": beta,
            "gamma": gamma,
            "R0": float(R0),
            "peak_infection": peak_infection,
            "peak_step": peak_step,
            "current_phase": current_phase,
            "herd_immunity_threshold": float(herd_immunity),
        }

        logger.info(
            f"Simulação SIR concluída. R₀={R0:.4f}, "
            f"pico_I={peak_infection:.4f} no passo {peak_step}, "
            f"fase={current_phase}"
        )

        return result
