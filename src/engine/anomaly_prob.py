import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AnomalyProbabilityEngine:
    """
    Motor matemático para cálculo de probabilidade e estatísticas de eventos baseados em anomalias.
    Calcula matrizes de transição de Markov e tempo estimado de retorno usando processos de Poisson.
    """

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """
        Calcula as métricas probabilísticas de anomalias com base nas marcações do DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame histórico contendo as colunas ['timestamp', 'anomaly'].
            
        Returns:
            dict: Dicionário contendo as métricas de transição, Poisson e agrupamentos temporais.
        """
        if df is None or df.empty or "anomaly" not in df.columns:
            raise ValueError("O DataFrame deve conter a coluna 'anomaly'.")

        anomaly_series = df["anomaly"].values
        total_points = len(anomaly_series)
        num_anomalies = int(anomaly_series.sum())
        
        # 1. Probabilidade Geral Empírica
        overall_prob = num_anomalies / total_points if total_points > 0 else 0.0

        # 2. Cadeia de Markov (Transição de Estados)
        # Estados: False (Estável - S), True (Anomalia - A)
        s_to_s = s_to_a = a_to_s = a_to_a = 0
        
        for i in range(len(anomaly_series) - 1):
            curr_state = anomaly_series[i]
            next_state = anomaly_series[i+1]
            
            if not curr_state: # Atual é Estável
                if not next_state:
                    s_to_s += 1
                else:
                    s_to_a += 1
            else: # Atual é Anomalia
                if not next_state:
                    a_to_s += 1
                else:
                    a_to_a += 1
                    
        total_s_starts = s_to_s + s_to_a
        total_a_starts = a_to_s + a_to_a
        
        # Cálculo de probabilidades condicionais
        p_s_s = s_to_s / total_s_starts if total_s_starts > 0 else 1.0
        p_s_a = s_to_a / total_s_starts if total_s_starts > 0 else 0.0
        p_a_s = a_to_s / total_a_starts if total_a_starts > 0 else 1.0
        p_a_a = a_to_a / total_a_starts if total_a_starts > 0 else 0.0

        # 3. Processo de Poisson e Inter-chegadas (Distância)
        anomaly_indices = np.where(anomaly_series == True)[0]
        
        if len(anomaly_indices) >= 2:
            distances = np.diff(anomaly_indices)
            mean_distance = float(distances.mean())
        else:
            # Fallback se tiver menos de duas anomalias
            mean_distance = float(total_points)

        # Taxa de ocorrência por período (lambda)
        rate_lambda = 1.0 / mean_distance if mean_distance > 0 else 0.0
        
        # Estado atual do ativo (último elemento)
        current_is_anomaly = bool(anomaly_series[-1])
        
        # Risco imediato para o próximo candle (Markov conditional probability)
        immediate_risk = p_a_a if current_is_anomaly else p_s_a

        # 4. Probabilidade Acumulada de Poisson para os próximos K períodos (candles)
        steps = np.arange(1, 31)
        poisson_probs = 1.0 - np.exp(-rate_lambda * steps)

        # 5. Análise de Fatores Temporais (Agrupamentos por Hora/Dia da Semana)
        hourly_distribution = []
        weekday_distribution = []
        
        try:
            df_temp = df.copy()
            df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
            
            # Análise por hora
            df_temp["hour"] = df_temp["timestamp"].dt.hour
            hourly_counts = df_temp.groupby("hour")["anomaly"].agg(["sum", "count"])
            hourly_counts["rate"] = (hourly_counts["sum"] / hourly_counts["count"]) * 100
            for hr, row in hourly_counts.iterrows():
                hourly_distribution.append({
                    "hour": int(hr),
                    "anomalies": int(row["sum"]),
                    "total": int(row["count"]),
                    "rate": float(row["rate"])
                })
                
            # Análise por dia da semana
            df_temp["weekday"] = df_temp["timestamp"].dt.day_name()
            df_temp["weekday_num"] = df_temp["timestamp"].dt.weekday
            weekday_counts = df_temp.groupby(["weekday", "weekday_num"])["anomaly"].agg(["sum", "count"])
            weekday_counts = weekday_counts.reset_index().sort_values("weekday_num")
            
            for _, row in weekday_counts.iterrows():
                weekday_distribution.append({
                    "day": str(row["weekday"]),
                    "anomalies": int(row["sum"]),
                    "total": int(row["count"]),
                    "rate": float(row["sum"] / row["count"] * 100) if row["count"] > 0 else 0.0
                })
        except Exception as e:
            logger.warning(f"Erro ao extrair métricas temporais: {e}. Usando distribuições vazias.")

        return {
            "overall_probability": overall_prob * 100, # em %
            "num_anomalies": num_anomalies,
            "total_points": total_points,
            "markov_matrix": {
                "S_to_S": p_s_s * 100,
                "S_to_A": p_s_a * 100,
                "A_to_S": p_a_s * 100,
                "A_to_A": p_a_a * 100
            },
            "mean_distance": mean_distance,
            "rate_lambda": rate_lambda,
            "current_state_anomaly": current_is_anomaly,
            "immediate_risk": immediate_risk * 100, # em %
            "poisson_curve": {
                "steps": steps.tolist(),
                "probabilities": (poisson_probs * 100).tolist() # em %
            },
            "hourly_distribution": hourly_distribution,
            "weekday_distribution": weekday_distribution
        }
