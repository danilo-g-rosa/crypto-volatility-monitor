import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AnomalyTrendEngine:
    """
    Motor analítico para estudo de eventos históricos (Event Study) e correspondência de padrões.
    Mapeia os retornos futuros após anomalias de alta e baixa para projetar a tendência
    e o preço provável quando ocorre uma nova anomalia.
    """

    def analyze_patterns(self, df: pd.DataFrame, max_steps: int = 5) -> dict:
        """
        Analisa o comportamento do preço pós-anomalia em toda a série histórica.
        
        Args:
            df (pd.DataFrame): DataFrame histórico contendo ['close', 'z_score', 'anomaly'].
            max_steps (int): Número de períodos futuros pós-evento a analisar.
            
        Returns:
            dict: Dicionário contendo as estatísticas agregadas e caminhos de preço médios.
        """
        if df is None or df.empty or "anomaly" not in df.columns or "z_score" not in df.columns:
            raise ValueError("O DataFrame deve conter as colunas 'anomaly', 'z_score' e 'close'.")

        close_prices = df["close"].values
        anomalies = df["anomaly"].values
        z_scores = df["z_score"].values
        n = len(df)

        # Listas para guardar os caminhos dos retornos acumulados (%) nos próximos 'max_steps' períodos
        pos_anomaly_paths = [] # Para anomalias com Z > 0
        neg_anomaly_paths = [] # Para anomalias com Z < 0

        # Mapeia cada ponto histórico
        for i in range(n):
            if anomalies[i]:
                z = z_scores[i]
                price_at_event = close_prices[i]
                
                if price_at_event == 0:
                    continue

                # Calcula os retornos acumulados de t+1 até t+max_steps
                path = []
                for step in range(1, max_steps + 1):
                    target_idx = i + step
                    if target_idx < n:
                        ret = ((close_prices[target_idx] - price_at_event) / price_at_event) * 100.0
                        path.append(ret)
                    else:
                        path.append(np.nan) # Evento na borda final do dataset
                
                # Separa por tipo de anomalia
                if z > 0:
                    pos_anomaly_paths.append(path)
                else:
                    neg_anomaly_paths.append(path)

        pos_anomaly_paths = np.array(pos_anomaly_paths) if pos_anomaly_paths else np.empty((0, max_steps))
        neg_anomaly_paths = np.array(neg_anomaly_paths) if neg_anomaly_paths else np.empty((0, max_steps))

        # Função interna para extrair métricas de caminhos de retorno
        def get_stats_for_paths(paths, is_positive):
            num_events = len(paths)
            if num_events == 0:
                return {
                    "count": 0,
                    "mean_returns": [0.0] * max_steps,
                    "win_rates": [50.0] * max_steps,
                    "max_returns": [0.0] * max_steps,
                    "min_returns": [0.0] * max_steps
                }

            mean_returns = []
            win_rates = []
            max_returns = []
            min_returns = []

            for step in range(max_steps):
                step_vals = paths[:, step]
                valid_vals = step_vals[~np.isnan(step_vals)]
                
                if len(valid_vals) > 0:
                    mean_ret = float(valid_vals.mean())
                    max_ret = float(valid_vals.max())
                    min_ret = float(valid_vals.min())
                    
                    # Para anomalias positivas (alta), o "acerto" é o preço subir (> 0)
                    # Para anomalias negativas (baixa), o "acerto" é o preço descer (< 0)
                    if is_positive:
                        wins = np.sum(valid_vals > 0)
                    else:
                        wins = np.sum(valid_vals < 0)
                        
                    win_rate = (wins / len(valid_vals)) * 100.0
                else:
                    mean_ret = 0.0
                    win_rate = 50.0
                    max_ret = 0.0
                    min_ret = 0.0

                mean_returns.append(mean_ret)
                win_rates.append(win_rate)
                max_returns.append(max_ret)
                min_returns.append(min_ret)

            return {
                "count": num_events,
                "mean_returns": mean_returns,
                "win_rates": win_rates,
                "max_returns": max_returns,
                "min_returns": min_returns
            }

        pos_stats = get_stats_for_paths(pos_anomaly_paths, is_positive=True)
        neg_stats = get_stats_for_paths(neg_anomaly_paths, is_positive=False)

        return {
            "pos_anomaly": pos_stats,
            "neg_anomaly": neg_stats,
            "max_steps": max_steps
        }

    def generate_current_forecast(self, df: pd.DataFrame, pattern_stats: dict, lookback: int = 5) -> dict:
        """
        Verifica se há uma anomalia recente e gera a projeção de preço baseada nos padrões históricos.
        
        Args:
            df (pd.DataFrame): DataFrame histórico contendo ['close', 'z_score', 'anomaly', 'timestamp'].
            pattern_stats (dict): Estatísticas calculadas pelo método analyze_patterns.
            lookback (int): Distância máxima em candles para considerar o evento ativo.
            
        Returns:
            dict: Informações de previsão ativa.
        """
        if df is None or df.empty:
            return {"active": False}

        anomalies = df["anomaly"].values
        anomaly_indices = np.where(anomalies == True)[0]

        if len(anomaly_indices) == 0:
            return {
                "active": False,
                "reason": "Nenhuma anomalia histórica detectada no conjunto de dados."
            }

        # Última anomalia
        last_anomaly_idx = anomaly_indices[-1]
        elapsed = len(df) - 1 - last_anomaly_idx

        # Se a anomalia ocorreu dentro do limite de lookback
        if elapsed <= lookback:
            last_anomaly_row = df.iloc[last_anomaly_idx]
            current_row = df.iloc[-1]
            
            z_val = float(last_anomaly_row["z_score"])
            anomaly_price = float(last_anomaly_row["close"])
            current_price = float(current_row["close"])
            
            is_pos = z_val > 0
            stats = pattern_stats["pos_anomaly"] if is_pos else pattern_stats["neg_anomaly"]
            
            # Definimos o alvo no período final (passo 5)
            # O alvo é calculado a partir do preço da anomalia multiplicado pelo retorno médio histórico do passo 5
            max_steps = pattern_stats["max_steps"]
            mean_ret_5 = stats["mean_returns"][max_steps - 1]
            win_rate_5 = stats["win_rates"][max_steps - 1]
            max_ret_5 = stats["max_returns"][max_steps - 1]
            min_ret_5 = stats["min_returns"][max_steps - 1]

            target_price = anomaly_price * (1.0 + (mean_ret_5 / 100.0))
            lower_target = anomaly_price * (1.0 + (min_ret_5 / 100.0))
            upper_target = anomaly_price * (1.0 + (max_ret_5 / 100.0))

            # Direção projetada com base no retorno médio esperado
            if is_pos:
                expected_direction = "ALTA" if mean_ret_5 >= 0 else "BAIXA"
            else:
                expected_direction = "BAIXA" if mean_ret_5 <= 0 else "ALTA"

            # Tempo restante para atingir o alvo
            remaining_candles = max_steps - elapsed
            
            # Se a anomalia ocorreu a mais de 5 candles mas menor que lookback,
            # nós projetamos com base nos dados que ainda restam do efeito (se houver).
            # Para manter simples, a previsão ativa dura até 5 períodos pós-anomalia.
            if remaining_candles > 0:
                return {
                    "active": True,
                    "anomaly_type": "Alta Volatilidade de Alta (+Z)" if is_pos else "Alta Volatilidade de Baixa (-Z)",
                    "anomaly_price": anomaly_price,
                    "anomaly_timestamp": last_anomaly_row["timestamp"],
                    "elapsed_candles": elapsed,
                    "remaining_candles": remaining_candles,
                    "direction": expected_direction,
                    "target_price": target_price,
                    "lower_target": lower_target,
                    "upper_target": upper_target,
                    "confidence": win_rate_5,
                    "historical_events": stats["count"]
                }

        return {
            "active": False,
            "reason": f"O mercado está estável. Nenhuma anomalia detectada nos últimos {lookback} candles."
        }
