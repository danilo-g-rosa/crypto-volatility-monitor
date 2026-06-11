import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MonteCarloEngine:
    """
    Motor matemático para execução de simulações de Monte Carlo de preços baseadas
    no Movimento Browniano Geométrico (GBM).
    """

    def run_simulation(self, df: pd.DataFrame, steps: int = 15, num_paths: int = 150) -> dict:
        """
        Executa a simulação de Monte Carlo a partir do último preço histórico.
        
        Args:
            df (pd.DataFrame): DataFrame histórico contendo ['close', 'timestamp'].
            steps (int): Quantidade de passos temporais futuros a simular.
            num_paths (int): Quantidade de caminhos simulados.
            
        Returns:
            dict: Resultados da simulação contendo caminhos, estatísticas de percentis e timestamps futuros.
        """
        if df is None or df.empty or len(df) < 5:
            raise ValueError("Dados históricos insuficientes para calibrar a simulação de Monte Carlo.")

        prices = df["close"].values
        last_price = float(prices[-1])
        
        # 1. Calcular os retornos logarítmicos
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # 2. Estimar os parâmetros do GBM (Drift e Volatilidade)
        mean_return = log_returns.mean()
        var_return = log_returns.var()
        volatility = log_returns.std()
        
        if volatility == 0:
            # Evita erro em dados constantes
            volatility = 0.01
            
        # Parâmetro de drift corrigido pelo desvio
        drift = mean_return - (0.5 * var_return)
        
        # 3. Execução da Simulação
        # Matriz de números normais aleatórios Z ~ N(0, 1)
        np.random.seed(42)  # Semente fixa para consistência visual nas renderizações do Streamlit
        random_shocks = np.random.normal(0, 1, (steps, num_paths))
        
        # Matriz de caminhos simulados (linhas = passos, colunas = caminhos)
        # Inicializa com o preço atual na linha 0
        sim_prices = np.zeros((steps + 1, num_paths))
        sim_prices[0, :] = last_price
        
        # Simula iterativamente passo a passo
        for t in range(1, steps + 1):
            shocks = random_shocks[t - 1, :]
            # GBM formula: S_t = S_{t-1} * exp(drift + vol * Z)
            sim_prices[t, :] = sim_prices[t - 1, :] * np.exp(drift + volatility * shocks)
            
        # 4. Cálculo de Estatísticas de Percentis (5%, 50%, 95%) por passo temporal
        percentile_5 = np.percentile(sim_prices, 5, axis=1)
        percentile_50 = np.percentile(sim_prices, 50, axis=1)  # Mediana
        percentile_95 = np.percentile(sim_prices, 95, axis=1)
        
        # 5. Geração de Timestamps Futuros Adaptados
        if len(df) > 1:
            time_delta = df["timestamp"].diff().mean()
        else:
            time_delta = pd.Timedelta(days=1)
            
        last_timestamp = df["timestamp"].iloc[-1]
        future_timestamps = [last_timestamp] + [last_timestamp + ((i + 1) * time_delta) for i in range(steps)]
        
        return {
            "last_price": last_price,
            "steps": steps,
            "num_paths": num_paths,
            "timestamps": future_timestamps,
            "paths": sim_prices.tolist(), # Matriz (passos+1, caminhos)
            "percentile_5": percentile_5.tolist(),
            "percentile_50": percentile_50.tolist(),
            "percentile_95": percentile_95.tolist(),
            "drift_diario": float(mean_return),
            "volatilidade_diaria": float(volatility)
        }
