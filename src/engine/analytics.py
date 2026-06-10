import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FinanceEngine:
    """
    Motor analítico independente para processamento de métricas financeiras e detecção de anomalias.
    
    Esta classe opera exclusivamente sobre DataFrames genéricos contendo colunas
    como 'close' (preço de fechamento) e 'volume', garantindo total independência de APIs externas.
    """
    
    def process(self, df: pd.DataFrame, period: int = 20, std_dev_mult: float = 2.0, z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Orquestra o cálculo das Bandas de Bollinger, Z-Score e detecção de anomalias.
        
        Args:
            df (pd.DataFrame): DataFrame contendo pelo menos as colunas 'close' e 'volume'.
            period (int): Período da média móvel para as Bandas de Bollinger. Padrão é 20.
            std_dev_mult (float): Multiplicador do desvio padrão para as Bandas de Bollinger. Padrão é 2.0.
            z_threshold (float): Limiar absoluto do Z-Score para considerar o preço anômalo. Padrão é 2.0.
            
        Returns:
            pd.DataFrame: Cópia do DataFrame original enriquecido com as colunas:
                          ['bb_middle', 'bb_upper', 'bb_lower', 'z_score', 'anomaly']
                          
        Raises:
            ValueError: Se o DataFrame for inválido, vazio ou não possuir as colunas necessárias.
            RuntimeError: Se ocorrer um erro durante os cálculos analíticos.
        """
        if df is None:
            raise ValueError("O DataFrame fornecido não pode ser nulo.")
            
        if df.empty:
            raise ValueError("O DataFrame fornecido está vazio. Não é possível calcular indicadores analíticos.")
            
        required_cols = {"close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes no DataFrame: {list(missing_cols)}")
            
        # Cria cópia profunda para evitar efeitos colaterais indesejados nos dados de entrada
        processed_df = df.copy()
        
        try:
            logger.info("Iniciando processamento analítico (Bandas de Bollinger, Z-Score e Anomalias)...")
            
            # 1. Bandas de Bollinger
            # Média Móvel Simples (SMA) do Preço de Fechamento
            processed_df["bb_middle"] = processed_df["close"].rolling(window=period).mean()
            # Desvio Padrão Móvel
            rolling_std = processed_df["close"].rolling(window=period).std()
            
            # Bandas Superior e Inferior
            processed_df["bb_upper"] = processed_df["bb_middle"] + (std_dev_mult * rolling_std)
            processed_df["bb_lower"] = processed_df["bb_middle"] - (std_dev_mult * rolling_std)
            
            # 2. Z-Score (Desvio em relação à Média Móvel dividido pelo Desvio Padrão Móvel)
            # Evita divisão por zero substituindo desvio padrão de zero por NaN
            safe_rolling_std = rolling_std.replace(0, np.nan)
            processed_df["z_score"] = (processed_df["close"] - processed_df["bb_middle"]) / safe_rolling_std
            processed_df["z_score"] = processed_df["z_score"].fillna(0) # Trata possíveis NaNs ou divisões por zero
            
            # 3. Detecção de Anomalias baseada no Z-Score absoluto ultrapassando o limiar (threshold)
            processed_df["anomaly"] = np.abs(processed_df["z_score"]) > z_threshold
            
            num_anomalies = int(processed_df["anomaly"].sum())
            logger.info(f"Processamento concluído. {num_anomalies} anomalias detectadas no conjunto de dados.")
            
            return processed_df
            
        except Exception as err:
            logger.error(f"Erro inesperado no cálculo dos indicadores matemáticos: {err}")
            raise RuntimeError(f"Falha no processamento analítico: {err}") from err
