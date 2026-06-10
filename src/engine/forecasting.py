import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import Holt
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class ForecastEngine:
    """
    Motor analítico para geração de previsões probabilísticas e consenso de preços.
    Integra modelos estatísticos e matemáticos como ARIMA, Holt (Suavização Exponencial)
    e Regressão Linear com cálculo de intervalo de confiança baseado em volatilidade histórica.
    """
    
    def __init__(self, steps: int = 7):
        """
        Inicializa o motor de previsão.
        
        Args:
            steps (int): Quantidade de candles/períodos futuros a projetar.
        """
        self.steps = steps

    def _predict_arima(self, series: pd.Series) -> np.ndarray:
        """Projeta preços futuros usando um modelo ARIMA(1, 1, 1)."""
        try:
            # Para desempenho e estabilidade, usamos no máximo os últimos 200 candles
            data = series.values[-200:]
            # order=(1,1,1) é um padrão robusto para passeios aleatórios com ruído
            model = ARIMA(data, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=self.steps)
            if len(forecast) == self.steps and not np.isnan(forecast).any():
                return forecast
        except Exception as e:
            logger.warning(f"Falha no ARIMA: {e}. Usando fallback constante.")
        
        # Fallback: Mantém o último preço constante
        return np.full(self.steps, series.iloc[-1])

    def _predict_holt(self, series: pd.Series) -> np.ndarray:
        """Projeta preços futuros usando Suavização Exponencial Dupla de Holt."""
        try:
            data = series.values[-200:]
            # Holt é adequado para séries com tendências sem sazonalidade rígida
            model = Holt(data, initialization_method="estimated")
            fitted = model.fit()
            forecast = fitted.forecast(self.steps)
            if len(forecast) == self.steps and not np.isnan(forecast).any():
                return forecast
        except Exception as e:
            logger.warning(f"Falha no Holt: {e}. Usando fallback constante.")
        
        # Fallback: Mantém o último preço constante
        return np.full(self.steps, series.iloc[-1])

    def _predict_linear_trend(self, series: pd.Series) -> np.ndarray:
        """Projeta preços futuros usando Regressão Linear baseada nos últimos 30 períodos."""
        try:
            # Focamos na tendência recente (últimos 30 candles)
            window = min(30, len(series))
            y = series.values[-window:]
            X = np.arange(window).reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X, y)
            
            X_future = np.arange(window, window + self.steps).reshape(-1, 1)
            forecast = model.predict(X_future)
            if len(forecast) == self.steps and not np.isnan(forecast).any():
                return forecast
        except Exception as e:
            logger.warning(f"Falha na Regressão Linear: {e}. Usando fallback constante.")
            
        return np.full(self.steps, series.iloc[-1])

    def run_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orquestra a execução de todos os modelos de previsão e monta o DataFrame futuro com consenso.
        
        Args:
            df (pd.DataFrame): DataFrame histórico contendo ['timestamp', 'close'].
            
        Returns:
            pd.DataFrame: DataFrame contendo as projeções futures mapeadas por timestamp.
        """
        if df is None or df.empty or len(df) < 5:
            raise ValueError("Dados históricos insuficientes para treinar os modelos de previsão.")
            
        prices = df["close"]
        last_price = prices.iloc[-1]
        
        # 1. Executa os modelos individuais
        arima_pred = self._predict_arima(prices)
        holt_pred = self._predict_holt(prices)
        linear_pred = self._predict_linear_trend(prices)
        
        # 2. Calcula o Consenso (média aritmética das projeções)
        consensus = (arima_pred + holt_pred + linear_pred) / 3.0
        
        # 3. Calcula o Intervalo de Confiança Probabilístico (95% de confiança)
        # Incerteza cresce proporcionalmente à volatilidade histórica e à raiz do tempo (sqrt(t))
        historical_diffs = prices.diff().dropna()
        volatility = historical_diffs.std() if len(historical_diffs) > 0 else (last_price * 0.01)
        
        # Fator de expansão da incerteza por passo temporal
        steps_seq = np.arange(1, self.steps + 1)
        std_errors = volatility * np.sqrt(steps_seq)
        
        # Limites (Z-score de 1.96 para 95% de confiança bicaudal)
        upper_bound = consensus + (1.96 * std_errors)
        lower_bound = consensus - (1.96 * std_errors)
        
        # Impedir preços negativos
        lower_bound = np.clip(lower_bound, a_min=0, a_max=None)
        
        # 4. Geração dos Timestamps Futuros Adaptados
        # Detecta dinamicamente a frequência de amostragem (intervalo de candles)
        if len(df) > 1:
            time_delta = df["timestamp"].diff().mean()
        else:
            time_delta = pd.Timedelta(days=1)
            
        last_timestamp = df["timestamp"].iloc[-1]
        future_timestamps = [last_timestamp + ((i + 1) * time_delta) for i in range(self.steps)]
        
        # 5. Consolidação do DataFrame de Saída
        forecast_df = pd.DataFrame({
            "timestamp": future_timestamps,
            "arima": arima_pred,
            "holt_winters": holt_pred,
            "linear_regression": linear_pred,
            "consensus": consensus,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound
        })
        
        return forecast_df
