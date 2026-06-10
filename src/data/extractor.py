from abc import ABC, abstractmethod
import pandas as pd
import requests
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """
    Interface/Classe Abstrata Base para Extratores de Dados Financeiros.
    Garante a padronização e modularidade para diferentes fontes de dados.
    """
    
    @abstractmethod
    def fetch_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Extrai dados de mercado para um ativo específico.
        
        Args:
            symbol (str): O símbolo do par de negociação (ex: 'BTCUSDT').
            interval (str): O intervalo das velas (ex: '1d', '4h', '1h').
            limit (int): Limite máximo de registros a serem retornados.
            
        Returns:
            pd.DataFrame: DataFrame padronizado com colunas:
                          ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        pass


class BinanceExtractor(BaseExtractor):
    """
    Implementação do extrator de dados utilizando a API pública REST da Binance.
    """
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    def fetch_data(self, symbol: str, interval: str = "1d", limit: int = 100) -> pd.DataFrame:
        """
        Consulta os dados de candlestick (klines) na Binance API.
        
        Args:
            symbol (str): Símbolo do par de negociação, ex: "BTCUSDT".
            interval (str): Intervalo de tempo das velas, ex: "1m", "5m", "15m", "1h", "4h", "1d".
            limit (int): Limite de candles retornados (máximo 1000). Padrão é 100.
            
        Returns:
            pd.DataFrame: DataFrame com colunas padronizadas indexado por timestamp
                          contendo 'open', 'high', 'low', 'close', 'volume' como tipos float.
                          
        Raises:
            ValueError: Se os parâmetros fornecidos forem inválidos ou se o retorno estiver vazio.
            ConnectionError: Em caso de falha de conexão de rede ou HTTP.
        """
        if not symbol:
            raise ValueError("O parâmetro 'symbol' não pode ser vazio.")
        
        symbol = symbol.upper().strip()
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            logger.info(f"Iniciando requisição na Binance para {symbol} (intervalo: {interval}, limite: {limit})...")
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            
            # Lança HTTPError se o status code for 4xx ou 5xx
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError(f"Nenhum dado retornado para o par {symbol} com o intervalo {interval}.")
                
            # O retorno da API da Binance é uma lista de listas contendo informações das KLines.
            # Mapeamos as primeiras 6 colunas relevantes do contrato de retorno.
            raw_columns = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ]
            
            df = pd.DataFrame(data, columns=raw_columns)
            
            # Mantendo apenas as colunas essenciais
            df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
            df.rename(columns={"open_time": "timestamp"}, inplace=True)
            
            # Conversão de timestamp em milissegundos para objeto datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # Conversão das demais colunas de strings para tipo numérico (float)
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
            # Remove linhas que possam ter falhas na conversão
            df.dropna(subset=numeric_cols, inplace=True)
            
            logger.info(f"Dados obtidos e tratados com sucesso. Total de registros: {len(df)}")
            return df
            
        except requests.exceptions.Timeout as t_err:
            logger.error(f"Timeout na conexão com a API da Binance: {t_err}")
            raise ConnectionError(f"A requisição para Binance excedeu o tempo limite de conexão: {t_err}") from t_err
            
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Erro ao requisitar dados da Binance: {req_err}")
            raise ConnectionError(f"Erro de comunicação de rede ao contactar a API da Binance: {req_err}") from req_err
            
        except (ValueError, KeyError) as json_err:
            logger.error(f"Erro no parse de dados JSON retornados pela Binance: {json_err}")
            raise ValueError(f"Erro na formatação dos dados retornados pela Binance: {json_err}") from json_err
