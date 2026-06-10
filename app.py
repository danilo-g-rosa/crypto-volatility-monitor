import streamlit as st
import logging
from src.data.extractor import BinanceExtractor
from src.engine.analytics import FinanceEngine
from src.engine.forecasting import ForecastEngine
from src.engine.anomaly_prob import AnomalyProbabilityEngine
from src.engine.anomaly_trend import AnomalyTrendEngine
from src.ui.components import apply_custom_css, render_metrics_cards, render_candlestick_chart, render_forecast_tab, render_anomaly_probability_tab, render_anomaly_trend_tab

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Função principal que gerencia o fluxo de controle e orquestração do painel.
    Executa a inicialização do estado de sessão (session_state), lê parâmetros
    do painel lateral (sidebar), executa extração condicional e processamento analítico,
    e finalmente invoca os renderizadores visuais.
    """
    # 1. Configurações da página do Streamlit
    st.set_page_config(
        page_title="Crypto Volatility Monitor",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 2. Injeta estilos customizados Cyberpunk
    apply_custom_css()
    
    # Cabeçalho Principal do Painel
    st.markdown("<h1>⚡ Crypto Volatility Monitor</h1>", unsafe_allow_html=True)
    st.markdown("### Monitoramento Analítico de Volatilidade e Detecção de Anomalias via Z-Score", unsafe_allow_html=True)
    st.write("")
    
    # 3. Painel de Controle Lateral (Sidebar)
    st.sidebar.markdown("<h2>⚙️ CONFIGURAÇÕES</h2>", unsafe_allow_html=True)
    
    # Parâmetros de Consulta de Dados
    symbol = st.sidebar.selectbox(
        "Selecione o Par Crypto:",
        options=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"],
        index=0,
        help="Par de negociação para extração na Binance."
    )
    
    interval = st.sidebar.selectbox(
        "Intervalo das Velas:",
        options=["15m", "1h", "4h", "1d", "1w"],
        index=3,
        help="Intervalo temporal para agregação de cada vela."
    )
    
    limit = st.sidebar.slider(
        "Quantidade de Candles:",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="Número total de candles a serem consultados."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h2>📊 INDICADORES</h2>", unsafe_allow_html=True)
    
    # Parâmetros do Motor Matemático
    bb_period = st.sidebar.slider(
        "Período das Bandas (Bollinger):",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Janela móvel para o cálculo da média simples (SMA) e do desvio padrão."
    )
    
    bb_std_dev = st.sidebar.slider(
        "Desvio Padrão (Bollinger):",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Multiplicador de desvio padrão para definir os limites superior e inferior."
    )
    
    z_threshold = st.sidebar.slider(
        "Limiar de Anomalia (Z-Score):",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Limite de desvio absoluto necessário para que um preço seja rotulado como anomalia."
    )
    
    # Botão de atualização manual
    force_refresh = st.sidebar.button("↻ ATUALIZAR DADOS")
    
    # 4. Inicializa o estado da sessão (Session State) para otimização de requisições de rede
    if "raw_data" not in st.session_state:
        st.session_state["raw_data"] = None
    if "processed_data" not in st.session_state:
        st.session_state["processed_data"] = None
    if "last_fetched_key" not in st.session_state:
        st.session_state["last_fetched_key"] = ""
    if "last_analytics_key" not in st.session_state:
        st.session_state["last_analytics_key"] = ""
        
    # Chave para determinar se os parâmetros de dados mudaram
    current_fetch_key = f"{symbol}_{interval}_{limit}"
    
    # Instanciando os componentes da arquitetura SoC
    extractor = BinanceExtractor()
    engine = FinanceEngine()
    
    # Determina se precisamos buscar dados novos de rede
    should_fetch = (
        st.session_state["raw_data"] is None or 
        st.session_state["last_fetched_key"] != current_fetch_key or
        force_refresh
    )
    
    if should_fetch:
        with st.spinner(f"Consultando API da Binance para {symbol}..."):
            try:
                # Executa extração de dados
                raw_df = extractor.fetch_data(symbol, interval, limit)
                st.session_state["raw_data"] = raw_df
                st.session_state["last_fetched_key"] = current_fetch_key
                # Remove dados processados anteriores para forçar a re-execução do motor analítico
                st.session_state["processed_data"] = None 
                logger.info(f"Novos dados brutos baixados e salvos em sessão para {symbol}.")
            except Exception as err:
                st.error(f"⚠️ Erro de Conexão/Extração de Dados: {err}")
                st.session_state["raw_data"] = None
                st.session_state["processed_data"] = None
                return
                
    # Determina se precisamos reprocessar as métricas analíticas
    # Processa se dados analíticos não existirem ou se as configurações do motor mudaram
    if st.session_state["raw_data"] is not None:
        current_analytics_key = f"{bb_period}_{bb_std_dev}_{z_threshold}"
        
        should_process = (
            st.session_state["processed_data"] is None or
            st.session_state["last_analytics_key"] != current_analytics_key
        )
        
        if should_process:
            with st.spinner("Processando indicadores analíticos (Bollinger & Z-Score)..."):
                try:
                    raw_df = st.session_state["raw_data"]
                    # Executa motor matemático puro
                    processed_df = engine.process(
                        raw_df, 
                        period=bb_period, 
                        std_dev_mult=bb_std_dev, 
                        z_threshold=z_threshold
                    )
                    st.session_state["processed_data"] = processed_df
                    st.session_state["last_analytics_key"] = current_analytics_key
                    logger.info("Recálculo de indicadores analíticos finalizado com sucesso.")
                except Exception as err:
                    st.error(f"⚠️ Erro no Motor de Processamento: {err}")
                    st.session_state["processed_data"] = None
                    return
                    
    # 5. Renderização dos Componentes de Visualização
    if st.session_state["processed_data"] is not None:
        processed_df = st.session_state["processed_data"]
        
        # Criação das Abas na parte superior do Dashboard
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Monitoramento de Volatilidade", 
            "🔮 Previsão Probabilística", 
            "⚡ Probabilidade de Anomalias",
            "📈 Padrões de Volatilidade"
        ])
        
        with tab1:
            # Renderiza a linha superior com os cards de KPI customizados
            render_metrics_cards(processed_df)
            
            st.write("")
            
            # Título do Gráfico e visualização do Plotly integrado
            st.markdown(f"### Monitoramento e Gráfico Analítico de {symbol} ({interval})")
            render_candlestick_chart(processed_df, z_threshold=z_threshold)
            
        with tab2:
            st.markdown("### 🔮 Projeção Estatística de Curto Prazo")
            st.markdown("Previsões baseadas no consenso de modelos ARIMA, Holt (Suavização Exponencial) e Regressão Linear.")
            
            # Slider de passos futuros dentro da aba
            steps = st.slider(
                "Períodos de Previsão (Candles à frente):",
                min_value=3,
                max_value=15,
                value=7,
                step=1,
                help="Quantidade de períodos futuros para projetar os preços."
            )
            
            # Execução do motor de previsão
            forecaster = ForecastEngine(steps=steps)
            with st.spinner("Computando modelos preditivos..."):
                try:
                    forecast_df = forecaster.run_forecast(processed_df)
                    render_forecast_tab(processed_df, forecast_df, steps=steps)
                except Exception as err:
                    st.error(f"⚠️ Erro ao calcular as previsões: {err}")
                    
        with tab3:
            st.markdown("### ⚡ Análise de Risco e Probabilidade de Anomalias")
            st.markdown("Estatísticas baseadas na distribuição e transição de anomalias computadas em tempo real.")
            
            # Execução do motor probabilístico de anomalias
            anomaly_engine = AnomalyProbabilityEngine()
            with st.spinner("Computando modelos probabilísticos de risco..."):
                try:
                    anomaly_metrics = anomaly_engine.calculate_metrics(processed_df)
                    render_anomaly_probability_tab(processed_df, anomaly_metrics, z_threshold=z_threshold)
                except Exception as err:
                    st.error(f"⚠️ Erro ao calcular as probabilidades de anomalia: {err}")
                    
        with tab4:
            st.markdown("### 📈 Reconhecimento de Padrões e Projeção de Tendências")
            st.markdown("Previsões baseadas na resposta histórica de preços pós-anomalias (Estudo de Eventos de Alta e Baixa Volatilidade).")
            
            # Execução do motor de padrões de tendência por anomalias
            trend_engine = AnomalyTrendEngine()
            with st.spinner("Analisando padrões históricos pós-evento..."):
                try:
                    event_stats = trend_engine.analyze_patterns(processed_df, max_steps=5)
                    current_forecast = trend_engine.generate_current_forecast(processed_df, event_stats, lookback=10)
                    render_anomaly_trend_tab(processed_df, event_stats, current_forecast)
                except Exception as err:
                    st.error(f"⚠️ Erro ao analisar padrões de tendência: {err}")
        
        # Rodapé com o carimbo de data/hora
        last_update_utc = processed_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"<div style='text-align: right; color: #5a5d6a; font-size: 0.8rem; margin-top: 15px;'>"
            f"Último registro de candle: {last_update_utc} UTC"
            f"</div>", 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
