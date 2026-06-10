import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def apply_custom_css() -> None:
    """
    Injeta regras CSS customizadas via HTML markdown no Streamlit.
    Configura a tipografia (Orbitron/Rajdhani), cores de fundo, bordas e efeitos de neon.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;700&display=swap');
    
    /* Variáveis Globais */
    :root {
        --neon-cyan: #00ffcc;
        --neon-pink: #ff0055;
        --bg-color: #0c0d14;
        --card-bg: rgba(15, 15, 27, 0.75);
        --font-main: 'Rajdhani', sans-serif;
        --font-header: 'Orbitron', sans-serif;
    }
    
    /* Configuração Global do App */
    .stApp {
        background-color: var(--bg-color);
        background-image: radial-gradient(circle at 50% 10%, #171b30 0%, #0c0d14 100%);
        color: #e2e8f0;
        font-family: var(--font-main);
    }
    
    /* Títulos e Cabeçalhos */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-header) !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    h1 {
        color: var(--neon-cyan) !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5), 0 0 20px rgba(0, 255, 204, 0.3);
        font-weight: 900 !important;
        border-bottom: 2px solid rgba(0, 255, 204, 0.2);
        padding-bottom: 10px;
        margin-bottom: 30px !important;
    }
    
    h2, h3 {
        color: #ffffff !important;
        border-left: 3px solid var(--neon-pink);
        padding-left: 12px;
    }
    
    /* Customização da Sidebar */
    [data-testid="stSidebar"] {
        background-color: #08090f !important;
        border-right: 1px solid rgba(0, 255, 204, 0.2) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--neon-cyan) !important;
        border-left: none !important;
        padding-left: 0 !important;
        font-size: 1.4rem;
        text-shadow: 0 0 5px rgba(0, 255, 204, 0.3);
    }
    
    /* Estilo de Botões */
    div.stButton > button {
        background: transparent !important;
        color: var(--neon-cyan) !important;
        border: 1px solid var(--neon-cyan) !important;
        border-radius: 4px !important;
        padding: 0.5rem 1.5rem !important;
        font-family: var(--font-header) !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-shadow: 0 0 5px rgba(0, 255, 204, 0.5) !important;
        box-shadow: 0 0 5px rgba(0, 255, 204, 0.2) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background: var(--neon-cyan) !important;
        color: #000000 !important;
        box-shadow: 0 0 15px var(--neon-cyan) !important;
        text-shadow: none !important;
    }
    
    /* Estilização para as cards de métrica */
    .kpi-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
        flex-wrap: wrap;
    }
    
    .kpi-card {
        background: var(--card-bg);
        border: 1px solid rgba(0, 255, 204, 0.15);
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        flex: 1;
        min-width: 200px;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 255, 204, 0.2);
        border-color: rgba(0, 255, 204, 0.4);
    }
    
    .kpi-title {
        font-family: var(--font-header);
        font-size: 0.8rem;
        color: #8a8d9a;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    
    .kpi-value {
        font-family: var(--font-header);
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .kpi-subtitle {
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Configurações de inputs do Streamlit */
    div[data-baseweb="select"] > div {
        background-color: #08090f !important;
        border-color: rgba(0, 255, 204, 0.2) !important;
        color: white !important;
    }
    
    div[data-baseweb="input"] {
        background-color: #08090f !important;
        border-color: rgba(0, 255, 204, 0.2) !important;
    }
    
    div[data-baseweb="input"] input {
        color: white !important;
    }
    
    /* Ocultar elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)


def render_metrics_cards(df: pd.DataFrame) -> None:
    """
    Renderiza cards de KPI customizados em layout de grid de colunas no Streamlit.
    
    Args:
        df (pd.DataFrame): DataFrame processado contendo os dados e resultados analíticos.
                           Deve conter colunas ['close', 'volume', 'z_score', 'anomaly'].
    """
    if df is None or df.empty:
        st.warning("Nenhum dado disponível para exibir as métricas.")
        return
        
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else last_row
    
    close_val = float(last_row["close"])
    prev_close = float(prev_row["close"])
    pct_change = ((close_val - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
    
    volume_val = float(last_row["volume"])
    z_score_val = float(last_row["z_score"])
    is_anomaly = bool(last_row["anomaly"])
    
    # Formatações de strings
    price_str = f"${close_val:,.2f}" if close_val >= 1.0 else f"${close_val:.6f}"
    change_color = "var(--neon-cyan)" if pct_change >= 0 else "var(--neon-pink)"
    change_sign = "+" if pct_change >= 0 else ""
    change_str = f"{change_sign}{pct_change:.2f}%"
    
    # Formatação amigável para volume grande
    if volume_val >= 1_000_000_000:
        vol_str = f"{volume_val / 1_000_000_000:.2f}B"
    elif volume_val >= 1_000_000:
        vol_str = f"{volume_val / 1_000_000:.2f}M"
    else:
        vol_str = f"{volume_val:,.0f}"
        
    z_color = "var(--neon-pink)" if abs(z_score_val) > 2.0 else "var(--neon-cyan)"
    z_str = f"{z_score_val:+.2f}"
    
    anomaly_status = "CRÍTICO" if is_anomaly else "ESTÁVEL"
    anomaly_color = "var(--neon-pink)" if is_anomaly else "var(--neon-cyan)"
    anomaly_shadow = "0 0 15px #ff0055" if is_anomaly else "0 0 10px #00ffcc"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--neon-cyan);">
            <div class="kpi-title">Último Preço</div>
            <div class="kpi-value" style="color: #ffffff; text-shadow: 0 0 5px rgba(255,255,255,0.4);">{price_str}</div>
            <div class="kpi-subtitle" style="color: {change_color};">{change_str} (Último Candle)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--neon-cyan);">
            <div class="kpi-title">Volume 24h</div>
            <div class="kpi-value" style="color: #ffffff;">{vol_str}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Quantidade Negociada</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {z_color};">
            <div class="kpi-title">Z-Score Desvio</div>
            <div class="kpi-value" style="color: {z_color}; text-shadow: 0 0 5px {z_color}40;">{z_str}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Afastamento da Média</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {anomaly_color}; box-shadow: inset 0 0 10px {anomaly_color}10;">
            <div class="kpi-title">Volatilidade Status</div>
            <div class="kpi-value" style="color: {anomaly_color}; text-shadow: {anomaly_shadow}; font-weight: 700;">{anomaly_status}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Anomalias Detectadas</div>
        </div>
        """, unsafe_allow_html=True)


def render_candlestick_chart(df: pd.DataFrame, z_threshold: float = 2.0) -> None:
    """
    Renderiza o gráfico interativo de velas e Z-Score utilizando Plotly.
    Possui uma área para o preço + Bandas de Bollinger e outra inferior mostrando a evolução do Z-Score.
    
    Args:
        df (pd.DataFrame): DataFrame contendo as colunas:
                           ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'bb_middle', 'bb_upper', 'bb_lower', 'z_score', 'anomaly']
        z_threshold (float): O limiar absoluto de Z-score para exibição de linhas pontilhadas de alerta.
    """
    if df is None or df.empty:
        st.warning("Dados indisponíveis para plotagem do gráfico.")
        return
        
    # Inicializando subplot: Row 1 = Preço, Row 2 = Z-Score
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08, 
        row_heights=[0.7, 0.3]
    )
    
    # 1. Row 1: Gráfico de Velas (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles",
            increasing_line_color="#00ffcc",
            decreasing_line_color="#ff0055",
            increasing_fillcolor="rgba(0, 255, 204, 0.2)",
            decreasing_fillcolor="rgba(255, 0, 85, 0.2)",
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 2. Row 1: Média Móvel Central das Bandas de Bollinger
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bb_middle"],
            name="Média Móvel (BB)",
            line=dict(color="rgba(226, 232, 240, 0.5)", width=1.5, dash="dash"),
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    
    # 3. Row 1: Banda Inferior
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bb_lower"],
            name="Banda Inferior",
            line=dict(color="#ff0055", width=1.0),
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    
    # 4. Row 1: Banda Superior (e preenchimento entre as bandas)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["bb_upper"],
            name="Banda Superior",
            line=dict(color="#00ffcc", width=1.0),
            fill="tonexty",
            fillcolor="rgba(0, 255, 204, 0.02)",
            hoverinfo="skip"
        ),
        row=1, col=1
    )
    
    # 5. Row 1: Marcadores de Anomalias de Volatilidade
    anomalies = df[df["anomaly"] == True]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["close"],
                mode="markers",
                name="Anomalias",
                marker=dict(
                    color="#ffff00",
                    size=10,
                    symbol="circle-open-dot",
                    line=dict(color="#ffff00", width=2)
                ),
                hovertemplate="<b>Anomalia de Preço</b><br>Data: %{x}<br>Fechamento: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
    # 6. Row 2: Z-Score Line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["z_score"],
            name="Z-Score",
            line=dict(color="#00ffcc", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 204, 0.05)",
            hovertemplate="<b>Z-Score:</b> %{y:+.2f}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 7. Row 2: Linhas Horizontais de Alerta (Limiares de Z-Score)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=[z_threshold] * len(df),
            name="Limiar Superior",
            line=dict(color="rgba(255, 0, 85, 0.4)", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=[-z_threshold] * len(df),
            name="Limiar Inferior",
            line=dict(color="rgba(255, 0, 85, 0.4)", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=2, col=1
    )
    
    # Customização de Layout e Temas Futuristas no Plotly
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(12, 13, 20, 0.0)", # Fundo transparente para mesclar com o app
        plot_bgcolor="rgba(12, 13, 20, 0.3)",
        font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        height=600
    )
    
    # Customização específica dos eixos
    fig.update_xaxes(
        showgrid=True, 
        gridcolor="rgba(31, 34, 53, 0.4)", 
        linecolor="rgba(0, 255, 204, 0.2)",
        zeroline=False
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridcolor="rgba(31, 34, 53, 0.4)", 
        linecolor="rgba(0, 255, 204, 0.2)",
        zeroline=False
    )
    
    # Customização do eixo Y2 (Z-Score)
    fig.update_yaxes(
        title_text="Z-Score", 
        row=2, col=1, 
        title_font=dict(size=12, color="#00ffcc"),
        range=[-max(abs(df["z_score"].max()), abs(df["z_score"].min()), z_threshold) - 0.5,
               max(abs(df["z_score"].max()), abs(df["z_score"].min()), z_threshold) + 0.5]
    )
    
    fig.update_yaxes(
        title_text="Preço", 
        row=1, col=1, 
        title_font=dict(size=12, color="#00ffcc")
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_tab(df: pd.DataFrame, forecast_df: pd.DataFrame, steps: int) -> None:
    """
    Renderiza a aba de Previsão Probabilística contendo cards de tendência,
    dados projetados e o gráfico Plotly estendido com intervalos de confiança.
    """
    if df is None or df.empty or forecast_df is None or forecast_df.empty:
        st.warning("Dados insuficientes para renderizar a previsão.")
        return

    # 1. Recuperar preços finais e variação estimada
    last_price = float(df["close"].iloc[-1])
    target_price = float(forecast_df["consensus"].iloc[-1])
    pct_change = ((target_price - last_price) / last_price) * 100
    
    # Detalhes do intervalo de confiança final
    lower_price = float(forecast_df["lower_bound"].iloc[-1])
    upper_price = float(forecast_df["upper_bound"].iloc[-1])
    
    # 2. Configurar estados e cores de tendência
    if pct_change > 1.5:
        trend_status = "ALTA"
        trend_color = "#00ffcc" # var(--neon-cyan)
        trend_shadow = "0 0 10px #00ffcc"
    elif pct_change < -1.5:
        trend_status = "BAIXA"
        trend_color = "#ff0055" # var(--neon-pink)
        trend_shadow = "0 0 10px #ff0055"
    else:
        trend_status = "NEUTRA"
        trend_color = "#ffff00" # yellow
        trend_shadow = "0 0 10px #ffff00"
        
    change_sign = "+" if pct_change >= 0 else ""
    change_str = f"{change_sign}{pct_change:.2f}%"
    
    # Formatação de preços
    fmt_price = lambda x: f"${x:,.2f}" if x >= 1.0 else f"${x:.6f}"
    
    # 3. Exibir KPIs de Previsão
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #ffff00;">
            <div class="kpi-title">Projeção Consenso</div>
            <div class="kpi-value" style="color: #ffffff;">{fmt_price(target_price)}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Em {steps} períodos</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {trend_color}; box-shadow: inset 0 0 10px {trend_color}10;">
            <div class="kpi-title">Tendência Esperada</div>
            <div class="kpi-value" style="color: {trend_color}; text-shadow: {trend_shadow}; font-weight: 700;">{trend_status}</div>
            <div class="kpi-subtitle" style="color: {trend_color};">{change_str} esperados</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #00ffcc;">
            <div class="kpi-title">Limite Superior (95%)</div>
            <div class="kpi-value" style="color: #ffffff;">{fmt_price(upper_price)}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Cenário Mais Otimista</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #ff0055;">
            <div class="kpi-title">Limite Inferior (95%)</div>
            <div class="kpi-value" style="color: #ffffff;">{fmt_price(lower_price)}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Cenário Mais Crítico</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # 4. Criar o gráfico de previsão interativo no Plotly
    # Mostra apenas os últimos 60 candles para não poluir o gráfico
    recent_history_len = min(60, len(df))
    hist_subset = df.iloc[-recent_history_len:]
    
    fig = go.Figure()
    
    # Linha do preço histórico real
    fig.add_trace(go.Scatter(
        x=hist_subset["timestamp"],
        y=hist_subset["close"],
        name="Preço Histórico",
        line=dict(color="#ffffff", width=2)
    ))
    
    # Para ligar visualmente o histórico ao futuro, adicionamos o último ponto histórico ao início das projeções
    last_hist_row = hist_subset.iloc[-1]
    
    # Criamos séries conectadas
    future_x = pd.concat([pd.Series([last_hist_row["timestamp"]]), forecast_df["timestamp"]])
    
    # Função auxiliar para conectar projeção
    def get_connected_series(forecast_col):
        return pd.concat([pd.Series([last_hist_row["close"]]), forecast_df[forecast_col]])
        
    # Área sombreada do intervalo de confiança
    lower_bound_conn = get_connected_series("lower_bound")
    upper_bound_conn = get_connected_series("upper_bound")
    
    fig.add_trace(go.Scatter(
        x=future_x,
        y=upper_bound_conn,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    fig.add_trace(go.Scatter(
        x=future_x,
        y=lower_bound_conn,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(255, 255, 0, 0.08)", # Sombreado amarelo discreto
        name="Intervalo de Confiança (95%)",
        hoverinfo="skip"
    ))
    
    # Projeção ARIMA
    fig.add_trace(go.Scatter(
        x=future_x,
        y=get_connected_series("arima"),
        name="ARIMA",
        line=dict(color="#ff0055", width=1.5, dash="dot")
    ))
    
    # Projeção Holt
    fig.add_trace(go.Scatter(
        x=future_x,
        y=get_connected_series("holt_winters"),
        name="Holt-Winters",
        line=dict(color="#00ffcc", width=1.5, dash="dot")
    ))
    
    # Projeção Regressão Linear
    fig.add_trace(go.Scatter(
        x=future_x,
        y=get_connected_series("linear_regression"),
        name="Regressão Linear",
        line=dict(color="#8a8d9a", width=1.5, dash="dot")
    ))
    
    # Projeção Consenso
    fig.add_trace(go.Scatter(
        x=future_x,
        y=get_connected_series("consensus"),
        name="Consenso (Média)",
        line=dict(color="#ffff00", width=2.5) # Linha amarela de destaque
    ))
    
    # Customização visual do Plotly (Tema Futurista / Cyberpunk)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(12, 13, 20, 0.0)",
        plot_bgcolor="rgba(12, 13, 20, 0.3)",
        font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
        hovermode="x unified",
        margin=dict(t=30, b=10, l=10, r=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        height=550
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(31, 34, 53, 0.4)",
        linecolor="rgba(0, 255, 204, 0.2)",
        zeroline=False
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(31, 34, 53, 0.4)",
        linecolor="rgba(0, 255, 204, 0.2)",
        zeroline=False,
        title_text="Preço",
        title_font=dict(size=12, color="#00ffcc")
    )
    
    st.plotly_chart(fig, use_container_width=True)

