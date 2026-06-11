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


def render_forecast_tab(df: pd.DataFrame, forecast_df: pd.DataFrame, steps: int, mc_result: dict = None, merton_result: dict = None, kalman_result: dict = None) -> None:
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

    # 5. Seção da Simulação de Monte Carlo
    if mc_result is not None:
        st.write("")
        st.markdown("##### 🎲 Caminhos Simulados de Preço (Simulação de Monte Carlo)")
        st.write("Exibe 150 caminhos futuros simulados aleatoriamente com base na volatilidade e drift históricos.")
        
        fig_mc = go.Figure()
        
        # Plotar caminhos simulados
        timestamps_mc = mc_result["timestamps"]
        paths = mc_result["paths"]
        
        # Mostra apenas um subconjunto de caminhos para não travar a renderização do Plotly (ex: 50 caminhos)
        num_to_plot = min(50, len(paths[0]))
        for i in range(num_to_plot):
            path_y = [paths[t][i] for t in range(len(paths))]
            fig_mc.add_trace(go.Scatter(
                x=timestamps_mc,
                y=path_y,
                mode="lines",
                line=dict(color="rgba(138, 141, 154, 0.08)", width=1),
                showlegend=False,
                hoverinfo="skip"
            ))
            
        # Sombreado da área de confiança 5% - 95%
        fig_mc.add_trace(go.Scatter(
            x=timestamps_mc,
            y=mc_result["percentile_95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig_mc.add_trace(go.Scatter(
            x=timestamps_mc,
            y=mc_result["percentile_5"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 235, 59, 0.05)", # sombreado amarelo
            name="Intervalo de Caminhos (5% - 95%)",
            hoverinfo="skip"
        ))
        
        # Mediana simulada
        fig_mc.add_trace(go.Scatter(
            x=timestamps_mc,
            y=mc_result["percentile_50"],
            name="Caminho Mediano (50%)",
            line=dict(color="#ffff00", width=2.5)
        ))
        
        # Histórico recente
        fig_mc.add_trace(go.Scatter(
            x=hist_subset["timestamp"],
            y=hist_subset["close"],
            name="Histórico Real",
            line=dict(color="#ffffff", width=2)
        ))
        
        fig_mc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            hovermode="x unified",
            margin=dict(t=15, b=10, l=10, r=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11)
            ),
            height=400
        )
        
        fig_mc.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        fig_mc.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        
        st.plotly_chart(fig_mc, use_container_width=True)

    # 6. Seção Merton Jump-Diffusion
    if merton_result is not None:
        st.write("")
        st.markdown("##### 💥 Simulação de Merton Jump-Diffusion (Saltos de Preço)")
        st.write("Extensão do Monte Carlo com saltos estocásticos via Processo de Poisson Composto — modela crashes e pumps súbitos.")
        
        # KPIs do Merton
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e040fb;">
                <div class="kpi-title">Frequência de Saltos (λ)</div>
                <div class="kpi-value" style="color: #e040fb;">{merton_result.get('lambda_jumps', 0):.4f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Saltos por período</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e040fb;">
                <div class="kpi-title">Magnitude Média (μ_J)</div>
                <div class="kpi-value" style="color: #ffffff;">{merton_result.get('mu_jump', 0)*100:+.2f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Tamanho médio do salto</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e040fb;">
                <div class="kpi-title">Volatilidade do Salto (σ_J)</div>
                <div class="kpi-value" style="color: #ffffff;">{merton_result.get('sigma_jump', 0)*100:.2f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Dispersão dos saltos</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e040fb;">
                <div class="kpi-title">Saltos Médios por Caminho</div>
                <div class="kpi-value" style="color: #ffffff;">{merton_result.get('avg_jumps_per_path', 0):.1f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Em {steps} períodos simulados</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Gráfico Merton
        fig_merton = go.Figure()
        timestamps_m = merton_result["timestamps"]
        
        # Banda de confiança Merton
        fig_merton.add_trace(go.Scatter(
            x=timestamps_m, y=merton_result["percentile_95"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig_merton.add_trace(go.Scatter(
            x=timestamps_m, y=merton_result["percentile_5"],
            mode="lines", fill="tonexty",
            fillcolor="rgba(224, 64, 251, 0.08)",
            name="Intervalo Merton (5%-95%)", hoverinfo="skip"
        ))
        
        # Mediana Merton
        fig_merton.add_trace(go.Scatter(
            x=timestamps_m, y=merton_result["percentile_50"],
            name="Mediana Merton (com Saltos)",
            line=dict(color="#e040fb", width=2.5)
        ))
        
        # GBM puro para comparação
        if "gbm_only_median" in merton_result:
            fig_merton.add_trace(go.Scatter(
                x=timestamps_m, y=merton_result["gbm_only_median"],
                name="GBM Puro (sem Saltos)",
                line=dict(color="#8a8d9a", width=1.5, dash="dot")
            ))
        
        # Histórico
        fig_merton.add_trace(go.Scatter(
            x=hist_subset["timestamp"], y=hist_subset["close"],
            name="Histórico Real", line=dict(color="#ffffff", width=2)
        ))
        
        fig_merton.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            hovermode="x unified",
            margin=dict(t=15, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            height=400
        )
        fig_merton.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        fig_merton.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        st.plotly_chart(fig_merton, use_container_width=True)

    # 7. Seção Filtro de Kalman
    if kalman_result is not None:
        st.write("")
        st.markdown("##### 🎯 Filtro de Kalman (Preço Denoised e Projeção Adaptativa)")
        st.write("Estimador bayesiano linear que filtra o ruído do preço observado para revelar a tendência real subjacente.")
        
        # KPIs Kalman
        k_col1, k_col2 = st.columns(2)
        with k_col1:
            noise_r = kalman_result.get("noise_ratio", 0)
            noise_label = "Alto Ruído" if noise_r > 5 else ("Moderado" if noise_r > 1 else "Baixo Ruído")
            noise_color = "#ff0055" if noise_r > 5 else ("#ffff00" if noise_r > 1 else "#00ffcc")
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {noise_color};">
                <div class="kpi-title">Razão Ruído/Processo (R/Q)</div>
                <div class="kpi-value" style="color: {noise_color};">{noise_r:.2f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">{noise_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with k_col2:
            if kalman_result.get("forecast_prices"):
                kalman_target = kalman_result["forecast_prices"][-1]
                fmt_kt = f"${kalman_target:,.2f}" if kalman_target >= 1.0 else f"${kalman_target:.6f}"
            else:
                fmt_kt = "N/A"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #40c4ff;">
                <div class="kpi-title">Preço Projetado (Kalman)</div>
                <div class="kpi-value" style="color: #40c4ff;">{fmt_kt}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Em {steps} períodos</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Gráfico Kalman
        fig_kalman = go.Figure()
        
        # Preço real (últimos 60)
        fig_kalman.add_trace(go.Scatter(
            x=hist_subset["timestamp"], y=hist_subset["close"],
            name="Preço Observado", line=dict(color="rgba(255,255,255,0.4)", width=1)
        ))
        
        # Preço filtrado
        filtered_len = len(kalman_result["filtered_prices"])
        kalman_hist_idx = max(0, filtered_len - len(hist_subset))
        filtered_slice = kalman_result["filtered_prices"][kalman_hist_idx:]
        upper_slice = kalman_result.get("filtered_upper", [])[kalman_hist_idx:]
        lower_slice = kalman_result.get("filtered_lower", [])[kalman_hist_idx:]
        
        if len(filtered_slice) == len(hist_subset):
            fig_kalman.add_trace(go.Scatter(
                x=hist_subset["timestamp"], y=upper_slice,
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig_kalman.add_trace(go.Scatter(
                x=hist_subset["timestamp"], y=lower_slice,
                mode="lines", fill="tonexty",
                fillcolor="rgba(64, 196, 255, 0.06)",
                name="Banda de Incerteza Kalman", hoverinfo="skip"
            ))
            fig_kalman.add_trace(go.Scatter(
                x=hist_subset["timestamp"], y=filtered_slice,
                name="Preço Filtrado (Kalman)",
                line=dict(color="#40c4ff", width=2.5)
            ))
        
        # Projeção futura
        if kalman_result.get("forecast_prices") and kalman_result.get("forecast_timestamps"):
            fc_ts = kalman_result["forecast_timestamps"]
            fc_p = kalman_result["forecast_prices"]
            fc_u = kalman_result.get("forecast_upper", fc_p)
            fc_l = kalman_result.get("forecast_lower", fc_p)
            
            # Conectar ao último ponto
            import pandas as _pd
            last_ts = hist_subset["timestamp"].iloc[-1]
            last_fp = filtered_slice[-1] if filtered_slice else float(hist_subset["close"].iloc[-1])
            
            conn_ts = [last_ts] + list(fc_ts)
            conn_p = [last_fp] + list(fc_p)
            conn_u = [last_fp] + list(fc_u)
            conn_l = [last_fp] + list(fc_l)
            
            fig_kalman.add_trace(go.Scatter(
                x=conn_ts, y=conn_u,
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig_kalman.add_trace(go.Scatter(
                x=conn_ts, y=conn_l,
                mode="lines", fill="tonexty",
                fillcolor="rgba(64, 196, 255, 0.12)",
                name="Banda Projeção Kalman", hoverinfo="skip"
            ))
            fig_kalman.add_trace(go.Scatter(
                x=conn_ts, y=conn_p,
                name="Projeção Kalman",
                line=dict(color="#40c4ff", width=2.5, dash="dash")
            ))
        
        fig_kalman.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            hovermode="x unified",
            margin=dict(t=15, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            height=400
        )
        fig_kalman.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        fig_kalman.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        st.plotly_chart(fig_kalman, use_container_width=True)


def render_anomaly_probability_tab(df: pd.DataFrame, metrics: dict, z_threshold: float, garch_result: dict = None, pf_result: dict = None, evt_result: dict = None) -> None:
    """
    Renderiza a aba de Análise de Risco e Probabilidade de Anomalias.
    Exibe indicadores de risco imediato, matriz de transição de Markov e curva de Poisson.
    """
    if df is None or df.empty or metrics is None:
        st.warning("Dados insuficientes para renderizar a análise probabilística de anomalias.")
        return

    # 1. KPIs de Probabilidade
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid var(--neon-cyan);">
            <div class="kpi-title">Limiar Z-Score Atual</div>
            <div class="kpi-value" style="color: var(--neon-cyan); text-shadow: 0 0 5px rgba(0,255,204,0.4);">{z_threshold:,.1f}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Sensibilidade de Filtro</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        risk_color = "var(--neon-pink)" if metrics["immediate_risk"] > 10 else "var(--neon-cyan)"
        risk_shadow = "0 0 10px #ff0055" if metrics["immediate_risk"] > 10 else "0 0 10px #00ffcc"
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {risk_color}; box-shadow: inset 0 0 10px {risk_color}10;">
            <div class="kpi-title">Risco Próximo Período</div>
            <div class="kpi-value" style="color: {risk_color}; text-shadow: {risk_shadow}; font-weight: 700;">{metrics["immediate_risk"]:.1f}%</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Condicionado ao Estado Atual</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #ffff00;">
            <div class="kpi-title">Frequência de Anomalia</div>
            <div class="kpi-value" style="color: #ffffff;">{metrics["overall_probability"]:.1f}%</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">{metrics["num_anomalies"]} de {metrics["total_points"]} candles</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #e2e8f0;">
            <div class="kpi-title">Distância Média</div>
            <div class="kpi-value" style="color: #ffffff;">{metrics["mean_distance"]:.1f}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Candles entre ocorrências</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 2. Layout do Corpo Principal: Gráficos de Markov e Poisson
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 🔄 Matriz de Transição de Markov (Probabilidade de Mudança)")
        st.write("Exibe a probabilidade de transição entre estados de mercado de um período para o outro.")
        
        # Matriz de Transição
        matrix = metrics["markov_matrix"]
        z_matrix = [
            [matrix.get("S_to_S", 0.0), matrix.get("S_to_A", 0.0)],
            [matrix.get("A_to_S", 0.0), matrix.get("A_to_A", 0.0)]
        ]
        
        fig_markov = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=["Ir para: Estável", "Ir para: Anomalia"],
            y=["De: Estável", "De: Anomalia"],
            colorscale=[[0, "rgba(12, 13, 20, 0.95)"], [0.5, "rgba(0, 255, 204, 0.4)"], [1, "rgba(255, 0, 85, 0.75)"]],
            text=[[f"{z_matrix[0][0]:.1f}%", f"{z_matrix[0][1]:.1f}%"],
                  [f"{z_matrix[1][0]:.1f}%", f"{z_matrix[1][1]:.1f}%"]],
            texttemplate="%{text}",
            textfont=dict(family="Rajdhani, sans-serif", size=16, color="#ffffff"),
            showscale=False
        ))
        
        fig_markov.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            height=300,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_markov, use_container_width=True)

    with chart_col2:
        st.markdown("#### ⏳ Risco Acumulado nos Próximos Períodos (Processo de Poisson)")
        st.write("A probabilidade de enfrentar ao menos uma nova anomalia nas próximas velas à frente.")
        
        fig_poisson = go.Figure()
        fig_poisson.add_trace(go.Scatter(
            x=metrics["poisson_curve"]["steps"],
            y=metrics["poisson_curve"]["probabilities"],
            mode="lines+markers",
            name="Risco Acumulado",
            line=dict(color="#ffff00", width=2),
            marker=dict(color="#ffff00", size=6),
            hovertemplate="<b>Períodos à frente:</b> %{x}<br><b>Probabilidade:</b> %{y:.1f}%<extra></extra>"
        ))
        
        fig_poisson.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(title="Número de Períodos Futuros (Candles)"),
            yaxis=dict(title="Probabilidade (%)", range=[0, 100])
        )
        
        fig_poisson.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        fig_poisson.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        st.plotly_chart(fig_poisson, use_container_width=True)

    # 3. Análise de Fatores Temporais (Janelas de Volatilidade)
    if metrics["hourly_distribution"] or metrics["weekday_distribution"]:
        st.write("")
        st.markdown("#### ⏰ Perfil Temporal de Risco e Volatilidade")
        st.write("Identificação das janelas de tempo com maior propensão histórica à ocorrência de anomalias.")

        temp_col1, temp_col2 = st.columns(2)

        with temp_col1:
            if metrics["hourly_distribution"]:
                hours = [item["hour"] for item in metrics["hourly_distribution"]]
                rates = [item["rate"] for item in metrics["hourly_distribution"]]
                
                fig_hour = go.Figure(data=go.Bar(
                    x=hours,
                    y=rates,
                    marker_color="rgba(0, 255, 204, 0.65)",
                    marker_line=dict(color="var(--neon-cyan)", width=1),
                    hovertemplate="<b>Hora UTC:</b> %{x}h<br><b>Taxa de Anomalia:</b> %{y:.1f}%<extra></extra>"
                ))
                
                fig_hour.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(12, 13, 20, 0.0)",
                    plot_bgcolor="rgba(12, 13, 20, 0.3)",
                    font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis=dict(title="Hora do Dia (UTC)", tickmode="linear", tick0=0, dtick=2),
                    yaxis=dict(title="Taxa de Anomalia (%)")
                )
                fig_hour.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                fig_hour.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                st.plotly_chart(fig_hour, use_container_width=True)
            else:
                st.info("Distribuição horária indisponível.")

        with temp_col2:
            if metrics["weekday_distribution"]:
                days = [item["day"] for item in metrics["weekday_distribution"]]
                rates = [item["rate"] for item in metrics["weekday_distribution"]]
                
                # Tradução amigável dos dias da semana
                day_translations = {
                    "Monday": "Seg", "Tuesday": "Ter", "Wednesday": "Qua",
                    "Thursday": "Qui", "Friday": "Sex", "Saturday": "Sáb", "Sunday": "Dom"
                }
                translated_days = [day_translations.get(day, day[:3]) for day in days]
                
                fig_day = go.Figure(data=go.Bar(
                    x=translated_days,
                    y=rates,
                    marker_color="rgba(255, 0, 85, 0.65)",
                    marker_line=dict(color="var(--neon-pink)", width=1),
                    hovertemplate="<b>Dia:</b> %{x}<br><b>Taxa de Anomalia:</b> %{y:.1f}%<extra></extra>"
                ))
                
                fig_day.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(12, 13, 20, 0.0)",
                    plot_bgcolor="rgba(12, 13, 20, 0.3)",
                    font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis=dict(title="Dia da Semana"),
                    yaxis=dict(title="Taxa de Anomalia (%)")
                )
                fig_day.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                fig_day.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                st.plotly_chart(fig_day, use_container_width=True)
            else:
                st.info("Distribuição semanal indisponível.")

    # 4. Seção do Modelo GARCH(1,1)
    if garch_result is not None:
        st.write("")
        st.markdown("#### 📈 Previsão de Volatilidade Dinâmica GARCH(1,1)")
        st.write("Estima a variação de risco condicional com reversão automática para a média de longo prazo.")
        
        # Colunas de métricas do GARCH
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        
        with g_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid var(--neon-cyan);">
                <div class="kpi-title">Volatilidade Atual (GARCH)</div>
                <div class="kpi-value" style="color: var(--neon-cyan);">{garch_result["current_vol"]:.2f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Último Candle Calculado</div>
            </div>
            """, unsafe_allow_html=True)
            
        with g_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e2e8f0;">
                <div class="kpi-title">Volatilidade de Longo Prazo</div>
                <div class="kpi-value" style="color: #ffffff;">{garch_result["vol_long_term"]:.2f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Média Estacionária (Uncondicional)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with g_col3:
            persistence = garch_result["persistence"]
            p_desc = "Estável (Reversão à Média)" if persistence < 1.0 else "Instável (Passeio Aleatório)"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #ffff00;">
                <div class="kpi-title">Persistência da Volatilidade</div>
                <div class="kpi-value" style="color: #ffff00;">{persistence:.4f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">{p_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with g_col4:
            opt_status = "SUCESSO" if garch_result["success"] else "FALLBACK"
            opt_color = "var(--neon-cyan)" if garch_result["success"] else "var(--neon-pink)"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {opt_color};">
                <div class="kpi-title">Calibração GARCH MLE</div>
                <div class="kpi-value" style="color: {opt_color};">{opt_status}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">ARCH (α): {garch_result["alpha"]:.3f} | GARCH (β): {garch_result["beta"]:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Gráfico da Volatilidade GARCH
        fig_garch = go.Figure()
        
        # Histórico recente da volatilidade (últimos 45 candles)
        history_len = min(45, len(garch_result["history_vol"]))
        hist_vol = garch_result["history_vol"][-history_len:]
        hist_timestamps = df["timestamp"].iloc[-history_len:]
        
        fig_garch.add_trace(go.Scatter(
            x=hist_timestamps,
            y=hist_vol,
            name="Volatilidade Condicional Estimada",
            line=dict(color="#ffffff", width=2)
        ))
        
        # Conexão entre o último ponto histórico e o primeiro futuro
        last_hist_time = df["timestamp"].iloc[-1]
        last_hist_vol = garch_result["history_vol"][-1]
        
        future_x = [last_hist_time] + garch_result["timestamps"]
        future_y = [last_hist_vol] + garch_result["forecast_vol"]
        
        # Projeção futura da volatilidade
        fig_garch.add_trace(go.Scatter(
            x=future_x,
            y=future_y,
            name="Projeção GARCH(1,1)",
            line=dict(color="#ffff00", width=2.5)
        ))
        
        # Linha horizontal de longo prazo
        fig_garch.add_trace(go.Scatter(
            x=future_x,
            y=[garch_result["vol_long_term"]] * len(future_x),
            name="Volatilidade de Longo Prazo",
            line=dict(color="rgba(255, 0, 85, 0.5)", width=1.5, dash="dash")
        ))
        
        fig_garch.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            hovermode="x unified",
            margin=dict(t=15, b=10, l=10, r=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11)
            ),
            height=350,
            yaxis=dict(title="Volatilidade (%)", ticksuffix="%")
        )
        
        fig_garch.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        fig_garch.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        
        st.plotly_chart(fig_garch, use_container_width=True)

    # 5. Seção Filtro de Partículas
    if pf_result is not None:
        st.write("")
        st.markdown("#### 🔬 Filtro de Partículas (Estimação Bayesiana de Volatilidade Latente)")
        st.write("Sequential Monte Carlo com reamostragem sistemática para estimar o regime de volatilidade oculto do mercado.")
        
        pf_col1, pf_col2 = st.columns(2)
        with pf_col1:
            regime = pf_result.get("current_regime", "N/A")
            regime_prob = pf_result.get("current_regime_prob", 0)
            regime_color = "#ff0055" if regime == "HIGH_VOL" else "#00ffcc"
            regime_label = "ALTA VOLATILIDADE" if regime == "HIGH_VOL" else "BAIXA VOLATILIDADE"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {regime_color}; box-shadow: inset 0 0 10px {regime_color}10;">
                <div class="kpi-title">Regime de Volatilidade Latente</div>
                <div class="kpi-value" style="color: {regime_color}; text-shadow: 0 0 10px {regime_color}; font-weight: 700;">{regime_label}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Probabilidade: {regime_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with pf_col2:
            ess_vals = pf_result.get("effective_sample_size", [])
            avg_ess = np.mean(ess_vals) if ess_vals else 0
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #ffff00;">
                <div class="kpi-title">Eficiência Amostral Média (ESS)</div>
                <div class="kpi-value" style="color: #ffff00;">{avg_ess:.0f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Qualidade da estimação</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Gráfico do Filtro de Partículas
        pf_timestamps = pf_result.get("timestamps", [])
        pf_vol = pf_result.get("volatility_estimate", [])
        pf_upper = pf_result.get("volatility_upper", [])
        pf_lower = pf_result.get("volatility_lower", [])
        
        if pf_timestamps and pf_vol:
            # Mostrar últimos 60
            pf_show = min(60, len(pf_timestamps))
            
            fig_pf = go.Figure()
            fig_pf.add_trace(go.Scatter(
                x=pf_timestamps[-pf_show:], y=pf_upper[-pf_show:],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig_pf.add_trace(go.Scatter(
                x=pf_timestamps[-pf_show:], y=pf_lower[-pf_show:],
                mode="lines", fill="tonexty",
                fillcolor="rgba(255, 235, 59, 0.08)",
                name="Banda de Partículas (5%-95%)", hoverinfo="skip"
            ))
            fig_pf.add_trace(go.Scatter(
                x=pf_timestamps[-pf_show:], y=pf_vol[-pf_show:],
                name="Volatilidade Estimada (Posterior)",
                line=dict(color="#ffff00", width=2.5),
                hovertemplate="<b>Vol:</b> %{y:.4f}<extra></extra>"
            ))
            
            fig_pf.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(12, 13, 20, 0.0)",
                plot_bgcolor="rgba(12, 13, 20, 0.3)",
                font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                hovermode="x unified",
                margin=dict(t=15, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                height=300,
                yaxis=dict(title="Volatilidade Latente")
            )
            fig_pf.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
            fig_pf.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
            st.plotly_chart(fig_pf, use_container_width=True)

    # 6. Seção EVT + GMM
    if evt_result is not None:
        st.write("")
        st.markdown("#### 📊 Teoria de Valores Extremos (EVT) + Mistura Gaussiana (GMM)")
        st.write("Modelagem de caudas pesadas via Distribuição de Pareto Generalizada e identificação de regimes via clusterização gaussiana.")
        
        evt_col1, evt_col2, evt_col3, evt_col4 = st.columns(4)
        with evt_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #ff0055;">
                <div class="kpi-title">Value at Risk (95%)</div>
                <div class="kpi-value" style="color: #ff0055;">{evt_result.get('var_95', 0)*100:.2f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Perda máx. esperada</div>
            </div>
            """, unsafe_allow_html=True)
        with evt_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #ff0055;">
                <div class="kpi-title">CVaR / Expected Shortfall</div>
                <div class="kpi-value" style="color: #ff0055;">{evt_result.get('cvar_95', 0)*100:.2f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Perda média na cauda</div>
            </div>
            """, unsafe_allow_html=True)
        with evt_col3:
            gpd_shape = evt_result.get("gpd_shape", 0)
            shape_label = "Cauda Pesada" if gpd_shape > 0 else "Cauda Leve"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e040fb;">
                <div class="kpi-title">Forma GPD (ξ)</div>
                <div class="kpi-value" style="color: #e040fb;">{gpd_shape:.4f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">{shape_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with evt_col4:
            current_gmm_regime = evt_result.get("current_regime", "N/A")
            gmm_prob = evt_result.get("current_regime_prob", 0)
            gmm_color = "#ff0055" if "Alta" in current_gmm_regime else ("#ffff00" if "Média" in current_gmm_regime else "#00ffcc")
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {gmm_color};">
                <div class="kpi-title">Regime GMM Atual</div>
                <div class="kpi-value" style="color: {gmm_color}; font-size: 1.4rem;">{current_gmm_regime.upper()}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Prob: {gmm_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Gráfico dos retornos com regimes GMM colorizados
        returns = evt_result.get("returns", [])
        regime_labels = evt_result.get("gmm_regime_labels", [])
        
        if returns and regime_labels and len(returns) == len(regime_labels):
            show_n = min(60, len(returns))
            ret_slice = returns[-show_n:]
            lab_slice = regime_labels[-show_n:]
            ts_slice = df["timestamp"].iloc[-show_n:] if len(df) >= show_n else df["timestamp"]
            
            fig_gmm = go.Figure()
            
            regime_colors = {
                "Baixa Volatilidade": "#00ffcc",
                "Média Volatilidade": "#ffff00",
                "Alta Volatilidade": "#ff0055"
            }
            
            for regime_name, color in regime_colors.items():
                mask = [1 if l == regime_name else None for l in lab_slice]
                y_vals = [r * 100 if m == 1 else None for r, m in zip(ret_slice, mask)]
                fig_gmm.add_trace(go.Bar(
                    x=list(ts_slice), y=y_vals,
                    name=regime_name,
                    marker_color=color,
                    opacity=0.7
                ))
            
            fig_gmm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(12, 13, 20, 0.0)",
                plot_bgcolor="rgba(12, 13, 20, 0.3)",
                font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                barmode="overlay",
                margin=dict(t=15, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                height=280,
                yaxis=dict(title="Retorno (%)", ticksuffix="%"),
                xaxis=dict(title="")
            )
            fig_gmm.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
            fig_gmm.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
            st.plotly_chart(fig_gmm, use_container_width=True)


def render_anomaly_trend_tab(df: pd.DataFrame, event_stats: dict, current_forecast: dict) -> None:
    """
    Renderiza a aba de Padrões e Projeção de Tendência baseada em anomalias.
    Exibe a previsão ativa do evento atual (se houver) e estatísticas históricas de retornos médios.
    """
    if df is None or df.empty or event_stats is None or current_forecast is None:
        st.warning("Dados insuficientes para renderizar a projeção por anomalias.")
        return

    fmt_price = lambda x: f"${x:,.2f}" if x >= 1.0 else f"${x:.6f}"

    # 1. Seção de Previsão Ativa
    st.markdown("#### 🎯 Projeção de Tendência Ativa (Efeito de Anomalia Recente)")
    
    if current_forecast.get("active", False):
        direction = current_forecast["direction"]
        color = "var(--neon-cyan)" if direction == "ALTA" else "var(--neon-pink)"
        shadow = "0 0 15px #00ffcc" if direction == "ALTA" else "0 0 15px #ff0055"
        
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {color}; box-shadow: inset 0 0 10px {color}10; height: 100%;">
                <div class="kpi-title">Tendência Projetada</div>
                <div class="kpi-value" style="color: {color}; text-shadow: {shadow}; font-weight: 700; font-size: 2.2rem; margin-top: 10px;">{direction}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a; margin-top: 5px;">Baseado em Padrão Histórico</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #ffff00; height: 100%;">
                <div class="kpi-title">Alvo e Faixa de Preço Estimados</div>
                <div class="kpi-value" style="color: #ffffff; font-size: 2rem;">{fmt_price(current_forecast["target_price"])}</div>
                <div class="kpi-subtitle" style="color: #ffff00;">
                    Margem: {fmt_price(current_forecast["lower_target"])} a {fmt_price(current_forecast["upper_target"])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e2e8f0; height: 100%;">
                <div class="kpi-title">Grau de Confiança</div>
                <div class="kpi-value" style="color: #ffffff; font-size: 2.2rem;">{current_forecast["confidence"]:.1f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Amostra: {current_forecast["historical_events"]} eventos anteriores</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Alerta detalhado do evento
        st.write("")
        st.info(
            f"⚡ **Efeito Ativo:** Uma anomalia do tipo **{current_forecast['anomaly_type']}** ocorreu há **{current_forecast['elapsed_candles']}** candles atrás "
            f"(Preço de Fechamento na anomalia: **{fmt_price(current_forecast['anomaly_price'])}** em **{current_forecast['anomaly_timestamp']} UTC**). "
            f"O tempo restante estimado para o padrão se concretizar é de **{current_forecast['remaining_candles']}** candles."
        )
    else:
        # Mensagem estilizada indicando ausência de anomalia ativa
        st.markdown(f"""
        <div class="kpi-card" style="border: 1px dashed rgba(255,255,255,0.15); padding: 30px; text-align: center; margin-bottom: 25px;">
            <div class="kpi-title" style="font-size: 0.95rem; color: #8a8d9a; letter-spacing: 2px;">STATUS DO MERCADO</div>
            <div class="kpi-value" style="color: var(--neon-cyan); text-shadow: 0 0 10px rgba(0, 255, 204, 0.4); font-size: 1.6rem; font-weight: 700; margin: 12px 0;">ESTÁVEL / NEUTRO</div>
            <div class="kpi-subtitle" style="color: #8a8d9a; max-width: 600px; margin: 0 auto;">
                {current_forecast.get("reason", "Nenhuma anomalia de volatilidade ativa nos últimos candles. O mercado está operando em equilíbrio, aguardando novos picos de Z-Score para projeção de padrões.")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # 2. Estatísticas históricas e caminhos de preço médios
    st.markdown("#### 📊 Estudo de Eventos Históricos ( Forward Returns )")
    st.write("Estatísticas acumuladas das anomalias passadas mapeadas e seus respectivos comportamentos pós-evento.")

    col_pos, col_neg = st.columns(2)
    max_steps = event_stats["max_steps"]

    with col_pos:
        pos = event_stats["pos_anomaly"]
        st.markdown("##### 🟢 Anomalias de Alta (Z-Score > 0)")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.metric("Eventos Encontrados", pos["count"])
        with p_col2:
            st.metric("Retorno Médio (t+5)", f"{pos['mean_returns'][max_steps-1]:+.2f}%")
        with p_col3:
            st.metric("Taxa de Alta (t+5)", f"{pos['win_rates'][max_steps-1]:.1f}%")

    with col_neg:
        neg = event_stats["neg_anomaly"]
        st.markdown("##### 🔴 Anomalias de Baixa (Z-Score < 0)")
        n_col1, n_col2, n_col3 = st.columns(3)
        with n_col1:
            st.metric("Eventos Encontrados", neg["count"])
        with n_col2:
            st.metric("Retorno Médio (t+5)", f"{neg['mean_returns'][max_steps-1]:+.2f}%")
        with n_col3:
            st.metric("Taxa de Queda (t+5)", f"{neg['win_rates'][max_steps-1]:.1f}%")

    st.write("")
    
    # 3. Gráfico do caminho médio do preço (Event Study Plot)
    st.markdown("##### 📈 Comportamento Médio do Preço Pós-Evento (1 a 5 períodos)")
    st.write("Evolução percentual média acumulada do preço após a sinalização de uma anomalia (Ponto 0 é o preço no candle da anomalia).")

    x_steps = [f"t+{i}" for i in range(max_steps + 1)]
    x_indices = list(range(max_steps + 1))
    
    # Adicionamos o ponto 0 ao início das séries
    y_pos = [0.0] + pos["mean_returns"]
    y_neg = [0.0] + neg["mean_returns"]

    fig_study = go.Figure()
    
    # Linha zero de referência
    fig_study.add_trace(go.Scatter(
        x=x_indices,
        y=[0.0] * (max_steps + 1),
        mode="lines",
        line=dict(color="rgba(226, 232, 240, 0.25)", width=1, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Caminho após anomalias positivas (Alta)
    fig_study.add_trace(go.Scatter(
        x=x_indices,
        y=y_pos,
        mode="lines+markers",
        name="Após Anomalia de Alta (+Z)",
        line=dict(color="#00ffcc", width=2.5),
        marker=dict(color="#00ffcc", size=8),
        hovertemplate="<b>Período:</b> %{x}<br><b>Retorno Médio:</b> %{y:+.2f}%<extra></extra>"
    ))

    # Caminho após anomalias negativas (Baixa)
    fig_study.add_trace(go.Scatter(
        x=x_indices,
        y=y_neg,
        mode="lines+markers",
        name="Após Anomalia de Baixa (-Z)",
        line=dict(color="#ff0055", width=2.5),
        marker=dict(color="#ff0055", size=8),
        hovertemplate="<b>Período:</b> %{x}<br><b>Retorno Médio:</b> %{y:+.2f}%<extra></extra>"
    ))

    fig_study.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(12, 13, 20, 0.0)",
        plot_bgcolor="rgba(12, 13, 20, 0.3)",
        font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
        height=350,
        margin=dict(t=15, b=10, l=10, r=10),
        xaxis=dict(
            title="Períodos Pós-Evento (Candles futuros)",
            tickmode="array",
            tickvals=x_indices,
            ticktext=x_steps
        ),
        yaxis=dict(
            title="Retorno Médio Acumulado (%)",
            tickformat="+.2f%"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        )
    )
    
    fig_study.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
    fig_study.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
    
    st.plotly_chart(fig_study, use_container_width=True)


def render_interdisciplinary_tab(df: pd.DataFrame, ising_result: dict, sir_result: dict, lv_result: dict) -> None:
    """
    Renderiza a aba de Modelos Interdisciplinares: Ising (Sentimento de Mercado),
    SIR (Propagação de Pânico) e Lotka-Volterra (Dinâmica Predador-Presa).
    """
    if df is None or df.empty:
        st.warning("Dados insuficientes para renderizar os modelos interdisciplinares.")
        return

    # ===================== ISING MODEL =====================
    if ising_result is not None:
        st.markdown("#### 🧲 Modelo de Ising — Sentimento Coletivo do Mercado")
        st.write("Simulação de Física Estatística onde agentes (spins) compram (+1) ou vendem (-1). A magnetização média revela o consenso do mercado.")
        
        # KPIs Ising
        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        
        sentiment = ising_result.get("sentiment", "INDECISO")
        mag = ising_result.get("magnetization", 0)
        sent_color = "#00ffcc" if sentiment == "BULLISH" else ("#ff0055" if sentiment == "BEARISH" else "#ffff00")
        sent_shadow = f"0 0 15px {sent_color}"
        
        with i_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {sent_color}; box-shadow: inset 0 0 10px {sent_color}10;">
                <div class="kpi-title">Sentimento do Mercado</div>
                <div class="kpi-value" style="color: {sent_color}; text-shadow: {sent_shadow}; font-weight: 700;">{sentiment}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Magnetização: {mag:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with i_col2:
            phase = ising_result.get("phase", "N/A")
            phase_color = "#00ffcc" if phase == "ORDERED" else "#ff0055"
            phase_label = "ORDENADO" if phase == "ORDERED" else "DESORDENADO"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {phase_color};">
                <div class="kpi-title">Fase do Sistema</div>
                <div class="kpi-value" style="color: {phase_color};">{phase_label}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">T={ising_result.get('temperature', 0):.2f} | Tc={ising_result.get('critical_temp', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with i_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e2e8f0;">
                <div class="kpi-title">Acoplamento (J)</div>
                <div class="kpi-value" style="color: #ffffff;">{ising_result.get('coupling_J', 0):.4f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Força do comportamento de manada</div>
            </div>
            """, unsafe_allow_html=True)
        with i_col4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e2e8f0;">
                <div class="kpi-title">Campo Externo (h)</div>
                <div class="kpi-value" style="color: #ffffff;">{ising_result.get('field_h', 0):+.4f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Tendência direcional</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Gráficos Ising: Heatmap do Grid + Magnetização
        ising_c1, ising_c2 = st.columns(2)
        
        with ising_c1:
            st.markdown("##### Grade de Spins (Compradores vs Vendedores)")
            grid = ising_result.get("grid_final", [])
            if grid:
                fig_grid = go.Figure(data=go.Heatmap(
                    z=grid,
                    colorscale=[[0, "#ff0055"], [0.5, "#1a1a2e"], [1, "#00ffcc"]],
                    showscale=False,
                    hovertemplate="Spin: %{z}<extra></extra>"
                ))
                fig_grid.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(12, 13, 20, 0.0)",
                    plot_bgcolor="rgba(12, 13, 20, 0.3)",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300,
                    xaxis=dict(visible=False), yaxis=dict(visible=False)
                )
                st.plotly_chart(fig_grid, use_container_width=True)
        
        with ising_c2:
            st.markdown("##### Evolução da Magnetização (MCMC)")
            mag_hist = ising_result.get("magnetization_history", [])
            if mag_hist:
                fig_mag = go.Figure()
                fig_mag.add_trace(go.Scatter(
                    x=list(range(len(mag_hist))), y=mag_hist,
                    name="Magnetização",
                    line=dict(color="#00ffcc", width=2),
                    fill="tozeroy", fillcolor="rgba(0, 255, 204, 0.05)"
                ))
                fig_mag.add_trace(go.Scatter(
                    x=list(range(len(mag_hist))), y=[0]*len(mag_hist),
                    mode="lines", line=dict(color="rgba(255,255,255,0.2)", dash="dash"),
                    showlegend=False, hoverinfo="skip"
                ))
                fig_mag.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(12, 13, 20, 0.0)",
                    plot_bgcolor="rgba(12, 13, 20, 0.3)",
                    font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300,
                    xaxis=dict(title="Passos MCMC"),
                    yaxis=dict(title="Magnetização M", range=[-1.1, 1.1])
                )
                fig_mag.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                fig_mag.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                st.plotly_chart(fig_mag, use_container_width=True)
    
    st.markdown("---")
    
    # ===================== SIR MODEL =====================
    if sir_result is not None:
        st.markdown("#### 🦠 Modelo SIR — Propagação de Pânico/Euforia no Mercado")
        st.write("Adaptação epidemiológica: S (Neutros), I (Em Pânico/Euforia), R (Recuperados). Projeta a propagação de sentimento extremo.")
        
        sir_col1, sir_col2, sir_col3, sir_col4 = st.columns(4)
        
        r0 = sir_result.get("R0", 0)
        r0_color = "#ff0055" if r0 > 1 else "#00ffcc"
        current_phase = sir_result.get("current_phase", "ESTÁVEL")
        phase_colors = {"INÍCIO": "#ffff00", "ACELERAÇÃO": "#ff9100", "PICO": "#ff0055", "DECLÍNIO": "#e040fb", "ESTÁVEL": "#00ffcc"}
        ph_color = phase_colors.get(current_phase, "#8a8d9a")
        
        with sir_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {r0_color};">
                <div class="kpi-title">Número Reprodutivo (R₀)</div>
                <div class="kpi-value" style="color: {r0_color}; text-shadow: 0 0 10px {r0_color};">{r0:.2f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">{"Epidemia Ativa" if r0 > 1 else "Sob Controle"}</div>
            </div>
            """, unsafe_allow_html=True)
        with sir_col2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {ph_color};">
                <div class="kpi-title">Fase Atual</div>
                <div class="kpi-value" style="color: {ph_color}; font-weight: 700;">{current_phase}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Ciclo epidemiológico</div>
            </div>
            """, unsafe_allow_html=True)
        with sir_col3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #ff0055;">
                <div class="kpi-title">Pico de Infecção</div>
                <div class="kpi-value" style="color: #ffffff;">{sir_result.get('peak_infection', 0)*100:.1f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">No passo {sir_result.get('peak_step', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with sir_col4:
            hit = sir_result.get("herd_immunity_threshold", 0)
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e2e8f0;">
                <div class="kpi-title">Limiar Imunidade</div>
                <div class="kpi-value" style="color: #ffffff;">{hit*100:.1f}%</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">1 - 1/R₀</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Gráfico SIR
        sir_steps = sir_result.get("steps", [])
        sir_S = sir_result.get("S", [])
        sir_I = sir_result.get("I", [])
        sir_R = sir_result.get("R", [])
        
        if sir_steps and sir_S:
            fig_sir = go.Figure()
            fig_sir.add_trace(go.Scatter(
                x=sir_steps, y=[s*100 for s in sir_S],
                name="Suscetíveis (Neutros)",
                line=dict(color="#00ffcc", width=2),
                fill="tozeroy", fillcolor="rgba(0, 255, 204, 0.03)"
            ))
            fig_sir.add_trace(go.Scatter(
                x=sir_steps, y=[i*100 for i in sir_I],
                name="Infectados (Pânico/Euforia)",
                line=dict(color="#ff0055", width=2.5),
                fill="tozeroy", fillcolor="rgba(255, 0, 85, 0.05)"
            ))
            fig_sir.add_trace(go.Scatter(
                x=sir_steps, y=[r*100 for r in sir_R],
                name="Recuperados",
                line=dict(color="#8a8d9a", width=2),
                fill="tozeroy", fillcolor="rgba(138, 141, 154, 0.03)"
            ))
            
            fig_sir.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(12, 13, 20, 0.0)",
                plot_bgcolor="rgba(12, 13, 20, 0.3)",
                font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                hovermode="x unified",
                margin=dict(t=15, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                height=350,
                xaxis=dict(title="Períodos de Projeção"),
                yaxis=dict(title="Proporção da População (%)", ticksuffix="%")
            )
            fig_sir.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
            fig_sir.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
            st.plotly_chart(fig_sir, use_container_width=True)
    
    st.markdown("---")
    
    # ===================== LOTKA-VOLTERRA =====================
    if lv_result is not None:
        st.markdown("#### 🐺 Lotka-Volterra — Dinâmica Predador-Presa (Compradores vs Vendedores)")
        st.write("Modelo ecológico onde compradores (presas) e vendedores (predadores) oscilam em ciclos naturais de oferta e demanda.")
        
        lv_col1, lv_col2, lv_col3 = st.columns(3)
        
        dominance = lv_result.get("current_dominance", "N/A")
        dom_color = "#00ffcc" if dominance == "COMPRADORES" else "#ff0055"
        
        with lv_col1:
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid {dom_color};">
                <div class="kpi-title">Dominância Atual</div>
                <div class="kpi-value" style="color: {dom_color}; text-shadow: 0 0 10px {dom_color}; font-weight: 700;">{dominance}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Força dominante no mercado</div>
            </div>
            """, unsafe_allow_html=True)
        with lv_col2:
            cycle = lv_result.get("cycle_period", None)
            cycle_str = f"{cycle:.1f}" if cycle else "N/A"
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #ffff00;">
                <div class="kpi-title">Período do Ciclo</div>
                <div class="kpi-value" style="color: #ffff00;">{cycle_str}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Passos por oscilação</div>
            </div>
            """, unsafe_allow_html=True)
        with lv_col3:
            eq_b = lv_result.get("equilibrium_buyers", 0)
            eq_s = lv_result.get("equilibrium_sellers", 0)
            st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #e2e8f0;">
                <div class="kpi-title">Equilíbrio</div>
                <div class="kpi-value" style="color: #ffffff; font-size: 1.2rem;">B={eq_b:.2f} | V={eq_s:.2f}</div>
                <div class="kpi-subtitle" style="color: #8a8d9a;">Ponto fixo do sistema</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Gráficos LV: Dinâmica temporal e Retrato de Fase
        lv_c1, lv_c2 = st.columns(2)
        
        with lv_c1:
            st.markdown("##### Dinâmica Temporal")
            lv_steps = lv_result.get("steps", [])
            buyers = lv_result.get("buyers", [])
            sellers = lv_result.get("sellers", [])
            
            if lv_steps and buyers:
                fig_lv = go.Figure()
                fig_lv.add_trace(go.Scatter(
                    x=lv_steps, y=buyers,
                    name="Compradores (Presa)",
                    line=dict(color="#00ffcc", width=2.5)
                ))
                fig_lv.add_trace(go.Scatter(
                    x=lv_steps, y=sellers,
                    name="Vendedores (Predador)",
                    line=dict(color="#ff0055", width=2.5)
                ))
                fig_lv.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(12, 13, 20, 0.0)",
                    plot_bgcolor="rgba(12, 13, 20, 0.3)",
                    font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                    hovermode="x unified",
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
                    height=300,
                    xaxis=dict(title="Passos de Projeção"),
                    yaxis=dict(title="População")
                )
                fig_lv.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                fig_lv.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                st.plotly_chart(fig_lv, use_container_width=True)
        
        with lv_c2:
            st.markdown("##### Retrato de Fase (Espaço de Estados)")
            pp_x = lv_result.get("phase_portrait_x", [])
            pp_y = lv_result.get("phase_portrait_y", [])
            
            if pp_x and pp_y:
                fig_phase = go.Figure()
                fig_phase.add_trace(go.Scatter(
                    x=pp_x, y=pp_y,
                    mode="lines",
                    name="Trajetória",
                    line=dict(color="#e040fb", width=2)
                ))
                # Ponto de equilíbrio
                fig_phase.add_trace(go.Scatter(
                    x=[eq_b], y=[eq_s],
                    mode="markers",
                    name="Equilíbrio",
                    marker=dict(color="#ffff00", size=12, symbol="star")
                ))
                fig_phase.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(12, 13, 20, 0.0)",
                    plot_bgcolor="rgba(12, 13, 20, 0.3)",
                    font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300,
                    xaxis=dict(title="Compradores"),
                    yaxis=dict(title="Vendedores")
                )
                fig_phase.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                fig_phase.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)")
                st.plotly_chart(fig_phase, use_container_width=True)


def render_decision_tab(df: pd.DataFrame, pomdp_result: dict) -> None:
    """
    Renderiza a aba de Inteligência de Decisão via POMDP.
    Exibe o estado de crença (belief state), recomendação de ação e evolução temporal.
    """
    if df is None or df.empty or pomdp_result is None:
        st.warning("Dados insuficientes para renderizar o modelo de decisão POMDP.")
        return
    
    # KPIs Principais
    action = pomdp_result.get("recommended_action", "MANTER")
    confidence = pomdp_result.get("action_confidence", 0)
    dominant = pomdp_result.get("dominant_state", "NEUTRAL")
    current_belief = pomdp_result.get("current_belief", {})
    
    action_colors = {"COMPRAR": "#00ffcc", "VENDER": "#ff0055", "MANTER": "#ffff00"}
    state_colors = {"BULL": "#00ffcc", "BEAR": "#ff0055", "NEUTRAL": "#ffff00"}
    action_color = action_colors.get(action, "#8a8d9a")
    state_color = state_colors.get(dominant, "#8a8d9a")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {action_color}; box-shadow: inset 0 0 15px {action_color}10;">
            <div class="kpi-title">Ação Recomendada</div>
            <div class="kpi-value" style="color: {action_color}; text-shadow: 0 0 15px {action_color}; font-weight: 700; font-size: 2.2rem;">{action}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Confiança: {confidence*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        state_label = {"BULL": "ALTA (BULL)", "BEAR": "BAIXA (BEAR)", "NEUTRAL": "NEUTRO"}.get(dominant, dominant)
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {state_color};">
            <div class="kpi-title">Estado Dominante do Mercado</div>
            <div class="kpi-value" style="color: {state_color}; text-shadow: 0 0 10px {state_color};">{state_label}</div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">Inferido via observações parciais</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        bull_b = current_belief.get("BULL", 0)
        bear_b = current_belief.get("BEAR", 0)
        neut_b = current_belief.get("NEUTRAL", 0)
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #e2e8f0;">
            <div class="kpi-title">Vetor de Crença Atual</div>
            <div class="kpi-value" style="color: #ffffff; font-size: 1.2rem;">
                <span style="color: #00ffcc;">B:{bull_b*100:.0f}%</span> | 
                <span style="color: #ff0055;">b:{bear_b*100:.0f}%</span> | 
                <span style="color: #ffff00;">N:{neut_b*100:.0f}%</span>
            </div>
            <div class="kpi-subtitle" style="color: #8a8d9a;">P(Bull) | P(Bear) | P(Neutral)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    # Recompensas Estimadas
    rewards = pomdp_result.get("reward_estimates", {})
    if rewards:
        st.markdown("#### 💰 Valor Esperado por Ação")
        r_col1, r_col2, r_col3 = st.columns(3)
        for col_obj, (act, val) in zip([r_col1, r_col2, r_col3], [("COMPRAR", rewards.get("COMPRAR", 0)), ("VENDER", rewards.get("VENDER", 0)), ("MANTER", rewards.get("MANTER", 0))]):
            a_color = action_colors.get(act, "#8a8d9a")
            val_sign = "+" if val >= 0 else ""
            with col_obj:
                st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid {a_color};">
                    <div class="kpi-title">{act}</div>
                    <div class="kpi-value" style="color: {a_color};">{val_sign}{val:.2f}</div>
                    <div class="kpi-subtitle" style="color: #8a8d9a;">Recompensa esperada</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.write("")
    
    # Gráfico de Evolução do Belief State
    belief_history = pomdp_result.get("belief_history", [])
    timestamps = pomdp_result.get("timestamps", [])
    
    if belief_history and timestamps:
        st.markdown("#### 📈 Evolução do Estado de Crença (Belief State)")
        st.write("Distribuição de probabilidade sobre os 3 estados ocultos ao longo do tempo, atualizada a cada observação via algoritmo forward.")
        
        show_n = min(60, len(belief_history))
        ts_slice = timestamps[-show_n:]
        bh_slice = belief_history[-show_n:]
        
        bull_vals = [b.get("BULL", 0)*100 for b in bh_slice]
        bear_vals = [b.get("BEAR", 0)*100 for b in bh_slice]
        neut_vals = [b.get("NEUTRAL", 0)*100 for b in bh_slice]
        
        fig_belief = go.Figure()
        fig_belief.add_trace(go.Scatter(
            x=ts_slice, y=bull_vals,
            name="P(BULL)",
            line=dict(color="#00ffcc", width=2),
            stackgroup="one",
            fillcolor="rgba(0, 255, 204, 0.15)"
        ))
        fig_belief.add_trace(go.Scatter(
            x=ts_slice, y=neut_vals,
            name="P(NEUTRAL)",
            line=dict(color="#ffff00", width=2),
            stackgroup="one",
            fillcolor="rgba(255, 255, 0, 0.10)"
        ))
        fig_belief.add_trace(go.Scatter(
            x=ts_slice, y=bear_vals,
            name="P(BEAR)",
            line=dict(color="#ff0055", width=2),
            stackgroup="one",
            fillcolor="rgba(255, 0, 85, 0.15)"
        ))
        
        fig_belief.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            hovermode="x unified",
            margin=dict(t=15, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            height=400,
            yaxis=dict(title="Probabilidade (%)", range=[0, 100], ticksuffix="%")
        )
        fig_belief.update_xaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        fig_belief.update_yaxes(showgrid=True, gridcolor="rgba(31, 34, 53, 0.4)", linecolor="rgba(0, 255, 204, 0.2)")
        st.plotly_chart(fig_belief, use_container_width=True)
    
    # Matriz de Transição
    transition = pomdp_result.get("transition_matrix", {})
    if transition:
        st.write("")
        st.markdown("#### 🔄 Matriz de Transição de Estados (Calibrada)")
        st.write("Probabilidades de transição entre estados ocultos, estimadas empiricamente a partir dos dados históricos.")
        
        states = ["BULL", "BEAR", "NEUTRAL"]
        z_matrix = []
        text_matrix = []
        for s_from in states:
            row = []
            text_row = []
            for s_to in states:
                key = f"{s_from}_to_{s_to}"
                val = transition.get(key, 0)
                row.append(val * 100)
                text_row.append(f"{val*100:.1f}%")
            z_matrix.append(row)
            text_matrix.append(text_row)
        
        fig_trans = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=["→ BULL", "→ BEAR", "→ NEUTRO"],
            y=["BULL →", "BEAR →", "NEUTRO →"],
            colorscale=[[0, "rgba(12, 13, 20, 0.95)"], [0.5, "rgba(0, 255, 204, 0.4)"], [1, "rgba(255, 0, 85, 0.75)"]],
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(family="Rajdhani, sans-serif", size=14, color="#ffffff"),
            showscale=False
        ))
        fig_trans.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 13, 20, 0.0)",
            plot_bgcolor="rgba(12, 13, 20, 0.3)",
            font=dict(family="Rajdhani, sans-serif", size=13, color="#8a8d9a"),
            height=280,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_trans, use_container_width=True)

