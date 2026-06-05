"""Dashboard interface for visualizing stock prices, technical indicators, and forecasts."""

import os
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Stock Sentiment Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/stock_data.db")

@st.cache_data(ttl=600)
def fetch_ticker_data(ticker):
    """Fetch analytics datasets from database."""
    engine = create_engine(DATABASE_URL)
    
    query_technicals = """
    SELECT 
        sp.date, sp.open, sp.high, sp.low, sp.close, sp.volume,
        ef.rsi, ef.macd, ef.macd_signal, ef.bollinger_upper, ef.bollinger_lower
    FROM 
        stock_prices sp
    LEFT JOIN 
        engineered_features ef 
    ON 
        sp.ticker = ef.ticker AND sp.date = ef.date
    WHERE 
        sp.ticker = :ticker
    ORDER BY 
        sp.date ASC
    """
    
    query_forecasts = """
    SELECT 
        date, forecast_value, lower_bound, upper_bound 
    FROM 
        price_forecasts
    WHERE 
        ticker = :ticker
    ORDER BY 
        date ASC
    """
    
    query_news = """
    SELECT 
        published_at, source, title, url 
    FROM 
        stock_news
    WHERE 
        ticker = :ticker
    ORDER BY 
        published_at DESC
    """
    
    with engine.connect() as conn:
        tech_df = pd.read_sql(text(query_technicals), conn, params={"ticker": ticker})
        forecast_df = pd.read_sql(text(query_forecasts), conn, params={"ticker": ticker})
        news_df = pd.read_sql(text(query_news), conn, params={"ticker": ticker})
        
    return tech_df, forecast_df, news_df

st.title("📊 Stock Analytics & Sentiment Intelligence")

st.sidebar.header("Navigation")
tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
selected_ticker = st.sidebar.selectbox("Select Ticker:", tickers)

with st.spinner("Loading analytical data..."):
    tech_df, forecast_df, news_df = fetch_ticker_data(selected_ticker)

tab_technicals, tab_forecasts, tab_news = st.tabs([
    "📈 Market Technicals", 
    "🔮 Sentiment & Forecast Trends", 
    "📰 Live News Feed"
])

with tab_technicals:
    st.subheader(f"{selected_ticker} Technical Indicators")
    
    if not tech_df.empty:
        tech_df['date'] = pd.to_datetime(tech_df['date'])
        
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        fig.add_trace(
            go.Candlestick(
                x=tech_df['date'],
                open=tech_df['open'],
                high=tech_df['high'],
                low=tech_df['low'],
                close=tech_df['close'],
                name="OHLC Price"
            ),
            row=1, col=1
        )
        
        if 'bollinger_upper' in tech_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_df['date'], 
                    y=tech_df['bollinger_upper'], 
                    name="Bollinger Upper",
                    line=dict(color="rgba(255, 165, 0, 0.6)", width=1.5, dash="dash")
                ),
                row=1, col=1
            )
            
        if 'bollinger_lower' in tech_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_df['date'], 
                    y=tech_df['bollinger_lower'], 
                    name="Bollinger Lower",
                    line=dict(color="rgba(255, 165, 0, 0.6)", width=1.5, dash="dash")
                ),
                row=1, col=1
            )
            
        if 'rsi' in tech_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_df['date'], 
                    y=tech_df['rsi'], 
                    name="RSI (14)", 
                    line=dict(color="purple", width=1.5)
                ),
                row=2, col=1
            )
            fig.add_shape(type="line", x0=tech_df['date'].min(), y0=70, x1=tech_df['date'].max(), y1=70, line=dict(color="red", width=1, dash="dot"), row=2, col=1)
            fig.add_shape(type="line", x0=tech_df['date'].min(), y0=30, x1=tech_df['date'].max(), y1=30, line=dict(color="green", width=1, dash="dot"), row=2, col=1)
            
        if 'macd' in tech_df.columns and 'macd_signal' in tech_df.columns:
            macd_hist = tech_df['macd'] - tech_df['macd_signal']
            colors = ['green' if val >= 0 else 'red' for val in macd_hist]
            fig.add_trace(
                go.Bar(
                    x=tech_df['date'], 
                    y=macd_hist, 
                    name="MACD Hist", 
                    marker_color=colors
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=tech_df['date'], 
                    y=tech_df['macd'], 
                    name="MACD Line", 
                    line=dict(color="blue", width=1.5)
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=tech_df['date'], 
                    y=tech_df['macd_signal'], 
                    name="Signal Line", 
                    line=dict(color="orange", width=1.5)
                ),
                row=3, col=1
            )

        fig.update_layout(
            height=700,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            title_text=f"{selected_ticker} Technical Analysis Chart",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No technical price details found for this ticker.")

with tab_forecasts:
    st.subheader(f"{selected_ticker} Prophet 30-Day Forecast")
    
    if not forecast_df.empty and not tech_df.empty:
        tech_df['date'] = pd.to_datetime(tech_df['date'])
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        fig_forecast = go.Figure()
        
        fig_forecast.add_trace(
            go.Scatter(
                x=tech_df['date'], 
                y=tech_df['close'], 
                mode="lines",
                name="Historical Close", 
                line=dict(color="#1f77b4", width=2)
            )
        )
        
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df['date'], 
                y=forecast_df['forecast_value'], 
                mode="lines",
                name="Forecasted Value", 
                line=dict(color="#2ca02c", width=2, dash="dash")
            )
        )
        
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df['date'], 
                y=forecast_df['lower_bound'], 
                mode="lines",
                line=dict(color="rgba(44, 160, 44, 0)"),
                showlegend=False
            )
        )
        
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_df['date'], 
                y=forecast_df['upper_bound'], 
                mode="lines",
                fill="tonexty", 
                fillcolor="rgba(44, 160, 44, 0.2)",
                line=dict(color="rgba(44, 160, 44, 0)"),
                name="Confidence Interval"
            )
        )
        
        fig_forecast.update_layout(
            height=500,
            template="plotly_dark",
            title_text=f"{selected_ticker} Price Predictive Forecast",
            xaxis_title="Date",
            yaxis_title="Stock Price ($)"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("No forecasting predictions found.")

with tab_news:
    st.subheader(f"{selected_ticker} News Sentiment Feed")
    
    if not news_df.empty:
        analyzer = SentimentIntensityAnalyzer()
        
        for _, row in news_df.iterrows():
            title = row['title']
            published_at = row['published_at']
            source = row['source']
            url = row['url']
            
            score = analyzer.polarity_scores(title)['compound']
            
            if score > 0.15:
                badge_html = '<span style="background-color: #2e7d32; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; margin-right: 10px;">BULLISH</span>'
                card_border_color = '#2e7d32'
            elif score < -0.15:
                badge_html = '<span style="background-color: #c62828; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; margin-right: 10px;">BEARISH</span>'
                card_border_color = '#c62828'
            else:
                badge_html = '<span style="background-color: #424242; color: #ffffff; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; margin-right: 10px;">NEUTRAL</span>'
                card_border_color = '#424242'
                
            st.markdown(
                f"""
                <div style="padding: 15px; border-left: 5px solid {card_border_color}; background-color: #1e1e1e; border-radius: 0px 8px 8px 0px; margin-bottom: 12px; line-height: 1.5;">
                    <div style="display: flex; align-items: center; margin-bottom: 6px;">
                        {badge_html}
                        <span style="color: #888888; font-size: 0.85rem;">{published_at} | <strong>{source}</strong></span>
                    </div>
                    <a href="{url}" target="_blank" style="text-decoration: none; color: #4ba3e3; font-weight: bold; font-size: 1.1rem; hover: text-decoration: underline;">
                        {title}
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No news headlines found.")

