"""In-memory stock technical analytics and news sentiment parsing for cloud deployment."""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Real-Time Stock Sentiment Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222b;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2e333d;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

def get_mock_news(ticker, error_msg=None):
    """Generate simulated news headline data for ticker fallback."""
    import random
    
    mock_db = {
        "AAPL": [
            ("Apple Unveils iOS 18 with Deeply Integrated Artificial Intelligence Ecosystem", "TechCrunch", "https://techcrunch.com"),
            ("Apple Stock Rebounds as Global iPhone Shipment Data Outperforms Estimates", "Bloomberg", "https://bloomberg.com"),
            ("Antitrust Regulators Eye Apple's App Store Billing Policies in Renewed Probe", "The Wall Street Journal", "https://wsj.com"),
            ("Apple Supplier TSMC Signals Record Orders for Next-Gen 3nm Apple Silicon", "Reuters", "https://reuters.com"),
            ("Apple Explores Advanced Robotics for Smart Home Devices", "Bloomberg", "https://bloomberg.com")
        ],
        "GOOGL": [
            ("Google Announces Major Gemini 1.5 Pro Model Updates with Million Token Context Window", "VentureBeat", "https://venturebeat.com"),
            ("Google Cloud Revenue Rises 28% Driven by Accelerated Enterprise AI Workloads", "CNBC", "https://cnbc.com"),
            ("Justice Department's Google Search Monopoly Antitrust Case Enters Final Arguments", "Reuters", "https://reuters.com"),
            ("Google DeepMind Showcases AlphaFold 3 for Drug Discovery Applications", "Nature", "https://nature.com"),
            ("Google Shares Jump 4% on Generative AI Search Feature Integration", "MarketWatch", "https://marketwatch.com")
        ],
        "TSLA": [
            ("Tesla Reports Q2 Delivery Volume Outperforming Lowered Wall Street Forecasts", "Bloomberg", "https://bloomberg.com"),
            ("Elon Musk Promises Robotaxi Unveiling Event Set for Late Summer", "Reuters", "https://reuters.com"),
            ("Tesla Negotiates New Battery Material Contracts in Bid to Lower Costs", "CNBC", "https://cnbc.com"),
            ("Tesla Stock Rallies as FSD Subscription Rate Drops, Boosting Adoption", "Barrons", "https://barrons.com"),
            ("EV Price Wars Intensify in European and Asian Markets, Pressuring Tesla Margins", "The Wall Street Journal", "https://wsj.com")
        ],
        "MSFT": [
            ("Microsoft Exceeds Quarterly Income Forecasts on 31% Azure Cloud Growth", "CNBC", "https://cnbc.com"),
            ("Microsoft Introduces New Copilot+ PCs Featuring Dedicated AI Hardware", "The Verge", "https://theverge.com"),
            ("Microsoft and OpenAI Expand Supercomputer Partnership to Train Next-Gen Models", "TechCrunch", "https://techcrunch.com"),
            ("European Commission Investigates Microsoft Teams Bundling Practices", "Reuters", "https://reuters.com"),
            ("Microsoft Market Capitalization Nears Record Levels on Robust AI Revenues", "Bloomberg", "https://bloomberg.com")
        ],
        "NVDA": [
            ("Nvidia Blackwell AI Platform Enters Mass Production with Strong Pre-Orders", "Reuters", "https://reuters.com"),
            ("Nvidia Revenue Skyrockets by 260% Year-Over-Year on Insatiable Data Center Demand", "CNBC", "https://cnbc.com"),
            ("Competitors AMD and Intel Unveil Competitor AI Accelerator Architecture Chips", "VentureBeat", "https://venturebeat.com"),
            ("Nvidia Stock Crosses All-Time Highs as Demand Outstrips Fab Capacities", "MarketWatch", "https://marketwatch.com"),
            ("Global Supply Chain Challenges Check Speed of Nvidia Chip Shipments", "The Wall Street Journal", "https://wsj.com")
        ]
    }
    
    ticker_news = mock_db.get(ticker, [
        (f"Market Sentiment Shifts for {ticker} Amid Macroeconomic Adjustments", "Financial Times", "https://ft.com"),
        (f"Analysts Re-evaluate {ticker} Growth Target Forecasts After Recent Quarterly Review", "Bloomberg", "https://bloomberg.com")
    ])
    
    articles = []
    base_time = datetime.now()
    for i, (title, source, url) in enumerate(ticker_news):
        pub_time = base_time - timedelta(hours=i * 6 + random.randint(1, 5))
        articles.append({
            "title": title,
            "published_at": pub_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": source,
            "url": url,
            "is_mock": True
        })
    return articles

@st.cache_data(ttl=300)
def fetch_realtime_data(ticker):
    """Ingest market and news data, calculating technical features in-memory."""
    # Fetch market data
    df = pd.DataFrame()
    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period="5d", interval="5m")
    except Exception:
        df = pd.DataFrame()
        
    if not df.empty:
        df.index.name = 'date'
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['date'])
        
        # Compute technical indicators
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=30).std()
        
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['bollinger_upper'] = sma_20 + (2 * std_20)
        df['bollinger_lower'] = sma_20 - (2 * std_20)
    
    # Retrieve news headlines
    articles_list = []
    
    try:
        api_key = st.secrets["NEWS_API_KEY"]
    except Exception:
        api_key = None
        
    if api_key:
        try:
            newsapi_client = NewsApiClient(api_key=api_key)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            response = newsapi_client.get_everything(
                q=ticker,
                language='en',
                sort_by='relevancy',
                page_size=25,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            raw_articles = response.get('articles', [])
            for art in raw_articles:
                articles_list.append({
                    'title': art.get('title') or 'No Title',
                    'published_at': art.get('publishedAt') or '',
                    'source': art.get('source', {}).get('name') or 'Unknown',
                    'url': art.get('url') or '#'
                })
        except Exception as e:
            articles_list = get_mock_news(ticker, error_msg=str(e))
    else:
        articles_list = get_mock_news(ticker, error_msg="NEWS_API_KEY missing from secrets")
        
    return df, articles_list


# Sidebar Configuration
st.sidebar.title("⚡ Sentiment Engine")
st.sidebar.markdown("---")
tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
selected_ticker = st.sidebar.selectbox("Select Target Ticker:", tickers, index=0)

st.sidebar.markdown("""
### Cloud Deployment Architecture
- **In-Memory Pipeline**: Zero database writes.
- **RAM Optimized**: Using local lexicon-based VADER parser (FinBERT disabled).
- **Security**: Using encrypted Streamlit secrets.
- **API Caching**: 5-minute expiration window.
""")

st.title("⚡ Real-Time Stock Analytics + Sentiment Intelligence Dashboard")
st.markdown(f"Currently viewing real-time intraday metrics and headlines for **{selected_ticker}**.")

with st.spinner("Fetching live feed and computing technicals..."):
    price_df, news_feed = fetch_realtime_data(selected_ticker)

if not price_df.empty:
    latest = price_df.iloc[-1]
    previous = price_df.iloc[-2] if len(price_df) > 1 else latest
    
    price_diff = latest['Close'] - previous['Close']
    price_pct = (price_diff / previous['Close']) * 100 if previous['Close'] != 0 else 0.0
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Latest Price", f"${latest['Close']:.2f}", f"{price_pct:+.2f}%")
    with m_col2:
        st.metric("Intraday Volatility (30p)", f"{latest['volatility']*100:.3f}%" if not pd.isna(latest['volatility']) else "N/A")
    with m_col3:
        st.metric("RSI (14)", f"{latest['rsi']:.1f}" if not pd.isna(latest['rsi']) else "N/A")
    with m_col4:
        st.metric("MACD Hist", f"{latest['macd_histogram']:.4f}" if not pd.isna(latest['macd_histogram']) else "N/A")

    tab_tech, tab_sentiment = st.tabs(["📈 Real-Time Technicals", "📰 Live Sentiment Feed"])

    with tab_tech:
        st.subheader(f"{selected_ticker} Technical Indicators (5-Minute Intervals)")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Candlestick and Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=price_df['date'],
                open=price_df['Open'],
                high=price_df['High'],
                low=price_df['Low'],
                close=price_df['Close'],
                name="OHLC Price"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_df['date'],
                y=price_df['bollinger_upper'],
                name="Bollinger Upper",
                line=dict(color="rgba(255, 165, 0, 0.65)", width=1.5, dash="dash")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_df['date'],
                y=price_df['bollinger_lower'],
                name="Bollinger Lower",
                line=dict(color="rgba(255, 165, 0, 0.65)", width=1.5, dash="dash")
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=price_df['date'],
                y=price_df['rsi'],
                name="RSI (14)",
                line=dict(color="#b0bec5", width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_shape(
            type="line", x0=price_df['date'].min(), y0=70, x1=price_df['date'].max(), y1=70,
            line=dict(color="rgba(239, 83, 80, 0.5)", width=1.5, dash="dot"),
            row=2, col=1
        )
        fig.add_shape(
            type="line", x0=price_df['date'].min(), y0=30, x1=price_df['date'].max(), y1=30,
            line=dict(color="rgba(102, 187, 106, 0.5)", width=1.5, dash="dot"),
            row=2, col=1
        )
        
        # MACD
        colors = ['rgba(102, 187, 106, 0.75)' if val >= 0 else 'rgba(239, 83, 80, 0.75)' for val in price_df['macd_histogram']]
        fig.add_trace(
            go.Bar(
                x=price_df['date'],
                y=price_df['macd_histogram'],
                name="MACD Histogram",
                marker_color=colors
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_df['date'],
                y=price_df['macd'],
                name="MACD Line",
                line=dict(color="#4ba3e3", width=1.2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_df['date'],
                y=price_df['macd_signal'],
                name="Signal Line",
                line=dict(color="#ffb74d", width=1.2)
            ),
            row=3, col=1
        )
        
        # Figure styling
        fig.update_layout(
            height=850,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(
            showspikes=True,
            spikedash='dash',
            spikemode='across',
            spikesnap='cursor',
            gridcolor="rgba(255,255,255,0.05)"
        )
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        
        st.plotly_chart(fig, use_container_width=True)

    with tab_sentiment:
        st.subheader(f"Recent Sentiment Analysis for {selected_ticker}")
        
        if news_feed:
            if any(art.get('is_mock') for art in news_feed):
                st.info("ℹ️ **Offline Fallback:** Displaying simulated news headlines since the NewsAPI key is not configured or rate-limited.")
                
            analyzer = SentimentIntensityAnalyzer()
            
            for article in news_feed:
                title = article['title']
                source = article['source']
                url = article['url']
                
                try:
                    dt = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                    pub_str = dt.strftime("%b %d, %Y | %I:%M %p")
                except Exception:
                    pub_str = article['published_at']
                
                v_score = analyzer.polarity_scores(title)['compound']
                
                if v_score > 0.15:
                    badge = '<span style="background-color: rgba(46, 125, 50, 0.2); color: #81c784; border: 1px solid #2e7d32; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 10px;">BULLISH</span>'
                    border_color = 'rgba(76, 175, 80, 0.6)'
                elif v_score < -0.15:
                    badge = '<span style="background-color: rgba(198, 40, 40, 0.2); color: #e57373; border: 1px solid #c62828; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 10px;">BEARISH</span>'
                    border_color = 'rgba(244, 67, 54, 0.6)'
                else:
                    badge = '<span style="background-color: rgba(66, 66, 66, 0.2); color: #b0bec5; border: 1px solid #424242; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 10px;">NEUTRAL</span>'
                    border_color = 'rgba(158, 158, 158, 0.4)'
                
                st.markdown(
                    f"""
                    <div style="padding: 18px; border-left: 6px solid {border_color}; background-color: #1a1e24; border-radius: 0px 8px 8px 0px; margin-bottom: 15px; border-top: 1px solid #2d3139; border-right: 1px solid #2d3139; border-bottom: 1px solid #2d3139;">
                        <div style="display: flex; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                            {badge}
                            <span style="color: #90a4ae; font-size: 0.8rem;">{pub_str} | Source: <strong>{source}</strong></span>
                            <span style="color: #607d8b; font-size: 0.8rem; margin-left: auto;">Score: {v_score:+.2f}</span>
                        </div>
                        <a href="{url}" target="_blank" style="text-decoration: none; color: #4ba3e3; font-weight: 600; font-size: 1.05rem; transition: color 0.2s;">
                            {title}
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("No news articles found for this ticker.")
else:
    st.error("Unable to load price data. Check ticker configuration or internet connection.")

