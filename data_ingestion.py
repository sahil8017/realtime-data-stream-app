"""Data ingestion pipeline to fetch historical stock prices and news articles."""

import os
import logging
from datetime import datetime, timedelta
import yfinance as yf
from newsapi import NewsApiClient
from sqlalchemy.orm import sessionmaker

from models import init_db, StockPrice, StockNews

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    os.makedirs("data", exist_ok=True)

    database_url = os.getenv("DATABASE_URL", "sqlite:///data/stock_data.db")
    print(f"Initializing database: {database_url}")
    engine = init_db(database_url)
    Session = sessionmaker(bind=engine)

    news_api_key = os.getenv("NEWS_API_KEY")
    newsapi_client = None
    if news_api_key:
        try:
            newsapi_client = NewsApiClient(api_key=news_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize NewsAPI client: {e}")
            print("Warning: NewsAPI client initialization failed. News ingestion will be skipped.")
    else:
        print("Warning: NEWS_API_KEY not found. News ingestion will be skipped.")

    tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
    end_date = datetime.now()
    start_date_news = end_date - timedelta(days=30)
    start_date_news_str = start_date_news.strftime('%Y-%m-%d')
    end_date_news_str = end_date.strftime('%Y-%m-%d')

    print(f"Starting data ingestion for: {', '.join(tickers)}")

    for ticker in tickers:
        print(f"\nProcessing: {ticker}")
        price_records = []
        news_records = []

        # Fetch historical prices
        try:
            print(f"[{ticker}] Fetching historical market data...")
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period="1y")
            
            if not df.empty:
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                for index, row in df.iterrows():
                    price_date = index.date() if hasattr(index, "date") else index
                    price_rec = StockPrice(
                        ticker=ticker,
                        date=price_date,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume'])
                    )
                    price_records.append(price_rec)
                print(f"[{ticker}] Retrieved {len(price_records)} daily price data points.")
            else:
                print(f"[{ticker}] No price data returned.")
        except Exception as e:
            logger.error(f"[{ticker}] Failed to fetch market data: {e}")

        # Fetch headlines
        if newsapi_client:
            try:
                print(f"[{ticker}] Fetching news headlines...")
                response = newsapi_client.get_everything(
                    q=ticker,
                    language='en',
                    sort_by='relevancy',
                    page_size=25,
                    from_param=start_date_news_str,
                    to=end_date_news_str
                )
                articles = response.get('articles', [])
                for article in articles:
                    pub_date_str = article.get('publishedAt')
                    if not pub_date_str:
                        continue
                    try:
                        pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                    except Exception:
                        pub_date = datetime.now()

                    news_rec = StockNews(
                        ticker=ticker,
                        title=article.get('title') or 'No Title',
                        published_at=pub_date,
                        source=article.get('source', {}).get('name') or 'Unknown',
                        url=article.get('url') or ''
                    )
                    news_records.append(news_rec)
                print(f"[{ticker}] Retrieved {len(news_records)} news articles.")
            except Exception as e:
                logger.warning(f"[{ticker}] Failed to fetch news from NewsAPI: {e}")

        # Batch commit database records
        if price_records or news_records:
            print(f"[{ticker}] Committing records to database...")
            with Session() as session:
                try:
                    if price_records:
                        session.add_all(price_records)
                    if news_records:
                        session.add_all(news_records)
                    session.commit()
                    print(f"[{ticker}] Ingestion successful.")
                except Exception as db_err:
                    session.rollback()
                    logger.error(f"[{ticker}] Database commit failed: {db_err}")
        else:
            print(f"[{ticker}] No records to commit.")

    print("\nData ingestion process completed.")

if __name__ == "__main__":
    main()

