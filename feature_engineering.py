"""Calculates technical indicators on daily prices and persists features."""

import os
import logging
from datetime import date
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, String, Float, Integer, Date
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from models import Base

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class EngineeredFeature(Base):
    """Engineered stock features dataset."""
    __tablename__ = "engineered_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    returns: Mapped[float] = mapped_column(Float, nullable=True)
    volatility: Mapped[float] = mapped_column(Float, nullable=True)
    rsi: Mapped[float] = mapped_column(Float, nullable=True)
    macd: Mapped[float] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[float] = mapped_column(Float, nullable=True)
    bollinger_upper: Mapped[float] = mapped_column(Float, nullable=True)
    bollinger_lower: Mapped[float] = mapped_column(Float, nullable=True)

def main():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL", "sqlite:///data/stock_data.db")
    print(f"Connecting to database: {database_url}")
    engine = create_engine(database_url)
    
    EngineeredFeature.__table__.drop(engine, checkfirst=True)
    EngineeredFeature.__table__.create(engine, checkfirst=True)

    tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
    print("Loading daily stock prices...")
    raw_df = pd.read_sql("SELECT ticker, date, close FROM stock_prices", engine)
    raw_df = raw_df.drop_duplicates(subset=['ticker', 'date'])
    
    raw_df = raw_df[raw_df['ticker'].isin(tickers)]

    if raw_df.empty:
        print("No stock price records found. Run data_ingestion.py first.")
        return

    all_features = []

    for ticker in tickers:
        print(f"Engineering features for: {ticker}")
        ticker_df = raw_df[raw_df['ticker'] == ticker].copy()
        if ticker_df.empty:
            print(f"[{ticker}] No price data found.")
            continue

        # Sort and handle holidays / weekends by forward filling
        ticker_df['date'] = pd.to_datetime(ticker_df['date'])
        ticker_df = ticker_df.sort_values('date')
        ticker_df = ticker_df.set_index('date')

        all_dates = pd.date_range(start=ticker_df.index.min(), end=ticker_df.index.max(), freq='D')
        ticker_df = ticker_df.reindex(all_dates)
        ticker_df['ticker'] = ticker
        ticker_df = ticker_df.ffill()

        # Intraday Returns
        ticker_df['returns'] = ticker_df['close'].pct_change()

        # Volatility
        ticker_df['volatility'] = ticker_df['returns'].rolling(window=30).std()

        # RSI calculation using exponential weights
        delta = ticker_df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        ticker_df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema_12 = ticker_df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = ticker_df['close'].ewm(span=26, adjust=False).mean()
        ticker_df['macd'] = ema_12 - ema_26
        ticker_df['macd_signal'] = ticker_df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands (20-day)
        sma_20 = ticker_df['close'].rolling(window=20).mean()
        std_20 = ticker_df['close'].rolling(window=20).std()
        ticker_df['bollinger_upper'] = sma_20 + 2 * std_20
        ticker_df['bollinger_lower'] = sma_20 - 2 * std_20

        ticker_df = ticker_df.reset_index().rename(columns={'index': 'date'})
        all_features.append(ticker_df)

    if not all_features:
        print("No features engineered.")
        return

    Session = sessionmaker(bind=engine)
    with Session() as session:
        try:
            db_records = []
            for df_ticker in all_features:
                for _, row in df_ticker.iterrows():
                    rec_date = row['date'].date() if hasattr(row['date'], 'date') else row['date']
                    
                    def clean(val):
                        return None if pd.isna(val) else float(val)

                    record = EngineeredFeature(
                        ticker=row['ticker'],
                        date=rec_date,
                        close=float(row['close']),
                        returns=clean(row['returns']),
                        volatility=clean(row['volatility']),
                        rsi=clean(row['rsi']),
                        macd=clean(row['macd']),
                        macd_signal=clean(row['macd_signal']),
                        bollinger_upper=clean(row['bollinger_upper']),
                        bollinger_lower=clean(row['bollinger_lower'])
                    )
                    db_records.append(record)

            print(f"Saving {len(db_records)} records to database...")
            session.add_all(db_records)
            session.commit()
            print("Feature engineering successfully completed.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save features: {e}")

if __name__ == "__main__":
    main()

