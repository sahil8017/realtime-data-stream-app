"""Calculates news sentiment scores using FinBERT and VADER, aggregating them daily."""

import os
import logging
from datetime import date
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, String, Float, Integer, Date, UniqueConstraint
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from models import Base

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class DailySentiment(Base):
    """Daily sentiment index metrics."""
    __tablename__ = "daily_sentiment"
    __table_args__ = (UniqueConstraint('ticker', 'date', name='_ticker_date_uc'),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)

def main():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL", "sqlite:///data/stock_data.db")
    print(f"Connecting to database: {database_url}")
    engine = create_engine(database_url)

    DailySentiment.__table__.create(engine, checkfirst=True)

    # Load news headlines
    print("Loading news headlines...")
    news_df = pd.read_sql("SELECT ticker, title, published_at FROM stock_news", engine)
    
    tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
    news_df = news_df[news_df['ticker'].isin(tickers)]

    if news_df.empty:
        print("No stock news records found. Run data_ingestion.py first.")
        return

    news_df['date'] = pd.to_datetime(news_df['published_at']).dt.date

    # Run NLP sentiment scoring models
    print("Initializing NLP sentiment models...")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None)
    vader_analyzer = SentimentIntensityAnalyzer()

    headlines = news_df['title'].tolist()
    print(f"Analyzing sentiment for {len(headlines)} headlines...")
    
    finbert_predictions = finbert_pipeline(headlines)

    finbert_scores = []
    vader_scores = []
    composite_scores = []

    for idx, headline in enumerate(headlines):
        # Calculate FinBERT sentiment score
        preds = {item['label'].lower(): item['score'] for item in finbert_predictions[idx]}
        finbert_score = preds.get('positive', 0.0) - preds.get('negative', 0.0)
        finbert_scores.append(finbert_score)

        # Calculate VADER sentiment score
        vader_score = vader_analyzer.polarity_scores(headline)['compound']
        vader_scores.append(vader_score)

        # Composite score
        composite_score = 0.7 * finbert_score + 0.3 * vader_score
        composite_scores.append(composite_score)

    news_df['composite_score'] = composite_scores

    # Group scores by date
    print("Aggregating daily sentiment scores...")
    daily_sentiment = news_df.groupby(['ticker', 'date'])['composite_score'].mean().reset_index()

    # Save to database
    print("Upserting daily aggregated sentiment scores...")
    Session = sessionmaker(bind=engine)
    with Session() as session:
        try:
            for _, row in daily_sentiment.iterrows():
                stmt = insert(DailySentiment).values(
                    ticker=row['ticker'],
                    date=row['date'],
                    score=float(row['composite_score'])
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['ticker', 'date'],
                    set_=dict(score=stmt.excluded.score)
                )
                session.execute(stmt)
            session.commit()
            print(f"Successfully upserted {len(daily_sentiment)} sentiment entries.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to upsert sentiment data: {e}")

if __name__ == "__main__":
    main()

