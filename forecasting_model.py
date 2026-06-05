"""Generates future stock price forecasts using Prophet models with news sentiment as a regressor."""

import os
import logging
from datetime import date
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, String, Float, Integer, Date
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker
from prophet import Prophet

from models import Base

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class PriceForecast(Base):
    """Model-predicted price forecast data."""
    __tablename__ = "price_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_value: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound: Mapped[float] = mapped_column(Float, nullable=False)
    upper_bound: Mapped[float] = mapped_column(Float, nullable=False)

def main():
    load_dotenv()

    database_url = os.getenv("DATABASE_URL", "sqlite:///data/stock_data.db")
    print(f"Connecting to database: {database_url}")
    engine = create_engine(database_url)

    PriceForecast.__table__.drop(engine, checkfirst=True)
    PriceForecast.__table__.create(engine, checkfirst=True)

    # Load joined price and sentiment datasets
    print("Loading engineered features and daily sentiment...")
    query = """
    SELECT 
        ef.ticker, 
        ef.date, 
        ef.close, 
        ds.score as sentiment_score
    FROM 
        engineered_features ef
    INNER JOIN 
        daily_sentiment ds 
    ON 
        ef.ticker = ds.ticker AND ef.date = ds.date
    """
    aggregated_df = pd.read_sql(query, engine)

    tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
    forecasts_to_save = []

    for ticker in tickers:
        print(f"\nTraining forecasting model for: {ticker}")
        ticker_df = aggregated_df[aggregated_df['ticker'] == ticker].copy()
        
        if ticker_df.empty:
            print(f"[{ticker}] No aggregated price and sentiment data found.")
            continue

        # Prepare formatting expected by Prophet
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(ticker_df['date']),
            'y': ticker_df['close'],
            'sentiment': ticker_df['sentiment_score']
        })
        df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)

        # Hold out final 30 entries for model validation
        test_size = 30
        if len(df_prophet) <= test_size:
            test_size = max(1, len(df_prophet) // 5)
            print(f"[{ticker}] Warning: Data size ({len(df_prophet)}) is small. Using validation size: {test_size}.")

        split_idx = len(df_prophet) - test_size
        train_df = df_prophet.iloc[:split_idx]
        test_df = df_prophet.iloc[split_idx:]

        if len(train_df) >= 2:
            try:
                eval_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
                eval_model.add_regressor('sentiment')
                eval_model.fit(train_df)

                forecast_test = eval_model.predict(test_df)
                y_true = test_df['y'].values
                y_pred = forecast_test['yhat'].values

                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                print(f"[{ticker}] Validation Metrics -> RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            except Exception as e:
                logger.warning(f"[{ticker}] Failed evaluation validation: {e}")
        else:
            print(f"[{ticker}] Insufficient data for train/test split evaluation.")

        # Train final model on full history to generate future forecasts
        try:
            print(f"[{ticker}] Fitting final model on full dataset...")
            final_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            final_model.add_regressor('sentiment')
            final_model.fit(df_prophet)

            # Forecast next 30 days
            future = final_model.make_future_dataframe(periods=30, freq='D')

            # Populate future sentiment values using rolling mean
            last_7_days_avg = df_prophet['sentiment'].tail(7).mean()
            sentiment_map = dict(zip(df_prophet['ds'], df_prophet['sentiment']))
            future['sentiment'] = future['ds'].map(sentiment_map).fillna(last_7_days_avg)

            forecast = final_model.predict(future)
            future_predictions = forecast.tail(30)

            for _, row in future_predictions.iterrows():
                forecasts_to_save.append({
                    'ticker': ticker,
                    'date': row['ds'].date(),
                    'forecast_value': float(row['yhat']),
                    'lower_bound': float(row['yhat_lower']),
                    'upper_bound': float(row['yhat_upper'])
                })
            print(f"[{ticker}] Generated 30-day future forecast successfully.")
        except Exception as e:
            logger.error(f"[{ticker}] Forecasting failed: {e}")

    # Save forecasts
    if forecasts_to_save:
        print("\nSaving future predictions to database...")
        Session = sessionmaker(bind=engine)
        with Session() as session:
            try:
                db_records = [
                    PriceForecast(
                        ticker=item['ticker'],
                        date=item['date'],
                        forecast_value=item['forecast_value'],
                        lower_bound=item['lower_bound'],
                        upper_bound=item['upper_bound']
                    ) for item in forecasts_to_save
                ]
                session.add_all(db_records)
                session.commit()
                print(f"Successfully saved {len(db_records)} forecast entries.")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save forecasts: {e}")
    else:
        print("\nNo forecast data to save.")

if __name__ == "__main__":
    main()

