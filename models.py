"""Database schema definitions and initialization for the stock dashboard database."""

from datetime import date, datetime
from sqlalchemy import create_engine, String, Float, Integer, DateTime, Date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    """Base model class for declarative mapping."""
    pass

class StockPrice(Base):
    """Daily stock price records."""
    __tablename__ = "stock_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)

class StockNews(Base):
    """News articles associated with specific tickers."""
    __tablename__ = "stock_news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    url: Mapped[str] = mapped_column(String(1000), nullable=False)

def init_db(database_url: str):
    """Initialize database tables and return engine instance."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine

