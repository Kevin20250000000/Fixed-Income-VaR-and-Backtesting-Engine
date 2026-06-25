"""SQL-based data pipeline for trade and market ingestion."""

from sqlalchemy import create_engine, text
import pandas as pd


def create_engine_url(user: str, password: str, host: str, port: int, db: str) -> str:
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def fetch_trade_data(engine) -> pd.DataFrame:
    query = text("""
        SELECT trade_id, symbol, option_type, strike, expiry_date,
               notional, underlying_price, implied_vol, interest_rate,
               dividend_yield, trade_date
        FROM option_trades
        WHERE trade_date >= CURRENT_DATE - INTERVAL '1 year'
    """)
    return pd.read_sql(query, engine)


def fetch_market_data(engine) -> pd.DataFrame:
    query = text("""
        SELECT quote_date, symbol, underlying_price, implied_vol,
               interest_rate, dividend_yield
        FROM market_quotes
        WHERE quote_date = CURRENT_DATE
    """)
    return pd.read_sql(query, engine)
