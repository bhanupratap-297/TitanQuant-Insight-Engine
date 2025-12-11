# core/market_overview.py
import pandas as pd
import yfinance as yf

MARKETS = {
    "NIFTY 50 (India)": "^NSEI",
    "NIFTY 500 (India)": "^CRSLDX",   # NIFTY 500 index :contentReference[oaicite:5]{index=5}
    "S&P 500 (US)": "^GSPC",
    "NASDAQ 100 (US)": "^NDX",
    "Dow Jones (US)": "^DJI",
    "Nikkei 225 (Japan)": "^N225",
    "FTSE 100 (UK)": "^FTSE",
}

def get_index_history(symbol: str, period: str = "6mo") -> pd.DataFrame:
    return yf.download(symbol, period=period)

def get_market_constituents(market: str) -> pd.DataFrame:
    """
    Load from local CSV in /data (you prepare these files once).
    Columns: Symbol, Name, YahooTicker, Sector, MarketCap
    """
    if "NIFTY 50" in market:
        path = "data/nifty50_constituents.csv"
    elif "S&P 500" in market:
        path = "data/sp500_constituents.csv"
    elif "NASDAQ" in market:
        path = "data/nasdaq100_constituents.csv"
    else:
        raise ValueError("Market not configured yet.")

    return pd.read_csv(path)

def compute_gainers_losers(const_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expect const_df to already have today's price and pct_change.
    You can populate them using yfinance in the app.
    """
    const_df = const_df.sort_values("PctChange", ascending=False)
    top_gainers = const_df.head(10)
    top_losers = const_df.tail(10)
    return top_gainers, top_losers
