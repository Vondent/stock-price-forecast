import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from pipeline.config import TICKERS


def fetch_prices(tickers: list[str] = TICKERS, days_back: int = 7) -> pd.DataFrame:
    """Download OHLCV data for all tickers and return a single tidy DataFrame."""
    end   = datetime.today()
    start = end - timedelta(days=days_back)

    frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"WARNING: no data returned for {ticker}")
            continue

        # yfinance returns MultiIndex columns when auto_adjust=True on some versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index.name = "date"
        df = df.reset_index()
        df["ticker"] = ticker
        frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise RuntimeError("fetch_prices returned no data for any ticker")

    return pd.concat(frames, ignore_index=True)
