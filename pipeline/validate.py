import pandas as pd


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Run sanity checks and return a clean DataFrame. Raises on hard failures."""
    if df.empty:
        raise ValueError("Validation failed: DataFrame is empty")

    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Validation failed: missing columns {missing}")

    # Drop rows with null close price
    before = len(df)
    df = df.dropna(subset=["close"])
    dropped = before - len(df)
    if dropped:
        print(f"Validation: dropped {dropped} rows with null close")

    # Close must be positive
    bad_close = df[df["close"] <= 0]
    if not bad_close.empty:
        raise ValueError(
            f"Validation failed: {len(bad_close)} rows with close <= 0\n"
            f"{bad_close[['date', 'ticker', 'close']]}"
        )

    # High must be >= low
    bad_hl = df[df["high"] < df["low"]]
    if not bad_hl.empty:
        print(f"Validation: dropping {len(bad_hl)} rows where high < low")
        df = df[df["high"] >= df["low"]]

    # Warn on missing tickers
    from pipeline.config import TICKERS
    found   = set(df["ticker"].unique())
    missing_tickers = set(TICKERS) - found
    if missing_tickers:
        print(f"Validation WARNING: no data for tickers {missing_tickers}")

    print(f"Validation passed: {len(df)} rows across {df['ticker'].nunique()} tickers")
    return df.reset_index(drop=True)
