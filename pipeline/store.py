import os
import sqlite3
import pandas as pd

from pipeline.config import DATA_DIR, DB_PATH


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date    TEXT    NOT NULL,
            ticker  TEXT    NOT NULL,
            open    REAL,
            high    REAL,
            low     REAL,
            close   REAL    NOT NULL,
            volume  INTEGER,
            PRIMARY KEY (date, ticker)
        )
    """)
    conn.commit()
    conn.close()


def store_prices(df: pd.DataFrame) -> int:
    """Upsert rows into the prices table. Returns number of rows written."""
    init_db()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    # Stage into a temp table then INSERT OR REPLACE to handle duplicates cleanly
    df.to_sql("_staging", conn, if_exists="replace", index=False)
    conn.execute("""
        INSERT OR REPLACE INTO prices (date, ticker, open, high, low, close, volume)
        SELECT date, ticker, open, high, low, close, volume FROM _staging
    """)
    conn.execute("DROP TABLE IF EXISTS _staging")
    conn.commit()
    conn.close()

    print(f"store_prices: upserted {len(df)} rows to {DB_PATH}")
    return len(df)


def load_prices(ticker: str, days: int = 730) -> pd.DataFrame:
    """Load the most recent `days` rows for a single ticker."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM prices WHERE ticker = ? ORDER BY date ASC",
        conn, params=(ticker,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days).reset_index(drop=True)


def row_count(ticker: str | None = None) -> int:
    conn = sqlite3.connect(DB_PATH)
    if ticker:
        n = conn.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()[0]
    else:
        n = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    conn.close()
    return n
