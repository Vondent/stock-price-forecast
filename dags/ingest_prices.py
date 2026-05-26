"""
DAG 1 — ingest_stock_prices
Runs weekdays at 9 PM UTC (~5 PM ET, after market close).
Fetches the last 7 days of OHLCV data, validates, and upserts to SQLite.

Tasks:
    fetch  →  validate  →  store  →  verify
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from airflow.decorators import dag, task

default_args = {
    "owner":          "airflow",
    "retries":        2,
    "retry_delay":    timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="ingest_stock_prices",
    description="Daily OHLCV ingestion for TSLA, BAC, AAPL, F, GE via yfinance",
    schedule="0 21 * * 1-5",   # weekdays at 9 PM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "stocks", "yfinance"],
)
def ingest_stock_prices():

    @task
    def fetch() -> str:
        import tempfile
        from pipeline.fetch import fetch_prices

        df   = fetch_prices(days_back=7)
        path = os.path.join(tempfile.gettempdir(), "prices_raw.parquet")
        df.to_parquet(path, index=False)
        print(f"fetch: {len(df)} rows → {path}")
        return path

    @task
    def validate(raw_path: str) -> str:
        import tempfile
        import pandas as pd
        from pipeline.validate import validate_prices

        df        = pd.read_parquet(raw_path)
        validated = validate_prices(df)
        path      = os.path.join(tempfile.gettempdir(), "prices_validated.parquet")
        validated.to_parquet(path, index=False)
        return path

    @task
    def store(validated_path: str) -> int:
        import pandas as pd
        from pipeline.store import store_prices

        df = pd.read_parquet(validated_path)
        return store_prices(df)

    @task
    def verify(rows_written: int):
        from pipeline.store import row_count

        total = row_count()
        print(f"verify: wrote {rows_written} rows this run, {total} total in DB")
        if rows_written == 0:
            raise ValueError("No rows were written — check yfinance response")

    raw       = fetch()
    validated = validate(raw)
    written   = store(validated)
    verify(written)


dag = ingest_stock_prices()
