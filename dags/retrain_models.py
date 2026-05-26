"""
DAG 2 — retrain_forecast_models
Runs every Sunday at 6 AM UTC.
Retrains a Prophet model per ticker with a 10% RMSE quality gate.
Models that degrade beyond the threshold are rejected and the previous
model is kept in place.

Tasks (one parallel branch per ticker):
    retrain_{TICKER}
        load_data → train → evaluate → quality_gate → save
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.config import TICKERS, RETRAIN_LOOKBACK_DAYS, TEST_WINDOW_DAYS

default_args = {
    "owner":          "airflow",
    "retries":        1,
    "retry_delay":    timedelta(minutes=10),
    "email_on_failure": False,
}


def make_retrain_task(ticker: str):
    @task(task_id=f"retrain_{ticker}")
    def retrain():
        from pipeline.store    import load_prices
        from pipeline.train    import train_prophet, save_model
        from pipeline.evaluate import compute_rmse, check_degradation, save_metrics

        # ── Load ─────────────────────────────────────────────────────────────
        df = load_prices(ticker, days=RETRAIN_LOOKBACK_DAYS)
        if len(df) < TEST_WINDOW_DAYS + 30:
            raise ValueError(
                f"{ticker}: only {len(df)} rows in DB — need at least "
                f"{TEST_WINDOW_DAYS + 30} to train"
            )

        # ── Split ─────────────────────────────────────────────────────────────
        train_df = df.iloc[:-TEST_WINDOW_DAYS]
        test_df  = df.iloc[-TEST_WINDOW_DAYS:]

        # ── Train ─────────────────────────────────────────────────────────────
        print(f"{ticker}: training on {len(train_df)} rows")
        model = train_prophet(train_df)

        # ── Evaluate ──────────────────────────────────────────────────────────
        rmse = compute_rmse(model, test_df)
        print(f"{ticker}: test RMSE = {rmse:.4f}")

        # ── Quality gate ─────────────────────────────────────────────────────
        # Raises ValueError if RMSE degraded > 10% vs previous run
        check_degradation(rmse, ticker)

        # ── Save ──────────────────────────────────────────────────────────────
        save_model(model, ticker)
        save_metrics({"rmse": rmse, "train_rows": len(train_df), "test_rows": len(test_df),
                      "trained_at": datetime.utcnow().isoformat()}, ticker)

    return retrain


@dag(
    dag_id="retrain_forecast_models",
    description="Weekly Prophet retraining with 10% RMSE quality gate per ticker",
    schedule="0 6 * * 0",   # every Sunday at 6 AM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ml", "forecasting", "prophet", "stocks"],
)
def retrain_forecast_models():
    # Each ticker retrains in parallel — no cross-dependencies
    for ticker in TICKERS:
        make_retrain_task(ticker)()


dag = retrain_forecast_models()
