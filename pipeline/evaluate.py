import json
import os

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error

from pipeline.config import DEGRADATION_THRESHOLD, MODELS_DIR
from pipeline.train import build_prophet_df


def compute_rmse(model: Prophet, test_df: pd.DataFrame) -> float:
    """Predict on test dates and return RMSE against actual close prices."""
    ph_test  = build_prophet_df(test_df).rename(columns={"y": "_y"})
    forecast = model.predict(ph_test[["ds"]])
    y_true   = test_df["close"].values
    y_pred   = forecast["yhat"].values
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics_path(ticker: str) -> str:
    return os.path.join(MODELS_DIR, f"{ticker}_metrics.json")


def load_metrics(ticker: str) -> dict:
    path = _metrics_path(ticker)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_metrics(metrics: dict, ticker: str) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(_metrics_path(ticker), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics for {ticker}: {metrics}")


def check_degradation(
    new_rmse: float,
    ticker: str,
    threshold: float = DEGRADATION_THRESHOLD,
) -> None:
    """Raise if new RMSE is more than `threshold` worse than the saved baseline."""
    prev = load_metrics(ticker)
    if not prev:
        print(f"{ticker}: no previous metrics — accepting model as baseline")
        return

    old_rmse = prev.get("rmse")
    if old_rmse and new_rmse > old_rmse * (1 + threshold):
        pct = (new_rmse / old_rmse - 1) * 100
        raise ValueError(
            f"{ticker}: RMSE degraded {old_rmse:.4f} → {new_rmse:.4f} "
            f"(+{pct:.1f}%, threshold {threshold * 100:.0f}%) — model NOT saved"
        )
    print(f"{ticker}: RMSE {new_rmse:.4f} within threshold (prev {old_rmse:.4f})")
