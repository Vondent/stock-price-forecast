import os
import pickle

import pandas as pd
from prophet import Prophet

from pipeline.config import MODELS_DIR


def build_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape a prices DataFrame into Prophet's required ds/y format."""
    return (
        df[["date", "close"]]
        .rename(columns={"date": "ds", "close": "y"})
        .reset_index(drop=True)
    )


def train_prophet(df: pd.DataFrame) -> Prophet:
    """Fit a Prophet model on a prices DataFrame."""
    ph_df = build_prophet_df(df)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(ph_df)
    return model


def save_model(model: Prophet, ticker: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{ticker}_prophet.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model → {path}")
    return path


def load_model(ticker: str) -> Prophet | None:
    path = os.path.join(MODELS_DIR, f"{ticker}_prophet.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def forecast(model: Prophet, periods: int = 365) -> pd.DataFrame:
    """Generate a forward forecast for `periods` calendar days."""
    future = model.make_future_dataframe(periods=periods)
    return model.predict(future)
