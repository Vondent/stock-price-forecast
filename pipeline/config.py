import os

TICKERS = ["TSLA", "BAC", "AAPL", "F", "GE"]

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DB_PATH    = os.path.join(DATA_DIR, "prices.db")

RETRAIN_LOOKBACK_DAYS = 730   # 2 years of history for training
TEST_WINDOW_DAYS      = 30    # last 30 days held out for evaluation
DEGRADATION_THRESHOLD = 0.10  # fail if new RMSE > old RMSE * (1 + threshold)
