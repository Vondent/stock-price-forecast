# Stock Price Forecast

Interactive time-series forecasting app for any stock ticker — live data, 365-day Prophet forecast, and multi-stock comparison.

**[Live Demo →](https://vondent-stock-price-forecast.streamlit.app)**

## What It Does

Pulls live price data via yfinance and runs a full forecasting pipeline: exploratory analysis, stationarity testing, model selection, and a 1-year forward forecast with confidence intervals. Supports any ticker symbol with hourly-refreshed data.

## How It Works

1. **EDA** — price history, trading volume, daily returns, and candlestick charts across a user-selected date range
2. **Comparative analysis** — daily returns and total return across a customizable basket of stocks
3. **Forecasting** — Prophet fits an additive model with yearly and weekly seasonality components and projects 365 days ahead with confidence intervals

## Models & Methods

| | |
| --- | --- |
| Forecasting | Prophet, Auto-ARIMA |
| Model selection | Dickey-Fuller stationarity test, ACF/PACF analysis |
| Evaluation | RMSE, R² (baseline + RandomForestRegressor) |
| Validation | TimeSeriesSplit cross-validation |

## Tech Stack

| | |
| --- | --- |
| App | Streamlit |
| Data | yfinance (live, hourly cache) |
| Forecasting | Prophet, pmdarima |
| Visualization | Plotly |
| Language | Python |