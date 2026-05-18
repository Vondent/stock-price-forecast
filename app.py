import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from prophet import Prophet
from datetime import date, timedelta

st.set_page_config(page_title="Stock Forecast", layout="wide", page_icon="📈")

# ── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.section-header {
    font-size: 1.1rem; font-weight: 700; color: #00d4aa;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.25rem; padding-left: 0.75rem;
    border-left: 3px solid #00d4aa;
}
.kpi-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 1rem 1.25rem; text-align: center;
}
.kpi-label { font-size: 0.72rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #e6edf3; line-height: 1; }
.kpi-value.positive { color: #00d4aa; }
.kpi-value.negative { color: #f87171; }
.styled-divider { border: none; border-top: 1px solid #1f2937; margin: 1.5rem 0; }
section[data-testid="stSidebar"] { background: #080d19; border-right: 1px solid #1f2937; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(family="sans-serif", color="#9ca3af"),
    xaxis=dict(gridcolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937"),
    margin=dict(l=16, r=16, t=32, b=16),
    hovermode="x unified",
)

COMPARISON_TICKERS = ["TSLA", "AAPL", "MSFT", "AMZN", "NVDA"]
COMPARISON_COLORS  = ["#f2c53f", "#4ade80", "#60a5fa", "#f97316", "#a78bfa"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def kpi(label, value, positive=None):
    cls = "positive" if positive is True else ("negative" if positive is False else "")
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {cls}">{value}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)


# ── Data ─────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_mean"]   = (df["open"] + df["close"] + df["high"] + df["low"]) / 4
    df["daily_return"] = ((df["close"] - df["open"]) / df["open"]) * 100
    return df


@st.cache_resource
def fit_prophet(ticker: str, start: str, end: str):
    df = load_ticker(ticker, start, end)
    ph = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})
    m  = Prophet(yearly_seasonality="auto")
    m.fit(ph)
    future   = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    return m, forecast, ph


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 Stock Forecast")
    st.markdown("---")
    page = st.radio("", ["Overview & EDA", "Prophet Forecast"], label_visibility="collapsed")
    st.markdown("---")

    ticker_input = st.text_input("Ticker symbol", value="TSLA").upper().strip()

    end_default   = date.today()
    start_default = end_default - timedelta(days=5 * 365)
    start_date = st.date_input("Start date", value=start_default)
    end_date   = st.date_input("End date",   value=end_default)

    st.markdown("---")
    st.caption("Data via Yahoo Finance · Refreshed every hour")

start_str = start_date.strftime("%Y-%m-%d")
end_str   = end_date.strftime("%Y-%m-%d")

df = load_ticker(ticker_input, start_str, end_str)

if df.empty:
    st.error(f"No data found for **{ticker_input}**. Check the ticker and try again.")
    st.stop()

split_date = df["date"].iloc[int(len(df) * 0.8)]
train = df[df["date"] <= split_date]
test  = df[df["date"] >  split_date]
accent = "#00d4aa"


# ── Overview & EDA ───────────────────────────────────────────────────────────

if page == "Overview & EDA":

    info = yf.Ticker(ticker_input).fast_info
    company_name = getattr(info, "display_name", ticker_input)

    st.markdown(f"# {company_name}")
    st.markdown(
        f"<span style='color:{accent};font-weight:700;font-size:1rem;'>{ticker_input}</span>"
        f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.9rem;'>"
        f"{start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}</span>",
        unsafe_allow_html=True,
    )
    divider()

    # KPIs
    start_price   = df["close"].iloc[0]
    end_price     = df["close"].iloc[-1]
    growth        = (end_price - start_price) / start_price * 100
    all_time_high = df["high"].max()
    avg_volume    = df["volume"].mean()

    cols = st.columns(5)
    for col, (label, value, pos) in zip(cols, [
        ("Start Price",   f"${start_price:.2f}",       None),
        ("Current Price", f"${end_price:.2f}",         None),
        ("Total Return",  f"{growth:+.1f}%",           growth > 0),
        ("All-Time High", f"${all_time_high:.2f}",     None),
        ("Avg Daily Vol", f"{avg_volume/1e6:.1f}M",    None),
    ]):
        col.markdown(kpi(label, value, pos), unsafe_allow_html=True)

    divider()

    # Price history
    section("Price History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["date"], y=train["high"],  name="High",  line=dict(color="rgba(74,222,128,0.3)",  width=1)))
    fig.add_trace(go.Scatter(x=train["date"], y=train["low"],   name="Low",   line=dict(color="rgba(248,113,113,0.3)", width=1)))
    fig.add_trace(go.Scatter(x=train["date"], y=train["close"], name="Train", line=dict(color=accent, width=2)))
    fig.add_trace(go.Scatter(x=test["date"],  y=test["close"],  name="Test",  line=dict(color="#6b7280", width=2, dash="dot")))
    fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1,
                  xref="x", yref="paper", line=dict(dash="dash", color="#374151"))
    fig.add_annotation(x=split_date, y=1, xref="x", yref="paper",
                       text="Train / Test split", showarrow=False,
                       font=dict(color="#6b7280", size=11), yanchor="bottom")
    fig.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    section("Trading Volume")
    fig_vol = go.Figure(go.Bar(x=df["date"], y=df["volume"], marker_color=accent, opacity=0.7))
    fig_vol.update_layout(**PLOTLY_LAYOUT, height=220, xaxis_title="", yaxis_title="Volume")
    st.plotly_chart(fig_vol, use_container_width=True)

    divider()

    # Comparison: daily returns + growth
    section("Market Comparison")
    st.caption(f"Comparing {ticker_input} against a default basket — edit tickers in the expander below.")

    with st.expander("Customize comparison tickers"):
        compare_input = st.text_input("Tickers (comma-separated)", value=", ".join(COMPARISON_TICKERS))
        compare_tickers = [t.strip().upper() for t in compare_input.split(",") if t.strip()]

    col1, col2 = st.columns([3, 2])

    with col1:
        section("Daily Returns")
        fig_ret = go.Figure()
        colors = COMPARISON_COLORS + ["#e879f9", "#fb923c", "#34d399"]
        for i, t in enumerate(compare_tickers):
            cdf = load_ticker(t, start_str, end_str)
            if cdf.empty:
                continue
            color = colors[i % len(colors)]
            if t == ticker_input:
                color = accent
            fig_ret.add_trace(go.Scatter(
                x=cdf["date"], y=cdf["daily_return"],
                name=t, line=dict(color=color, width=1), opacity=0.85,
            ))
        fig_ret.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="", yaxis_title="Daily Return (%)")
        st.plotly_chart(fig_ret, use_container_width=True)

    with col2:
        section("Total Return")
        growth_rows = []
        for i, t in enumerate(compare_tickers):
            cdf = load_ticker(t, start_str, end_str)
            if cdf.empty:
                continue
            s, e = cdf["close"].iloc[0], cdf["close"].iloc[-1]
            growth_rows.append({"Ticker": t, "Return": (e - s) / s * 100, "color": colors[i % len(colors)]})

        gdf = pd.DataFrame(growth_rows).sort_values("Return", ascending=True)
        fig_g = go.Figure(go.Bar(
            x=gdf["Return"], y=gdf["Ticker"], orientation="h",
            marker_color=gdf["color"].tolist(),
            text=gdf["Return"].apply(lambda x: f"{x:+.1f}%"),
            textposition="outside",
        ))
        fig_g.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title="Return (%)", yaxis_title="")
        st.plotly_chart(fig_g, use_container_width=True)

    divider()

    # Candlestick (bonus — looks great, shows real finance knowledge)
    section("Candlestick — Last 90 Days")
    recent = df.tail(90)
    fig_candle = go.Figure(go.Candlestick(
        x=recent["date"],
        open=recent["open"], high=recent["high"],
        low=recent["low"],   close=recent["close"],
        increasing_line_color="#00d4aa", decreasing_line_color="#f87171",
    ))
    fig_candle.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="", yaxis_title="Price (USD)")
    fig_candle.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_candle, use_container_width=True)


# ── Prophet Forecast ─────────────────────────────────────────────────────────

elif page == "Prophet Forecast":

    st.markdown(f"# {ticker_input} — Prophet Forecast")
    st.markdown(
        f"<span style='color:{accent};font-weight:700;'>{ticker_input}</span>"
        f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.9rem;'>"
        f"Additive model · yearly + weekly seasonality · 365-day horizon</span>",
        unsafe_allow_html=True,
    )
    divider()

    with st.spinner(f"Fitting Prophet model for {ticker_input}…"):
        model, forecast, hist = fit_prophet(ticker_input, start_str, end_str)

    future_rows  = forecast[forecast["ds"] > hist["ds"].max()]
    current      = hist["y"].iloc[-1]
    end_forecast = future_rows["yhat"].iloc[-1]
    forecast_chg = (end_forecast - current) / current * 100

    cols = st.columns(4)
    for col, (label, value, pos) in zip(cols, [
        ("Current Price",   f"${current:.2f}",                     None),
        ("Forecast (1yr)",  f"${end_forecast:.2f}",                forecast_chg > 0),
        ("Expected Change", f"{forecast_chg:+.1f}%",               forecast_chg > 0),
        ("Upper Bound",     f"${future_rows['yhat_upper'].max():.2f}", None),
    ]):
        col.markdown(kpi(label, value, pos), unsafe_allow_html=True)

    divider()

    section("Price Forecast")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(
        x=pd.concat([future_rows["ds"], future_rows["ds"][::-1]]),
        y=pd.concat([future_rows["yhat_upper"], future_rows["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(0,212,170,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence band",
    ))
    fig_f.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], name="Actual",
                               line=dict(color=accent, width=2)))
    fig_f.add_trace(go.Scatter(x=future_rows["ds"], y=future_rows["yhat"],
                               name="Forecast", line=dict(color="#00d4aa", width=2, dash="dash")))
    forecast_start = hist["ds"].max()
    fig_f.add_shape(type="line", x0=forecast_start, x1=forecast_start, y0=0, y1=1,
                    xref="x", yref="paper", line=dict(dash="dot", color="#374151"))
    fig_f.add_annotation(x=forecast_start, y=1, xref="x", yref="paper",
                         text="Forecast start", showarrow=False,
                         font=dict(color="#6b7280", size=11), yanchor="bottom")
    fig_f.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig_f, use_container_width=True)

    divider()

    section("Trend & Seasonality Components")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        fig_trend = go.Figure(go.Scatter(
            x=forecast["ds"], y=forecast["trend"],
            line=dict(color=accent, width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        ))
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=260, title="Long-term Trend",
                                xaxis_title="", yaxis_title="Price (USD)")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        yearly = forecast[["ds", "yearly"]].copy()
        yearly["month"] = yearly["ds"].dt.month
        monthly = yearly.groupby("month")["yearly"].mean().reset_index()
        monthly["label"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
        fig_y = go.Figure(go.Bar(
            x=monthly["label"], y=monthly["yearly"],
            marker_color=["#00d4aa" if v >= 0 else "#f87171" for v in monthly["yearly"]],
        ))
        fig_y.update_layout(**PLOTLY_LAYOUT, height=260, title="Yearly Seasonality",
                            xaxis_title="", yaxis_title="Effect (USD)")
        st.plotly_chart(fig_y, use_container_width=True)

    with col3:
        weekly = forecast[["ds", "weekly"]].copy()
        weekly["dow"] = weekly["ds"].dt.dayofweek
        dow_avg = weekly.groupby("dow")["weekly"].mean().reset_index()
        dow_avg["day"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig_w = go.Figure(go.Bar(
            x=dow_avg["day"], y=dow_avg["weekly"],
            marker_color=["#00d4aa" if v >= 0 else "#f87171" for v in dow_avg["weekly"]],
        ))
        fig_w.update_layout(**PLOTLY_LAYOUT, height=260, title="Weekly Seasonality",
                            xaxis_title="", yaxis_title="Effect (USD)")
        st.plotly_chart(fig_w, use_container_width=True)

    divider()

    with st.expander("Raw forecast data (next 30 days)"):
        st.dataframe(
            future_rows[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(30).rename(columns={
                "ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"
            }).style.format({"Forecast": "${:.2f}", "Lower": "${:.2f}", "Upper": "${:.2f}"}),
            use_container_width=True,
        )
