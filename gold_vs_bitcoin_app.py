import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Gold vs Bitcoin Tracker",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Dark trading-terminal styling ----------
st.markdown("""
<style>
.stApp { background: #050b12; color: #e6eef8; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { background: #08131f; border-right: 1px solid #15364b; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1800px; }
h1, h2, h3 { color: #f3f7ff; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0b1b2a, #07111b);
    border: 1px solid #16405a; border-radius: 12px; padding: 14px 16px;
}
div[data-testid="stMetricLabel"] { color: #b8cce0; }
div[data-testid="stMetricValue"] { color: #f5f8ff; }
div[data-testid="stMetricDelta"] { font-weight: 600; }
hr { border-color: #17364a; }
.small-note { color: #9db2c8; font-size: 0.85rem; }
.status-amber { border: 1px solid #b77916; background: #211a0b; border-radius: 12px; padding: 16px; }
.status-green { border: 1px solid #087f5b; background: #071f1a; border-radius: 12px; padding: 16px; }
.status-red { border: 1px solid #a62935; background: #260d12; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1>🟡 Gold <span style='color:#8ca6bd'>vs</span> "
    "<span style='color:#ff9f0a'>₿ Bitcoin</span> Tracker</h1>"
    "<p style='color:#9db8d0;margin-top:-12px'>Relative strength · Trend · Momentum · Cycle risk</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Controls")
    period_label = st.selectbox("Chart history", ["1 year", "2 years", "3 years", "All available"], index=3)
    period_map = {"1 year": "1y", "2 years": "2y", "3 years": "3y", "All available": "max"}
    history_period = period_map[period_label]
    st.caption("Data source: Yahoo Finance")
    refresh = st.button("↻ Refresh data", use_container_width=True)
    st.divider()
    st.markdown("### What the indicators mean")
    st.caption("Cycle score: experimental price-only indicator based on distance from the 200-day average and 180-day momentum rank.")
    st.caption("Traffic light: rule-based summary of relative performance, trend and momentum. Not a validated forecast.")

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(period):
    btc = yf.download("BTC-USD", period=period, interval="1d", auto_adjust=True, progress=False)
    gold = yf.download("GC=F", period=period, interval="1d", auto_adjust=True, progress=False)
    def extract_close(df, ticker):
        if df.empty:
            return pd.Series(dtype=float, name=ticker)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        close.name = ticker
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    prices = pd.concat([extract_close(btc, "Bitcoin"), extract_close(gold, "Gold")], axis=1).sort_index()
    return prices.dropna(how="all")

if refresh:
    st.cache_data.clear()

with st.spinner("Loading the latest available daily prices…"):
    prices = load_prices(history_period)

if prices.empty or prices["Bitcoin"].dropna().empty or prices["Gold"].dropna().empty:
    st.error("Price data could not be loaded. Check your internet connection and try Refresh data.")
    st.stop()

prices = prices.dropna(subset=["Bitcoin", "Gold"]).copy()
btc = prices["Bitcoin"]
gold = prices["Gold"]
last_date = prices.index[-1]
btc_price = float(btc.iloc[-1])
gold_price = float(gold.iloc[-1])

def pct_change_days(series, days):
    if len(series) <= days:
        return np.nan
    return float((series.iloc[-1] / series.iloc[-days-1] - 1) * 100)

def fmt_pct(value):
    return "—" if pd.isna(value) else f"{value:+.2f}%"

btc_30, gold_30 = pct_change_days(btc, 30), pct_change_days(gold, 30)
btc_90, gold_90 = pct_change_days(btc, 90), pct_change_days(gold, 90)
btc_180, gold_180 = pct_change_days(btc, 180), pct_change_days(gold, 180)
btc_adv_30 = btc_30 - gold_30 if pd.notna(btc_30) and pd.notna(gold_30) else np.nan
btc_adv_90 = btc_90 - gold_90 if pd.notna(btc_90) and pd.notna(gold_90) else np.nan
btc_adv_180 = btc_180 - gold_180 if pd.notna(btc_180) and pd.notna(gold_180) else np.nan

btc_sma200 = btc.rolling(200, min_periods=200).mean()
trend_pct = float((btc.iloc[-1] / btc_sma200.iloc[-1] - 1) * 100) if pd.notna(btc_sma200.iloc[-1]) else np.nan
btc_mom180 = btc.pct_change(180) * 100
momentum_now = float(btc_mom180.iloc[-1]) if pd.notna(btc_mom180.iloc[-1]) else np.nan
momentum_then = float(btc_mom180.iloc[-31]) if len(btc_mom180) >= 31 and pd.notna(btc_mom180.iloc[-31]) else np.nan

# Transparent, price-only score. Percentile of trailing 3-year momentum plus trend distance.
mom_window = btc_mom180.dropna().tail(756)
mom_percentile = float((mom_window <= momentum_now).mean() * 100) if len(mom_window) and pd.notna(momentum_now) else np.nan
trend_component = float(np.clip(50 + trend_pct * 2.5, 0, 100)) if pd.notna(trend_pct) else np.nan
cycle_score = float(np.clip(0.5 * mom_percentile + 0.5 * trend_component, 0, 100)) if pd.notna(mom_percentile) and pd.notna(trend_component) else np.nan

btc_drawdown = float((btc.iloc[-1] / btc.cummax().iloc[-1] - 1) * 100)
gold_drawdown = float((gold.iloc[-1] / gold.cummax().iloc[-1] - 1) * 100)

advantages = [btc_adv_30, btc_adv_90, btc_adv_180]
btc_wins = sum(pd.notna(v) and v > 0 for v in advantages)
btc_losses = sum(pd.notna(v) and v < 0 for v in advantages)
momentum_state = "Insufficient history"
if pd.notna(momentum_now) and pd.notna(momentum_then):
    if momentum_now > momentum_then:
        momentum_state = "Improving"
    elif momentum_now < momentum_then:
        momentum_state = "Weakening"
    else:
        momentum_state = "Stable"

if btc_wins >= 2 and pd.notna(trend_pct) and trend_pct > 0 and pd.notna(momentum_now) and momentum_now > 0 and momentum_state == "Improving":
    light, light_class = "GREEN", "status-green"
    status_text = "Constructive: Bitcoin leads gold and trend and momentum agree."
elif btc_losses >= 2 and pd.notna(trend_pct) and trend_pct < 0 and pd.notna(momentum_now) and momentum_now < 0 and momentum_state == "Weakening":
    light, light_class = "RED", "status-red"
    status_text = "Defensive: Bitcoin trails gold and trend and momentum are weak."
else:
    light, light_class = "AMBER", "status-amber"
    status_text = "Mixed conditions: the stronger green or red criteria are not met."

# 1-day changes
btc_1d = float((btc.iloc[-1] / btc.iloc[-2] - 1) * 100) if len(btc) > 1 else np.nan
gold_1d = float((gold.iloc[-1] / gold.iloc[-2] - 1) * 100) if len(gold) > 1 else np.nan
updated = last_date.strftime("%d %b %Y")

st.markdown(f"<div class='small-note'>Last available market data: <b>{updated}</b> · Refreshes on demand · Data may be delayed</div>", unsafe_allow_html=True)
st.write("")

m = st.columns(7)
m[0].metric("Bitcoin price (USD)", f"${btc_price:,.2f}", fmt_pct(btc_1d))
m[1].metric("Gold futures (USD/oz)", f"${gold_price:,.2f}", fmt_pct(gold_1d))
m[2].metric("BTC cycle score", f"{cycle_score:.1f}/100" if pd.notna(cycle_score) else "—", "Price-only model")
m[3].metric("BTC vs 200-day avg", fmt_pct(trend_pct), "Above average" if pd.notna(trend_pct) and trend_pct > 0 else "Below average")
m[4].metric("BTC 180-day return", fmt_pct(btc_180), "Positive" if pd.notna(btc_180) and btc_180 > 0 else "Negative")
m[5].metric("BTC drawdown from high", fmt_pct(btc_drawdown))
m[6].metric("Gold drawdown from high", fmt_pct(gold_drawdown))

st.write("")
left, middle, right = st.columns([1.35, 1, 0.82], gap="medium")

chart_bg = "#07111b"
grid_color = "#173044"
font_color = "#dce8f5"
btc_color = "#ff9f0a"
gold_color = "#2f9bff"
purple_color = "#b54cff"

def dark_layout(fig, height=350):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
        font=dict(color=font_color, family="Arial"),
        margin=dict(l=10, r=10, t=25, b=10), height=height,
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
    )
    return fig

with left:
    st.subheader("Price history · indexed to 100")
    indexed = prices / prices.iloc[0] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indexed.index, y=indexed["Bitcoin"], name="Bitcoin", line=dict(color=btc_color, width=2.5)))
    fig.add_trace(go.Scatter(x=indexed.index, y=indexed["Gold"], name="Gold", line=dict(color=gold_color, width=2.5)))
    fig.update_yaxes(title_text="Growth of 100")
    st.plotly_chart(dark_layout(fig, 390), use_container_width=True)

with middle:
    st.subheader("Relative performance")
    perf = pd.DataFrame({
        "Period": ["30 days", "90 days", "180 days"],
        "Bitcoin": [btc_30, btc_90, btc_180],
        "Gold": [gold_30, gold_90, gold_180],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=perf["Period"], y=perf["Bitcoin"], name="Bitcoin", marker_color=btc_color))
    fig.add_trace(go.Bar(x=perf["Period"], y=perf["Gold"], name="Gold", marker_color=gold_color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Return (%)", zeroline=True)
    st.plotly_chart(dark_layout(fig, 390), use_container_width=True)

with right:
    st.subheader("Market status")
    st.markdown(f"<div class='{light_class}'><h2 style='margin:0;color:#ffbd45'>{light}</h2><p>{status_text}</p><hr><p>BTC leads gold: <b>{btc_wins} of 3 periods</b></p><p>BTC vs 200-day avg: <b>{fmt_pct(trend_pct)}</b></p><p>BTC 180-day return: <b>{fmt_pct(btc_180)}</b></p><p>Momentum: <b>{momentum_state}</b></p></div>", unsafe_allow_html=True)
    st.caption("Experimental rule-based summary; not a validated trading signal.")

bottom_left, bottom_right = st.columns(2, gap="medium")
with bottom_left:
    st.subheader("180-day relative strength")
    adv_series = btc_mom180.reindex(prices.index) - (gold.pct_change(180) * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=adv_series.index, y=adv_series, name="BTC advantage over gold", line=dict(color=purple_color, width=2.5), fill="tozeroy"))
    fig.add_hline(y=0, line_dash="dot", line_color="#8ca6bd")
    fig.update_yaxes(title_text="Percentage points")
    st.plotly_chart(dark_layout(fig, 330), use_container_width=True)

with bottom_right:
    st.subheader("Momentum monitor")
    momentum_table = pd.DataFrame({
        "BTC 180-day return (%)": btc_mom180,
        "Gold 180-day return (%)": gold.pct_change(180) * 100,
    }).dropna()
    momentum_table["BTC advantage (pp)"] = momentum_table.iloc[:, 0] - momentum_table.iloc[:, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=momentum_table.index, y=momentum_table["BTC 180-day return (%)"], name="BTC 180-day return", line=dict(color=btc_color, width=2)))
    fig.add_trace(go.Scatter(x=momentum_table.index, y=momentum_table["Gold 180-day return (%)"], name="Gold 180-day return", line=dict(color=gold_color, width=2)))
    fig.add_trace(go.Scatter(x=momentum_table.index, y=momentum_table["BTC advantage (pp)"], name="BTC advantage", line=dict(color=purple_color, width=2)))
    st.plotly_chart(dark_layout(fig, 330), use_container_width=True)
    if len(momentum_table) >= 8:
        change_7 = float(momentum_table["BTC advantage (pp)"].iloc[-1] - momentum_table["BTC advantage (pp)"].iloc[-8])
    else:
        change_7 = np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("BTC 180-day", fmt_pct(btc_180))
    c2.metric("Gold 180-day", fmt_pct(gold_180))
    c3.metric("BTC advantage", f"{btc_adv_180:+.2f} pp" if pd.notna(btc_adv_180) else "—")
    st.metric("Change over 7 observations", f"{change_7:+.2f} pp" if pd.notna(change_7) else "—", momentum_state)

with st.expander("Recent momentum data"):
    display_df = momentum_table.tail(10).copy()
    st.dataframe(display_df.round(2), use_container_width=True)

st.markdown("---")
st.markdown(
    "<p class='small-note'>Data source: Yahoo Finance (BTC-USD and GC=F gold futures). "
    "The cycle score is an experimental price-only indicator, not the original full OCM model. "
    "Gold futures are not the same as UK retail physical gold. This dashboard is for information "
    "and educational purposes only and is not financial advice.</p>",
    unsafe_allow_html=True,
)
