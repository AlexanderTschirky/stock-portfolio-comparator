import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
from urllib.parse import urlencode

# ------------------------------
# App Metadata & Config
# ------------------------------
st.set_page_config(page_title="Stock Comparator & Portfolio", layout="wide")
st.title("📊 Stock & Portfolio Comparator (yfinance)")

st.caption(
    "Built with Streamlit + yfinance. Education-use only. Prices are adjusted for splits/dividends (auto_adjust=True)."
)

# ------------------------------
# Helpers
# ------------------------------

def _annualize(ret_series: pd.Series):
    """Return annualized mean return and vol assuming 252 trading days."""
    mu_d = ret_series.mean()
    sd_d = ret_series.std()
    ann_ret = (1 + mu_d) ** 252 - 1
    ann_vol = sd_d * np.sqrt(252)
    return ann_ret, ann_vol


def perf_stats(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    stats = {}
    for c in prices.columns:
        if c not in rets:  # safety
            continue
        ann_ret, ann_vol = _annualize(rets[c])
        sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan
        stats[c] = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe (rf=0)": sharpe,
            "Max Drawdown": dd[c].min(),
        }
    return pd.DataFrame(stats).T


def norm_100(prices: pd.DataFrame) -> pd.DataFrame:
    return prices / prices.iloc[0] * 100.0


def portfolio_series(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weights = weights.reindex(prices.columns).fillna(0)
    # Normalize to 1.0 if sum > 0, else equal-weight
    if weights.sum() > 0:
        w = weights / weights.sum()
    else:
        w = pd.Series(1 / len(prices.columns), index=prices.columns)
    port_ret = prices.pct_change().fillna(0).dot(w)
    return (1 + port_ret).cumprod() * 100.0


@st.cache_data(ttl=30 * 60)
def load_prices(tickers, start, end, interval="1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    # yfinance returns MultiIndex when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        close = df.loc[:, (slice(None), "Close")]
        close.columns = [c[0] for c in close.columns]
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    # Drop all-NA columns
    close = close.dropna(how="all")
    return close


# ------------------------------
# Sidebar Controls
# ------------------------------
qparams = st.query_params

def_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]
url_tickers = qparams.get("stocks")
if url_tickers:
    # support both comma separated str and list
    tickers_default = (
        url_tickers.split(",") if isinstance(url_tickers, str) else list(url_tickers)
    )
else:
    tickers_default = def_tickers

with st.sidebar:
    st.header("⚙️ Controls")
    tickers = st.multiselect(
        "Tickers",
        options=sorted(
            list(
                set(
                    def_tickers
                    + [
                        "AMZN",
                        "META",
                        "TSLA",
                        "NFLX",
                        "AMD",
                        "INTC",
                        "IBM",
                        "ORCL",
                        "CRM",
                        "AVGO",
                        "ASML",
                        "SAP",
                        "SONY",
                        "BABA",
                        "JNJ",
                        "PG",
                        "KO",
                        "PEP",
                        "XOM",
                        "CVX",
                        "JPM",
                        "BAC",
                        "V",
                        "MA",
                        "T",
                        "VZ",
                        "NVO",
                        "RHHBY",
                        "NESN.SW",
                        "ROG.SW",
                        "UBSG.SW",
                    ]
                )
            )
        ),
        default=tickers_default,
        help="Add more by typing a ticker symbol (Yahoo Finance format).",
    )

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start", value=pd.Timestamp.today() - pd.Timedelta(days=365 * 5))
    with col2:
        end = st.date_input("End", value=pd.Timestamp.today())

    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.divider()
    st.subheader("Portfolio Weights (%)")
    weight_inputs = {}
    for t in tickers[:10]:  # keep UI compact; adjust as needed
        weight_inputs[t] = st.number_input(f"{t}", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    st.caption("Unassigned weights → equal-weight applied.")

    st.divider()
    if tickers:
        deep_link = "?" + urlencode({"stocks": ",".join(tickers)})
        st.text_input("Deep Link (copy):", value=deep_link)

# ------------------------------
# Data
# ------------------------------
if not tickers:
    st.info("Select at least one ticker from the sidebar to begin.")
    st.stop()

prices = load_prices(tickers, start, end, interval)
if prices.empty:
    st.warning("No price data returned. Try different dates/interval or symbols.")
    st.stop()

# Align dates & drop rows with any NA
prices = prices.dropna(how="any")

# ------------------------------
# KPIs
# ------------------------------
st.subheader("Key Metrics")
metrics_df = perf_stats(prices)
metrics_fmt = metrics_df.copy()
metrics_fmt["Ann. Return"] = (metrics_fmt["Ann. Return"] * 100).map(lambda x: f"{x:,.2f}%")
metrics_fmt["Ann. Vol"] = (metrics_fmt["Ann. Vol"] * 100).map(lambda x: f"{x:,.2f}%")
metrics_fmt["Sharpe (rf=0)"] = metrics_fmt["Sharpe (rf=0)"].map(lambda x: f"{x:,.2f}")
metrics_fmt["Max Drawdown"] = (metrics_fmt["Max Drawdown"] * 100).map(lambda x: f"{x:,.2f}%")

st.dataframe(metrics_fmt, use_container_width=True)

# ------------------------------
# Charts
# ------------------------------
colA, colB = st.columns(2)
with colA:
    st.markdown("**Indexed Prices (Start = 100)**")
    idx = norm_100(prices).reset_index().melt("Date", var_name="Ticker", value_name="Index")
    chart = (
        alt.Chart(idx)
        .mark_line()
        .encode(x="Date:T", y="Index:Q", color="Ticker:N", tooltip=["Date:T", "Ticker:N", "Index:Q"])
        .interactive()
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

with colB:
    st.markdown("**Rolling Volatility (20-period)**")
    rets = prices.pct_change()
    roll_vol = (rets.rolling(20).std() * np.sqrt(252) * 100).reset_index().melt("Date", var_name="Ticker", value_name="VolPct")
    chart2 = (
        alt.Chart(roll_vol.dropna())
        .mark_line()
        .encode(x="Date:T", y=alt.Y("VolPct:Q", title="Ann. Vol (%)"), color="Ticker:N", tooltip=["Date:T", "Ticker:N", alt.Tooltip("VolPct:Q", format=".2f")])
        .interactive()
        .properties(height=350)
    )
    st.altair_chart(chart2, use_container_width=True)

# Correlation Heatmap
st.markdown("**Correlation (Daily Returns)**")
cor = rets.dropna().corr()
cor_long = cor.reset_index().melt("index", var_name="col", value_name="corr").rename(columns={"index": "row"})
heat = (
    alt.Chart(cor_long)
    .mark_rect()
    .encode(x="row:N", y="col:N", tooltip=["row:N", "col:N", alt.Tooltip("corr:Q", format=".2f")], color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange")))
    .properties(height=300)
)
st.altair_chart(heat, use_container_width=True)

# ------------------------------
# Portfolio Comparison
# ------------------------------
st.header("📦 Portfolio")
weights = pd.Series({k: v / 100.0 for k, v in weight_inputs.items()})
port_custom = portfolio_series(prices, weights)
port_equal = portfolio_series(prices, pd.Series(0.0, index=prices.columns))

port_df = pd.concat([norm_100(prices).mean(axis=1) * 0 + port_custom, port_equal], axis=1)
port_df.columns = ["Custom", "Equal-Weight"]

port_long = port_df.reset_index().melt("Date", var_name="Series", value_name="Index")
chart3 = (
    alt.Chart(port_long)
    .mark_line()
    .encode(x="Date:T", y="Index:Q", color="Series:N", tooltip=["Date:T", "Series:N", alt.Tooltip("Index:Q", format=".2f")])
    .interactive()
    .properties(height=380)
)
st.altair_chart(chart3, use_container_width=True)

# Show normalized weights actually used
if weights.sum() > 0:
    w_used = (weights / weights.sum()).reindex(prices.columns).fillna(0)
else:
    w_used = pd.Series(1 / len(prices.columns), index=prices.columns)

w_tbl = (w_used * 100).to_frame("Weight %").style.format({"Weight %": "{:.2f}"})
st.subheader("Portfolio Weights Used")
st.dataframe(w_tbl, use_container_width=False)

# Download buttons
csv_prices = prices.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Download Prices (CSV)", data=csv_prices, file_name="prices.csv", mime="text/csv")

port_csv = port_df.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Download Portfolio Series (CSV)", data=port_csv, file_name="portfolio.csv", mime="text/csv")

st.caption("Tip: Use the Deep Link in the sidebar to share a pre-filled ticker selection.")
