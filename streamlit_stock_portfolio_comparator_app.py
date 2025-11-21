# -------------------------------------------------------------
# Streamlit Stock & Portfolio Comparator
# University of St. Gallen – Fundamentals and Methods of Computer Science
# Group project: Comparison and analysis of stocks and portfolios
# Group 05.10: Fabian Willem Sprokkereef, Fabrice Rohner,
#              Oliver Nicolas Wick, Ramazan Taskin, Alexander Boris Tschirky
#
# This Streamlit app allows users to
# - select a set of stocks (Yahoo Finance tickers),
# - download adjusted price data via the yfinance package (API requirement),
# - compute and display basic risk/return metrics,
# - build and compare a custom-weighted portfolio with an equal-weight portfolio,
# - visualize the results in interactive charts.
#
# The code is structured into the following parts:
# 1) Helper functions for financial calculations and data loading
# 2) Sidebar controls for user interaction (tickers, dates, interval, portfolio weights)
# 3) Data loading & cleaning
# 4) KPI table computation
# 5) Charts for price indices, rolling volatility, and correlations
# 6) Portfolio construction and comparison
#
# NOTE: This file is intended to be self-contained so that anyone can
#       run it directly with `streamlit run app.py`.
# -------------------------------------------------------------

import streamlit as st               # Streamlit: main framework for building the web app
import pandas as pd                 # Pandas: tabular data handling and time series
import numpy as np                  # NumPy: numerical calculations
import yfinance as yf               # yfinance: wrapper around Yahoo Finance API
import altair as alt                # Altair: declarative charting library
from urllib.parse import urlencode  # urlencode: to build a sharable URL with query parameters

from sklearn.linear_model import LogisticRegression         # ML model for classification
from sklearn.metrics import accuracy_score, confusion_matrix  # Metrics to evaluate the model

# ------------------------------
# App Metadata & Config
# ------------------------------

# Configure basic page settings such as title and layout
st.set_page_config(
    page_title="Stock Comparator & Portfolio",  # Title shown in browser tab
    layout="wide"                               # Use wide layout to have more space for charts
)

# Main title of the app, shown at the top of the page
st.title("📊 Stock & Portfolio Comparator (yfinance)")

# Short caption to explain data source and purpose of the app
st.caption(
    "Built with Streamlit + yfinance. Education-use only. "
    "Prices are adjusted for splits/dividends (auto_adjust=True)."
)

# ------------------------------
# Helper functions
# ------------------------------

def _annualize(ret_series: pd.Series):
    """
    Compute annualized mean return and volatility from a series of returns.

    Parameters
    ----------
    ret_series : pd.Series
        Series of periodic returns (here: daily percentage returns).

    Returns
    -------
    tuple[float, float]
        (annualized_return, annualized_volatility) assuming
        252 trading days per year.
    """
    mu_d = ret_series.mean()             # Average daily return
    sd_d = ret_series.std()              # Standard deviation of daily returns
    ann_ret = (1 + mu_d) ** 252 - 1      # Compound daily mean to annual return
    ann_vol = sd_d * np.sqrt(252)        # Scale daily volatility to annual volatility
    return ann_ret, ann_vol              # Return both metrics as a tuple


def perf_stats(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """
    Calculate performance metrics for each asset in the price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices with dates as index and tickers as columns.
    rf : float, optional
        Constant risk-free rate used for the Sharpe ratio (default: 0.0).

    Returns
    -------
    pd.DataFrame
        Table with annualized return, annualized volatility, Sharpe ratio
        and maximum drawdown for each ticker.
    """
    # Convert price levels to percentage returns
    rets = prices.pct_change().dropna()        # Daily returns, drop first NaN row
    cum = (1 + rets).cumprod()                 # Cumulative performance over time
    dd = cum / cum.cummax() - 1                # Drawdown: distance from running maximum
    stats = {}                                 # Dictionary to collect metrics per ticker

    # Loop over each ticker (column) in the price DataFrame
    for c in prices.columns:
        if c not in rets:                      # Safety check: skip if no returns available
            continue
        ann_ret, ann_vol = _annualize(rets[c]) # Annualized return and volatility for this ticker

        # Sharpe ratio: (return - risk-free) / volatility; guard against division by zero
        sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan

        # Store metrics for this ticker in the dictionary
        stats[c] = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe (rf=0)": sharpe,
            "Max Drawdown": dd[c].min(),       # Minimum drawdown = worst historical drawdown
        }

    # Convert dictionary into DataFrame and transpose:
    # rows = tickers, columns = metrics
    return pd.DataFrame(stats).T


def norm_100(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price columns so that the first available row equals 100.

    This allows us to compare relative performance across stocks.

    Returns an empty DataFrame if input is empty to avoid IndexError.
    """
    if prices is None or prices.empty:         # Handle empty input defensively
        return pd.DataFrame()
    base = prices.iloc[0].replace(0, np.nan)   # Use first row as base, replace zeros with NaN
    return prices.divide(base) * 100.0         # Scale each column so starting value is 100


def portfolio_series(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Build a portfolio index series from individual asset prices and weights.

    Parameters
    ----------
    prices : pd.DataFrame
        Price series (columns = tickers, index = dates).
    weights : pd.Series
        Portfolio weights per ticker (not necessarily normalized).

    Returns
    -------
    pd.Series
        Portfolio value index starting at 100.
    """
    # Align weights index to the columns of the price DataFrame
    weights = weights.reindex(prices.columns).fillna(0)

    # Normalize weights to sum 1 if user specified any positive weights,
    # otherwise fall back to an equal-weight portfolio
    if weights.sum() > 0:
        w = weights / weights.sum()           # Normalize so that weights sum to 1
    else:
        w = pd.Series(1 / len(prices.columns), index=prices.columns)  # Equal weights for each ticker

    # Compute daily portfolio returns as the dot product of returns and weights
    port_ret = prices.pct_change().fillna(0).dot(w)

    # Convert returns to a cumulative index that starts at 100
    return (1 + port_ret).cumprod() * 100.0


@st.cache_data(ttl=30 * 60)                   # Cache data for 30 minutes to avoid repeated downloads
def load_prices(tickers, start, end, interval="1d") -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers from Yahoo Finance.

    This function is cached by Streamlit to avoid repeated API calls when
    the user interacts with the app (e.g., changes tabs or scrolls).

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols in Yahoo Finance format.
    start : date or str
        Start date for the historical data.
    end : date or str
        End date for the historical data.
    interval : {"1d", "1wk", "1mo"}
        Frequency of the downloaded price data.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted close prices with dates as index and tickers
        as columns. Empty DataFrame if no data is returned.
    """
    if not tickers:                           # If user did not select any tickers, return empty DataFrame
        return pd.DataFrame()

    # Call yfinance to download historical price data
    df = yf.download(
        tickers=tickers,                      # Tickers to download
        start=start,                          # Start date
        end=end,                              # End date
        interval=interval,                    # Frequency: daily, weekly, monthly
        auto_adjust=True,                     # Adjust for splits and dividends
        progress=False,                       # Do not show progress bar in terminal
        group_by="ticker",                    # Group columns by ticker symbol
    )

    # yfinance returns a MultiIndex columns when multiple tickers are requested
    if isinstance(df.columns, pd.MultiIndex):
        # Select only the "Close" prices for each ticker from the MultiIndex
        close = df.loc[:, (slice(None), "Close")]
        # Flatten columns: keep only the ticker symbol (first level of the MultiIndex)
        close.columns = [c[0] for c in close.columns]
    else:
        # If only a single ticker is downloaded, df has a single-level column
        close = df[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop columns that are entirely NaN (no data for that ticker)
    close = close.dropna(how="all")

    return close                              # Return cleaned close prices

def build_ml_dataset(prices: pd.DataFrame, ticker: str, lookback: int = 5):
    """
    Build a supervised learning dataset for predicting next-day direction
    of a single stock based on its recent returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices for all tickers (columns) and dates (index).
    ticker : str
        Ticker symbol to build the dataset for.
    lookback : int
        Number of past days to use as features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns for past returns (e.g. ret_t-1, ..., ret_t-5).
    y : pd.Series
        Binary target: 1 if next-day return > 0, else 0.
    """
    # Make sure the requested ticker exists in the price data
    if ticker not in prices.columns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # Compute daily returns for the chosen ticker
    ret = prices[ticker].pct_change().dropna()

    # DataFrame to hold lagged returns and target
    data = pd.DataFrame(index=ret.index)

    # Create one feature column per lagged return
    for lag in range(1, lookback + 1):
        data[f"ret_t-{lag}"] = ret.shift(lag)

    # Target: direction of the next day's return (1 = up, 0 = down or flat)
    data["target"] = (ret.shift(-1) > 0).astype(int)

    # Drop rows with NaN values (created by shifting)
    data = data.dropna()

    # Split into feature matrix X and target vector y
    feature_cols = [f"ret_t-{lag}" for lag in range(1, lookback + 1)]
    X = data[feature_cols]
    y = data["target"]

    return X, y

# ------------------------------
# Sidebar Controls
# The sidebar collects all user inputs: ticker selection, date range,
# data interval, portfolio weights, and a shareable deep link.
# These inputs directly influence data loading and computations below.
# ------------------------------

# Read query parameters from the URL to support deep linking (e.g., ?stocks=AAPL,MSFT)
qparams = st.query_params

# Default list of tickers shown when the app starts
def_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]

# Try to fetch tickers from the URL parameter "stocks" (if present)
url_tickers = qparams.get("stocks")

# Decide which set of tickers should be used as default in the multiselect
if url_tickers:
    # Support both a comma-separated string and a list of values
    tickers_default = (
        url_tickers.split(",") if isinstance(url_tickers, str) else list(url_tickers)
    )
else:
    tickers_default = def_tickers            # Fallback: use predefined default tickers

# Create all sidebar elements within this context manager
with st.sidebar:
    # Section header for user controls
    st.header("⚙️ Controls")

    # Multi-select field for choosing stocks to analyze.
    # Default tickers are pre-filled, but users can add any valid Yahoo Finance symbol.
    tickers = st.multiselect(
        "Tickers",                           # Label shown above the widget
        options=sorted(                      # Sorted list of available tickers
            list(
                set(                         # Use set() to avoid duplicates
                    def_tickers
                    + [                      # Extended list of additional tickers
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
        default=tickers_default,             # Default selection derived from URL or predefined list
        help="Add more by typing a ticker symbol (Yahoo Finance format).",  # Hover help text
    )

    # Two-column layout for selecting start and end dates of the analysis period.
    col1, col2 = st.columns(2)              # Create two equal-width columns in the sidebar

    # Left column: start date input
    with col1:
        start = st.date_input(
            "Start",                        # Label of the date input
            value=pd.Timestamp.today() - pd.Timedelta(days=365 * 5)  # Default: 5 years ago
        )

    # Right column: end date input
    with col2:
        end = st.date_input(
            "End",                          # Label of the date input
            value=pd.Timestamp.today()      # Default: today
        )

    # Selectbox for choosing the data frequency (daily/weekly/monthly).
    interval = st.selectbox(
        "Interval",                         # Label of the selectbox
        ["1d", "1wk", "1mo"],               # Allowed yfinance intervals
        index=0                             # Default: "1d" (daily data)
    )

    # Horizontal separator line to visually separate sections
    st.divider()

    # Users can assign custom portfolio weights (0–100%).
    # Unassigned weights default to an equal-weight portfolio.
    st.subheader("Portfolio Weights (%)")   # Subheader for portfolio weight inputs

    weight_inputs = {}                      # Dictionary to store raw weight inputs from the user
    # Limit number of weight fields to first 10 tickers for readability
    for t in tickers[:10]:
        # Number input field for the weight of each ticker in percent
        weight_inputs[t] = st.number_input(
            f"{t}",                         # Label: ticker symbol
            min_value=0.0,                  # Minimum allowed value
            max_value=100.0,                # Maximum allowed value
            value=0.0,                      # Default value
            step=1.0                        # Step size for increments
        )

    # Brief explanation of how unspecified weights will be handled
    st.caption("Unassigned weights → equal-weight applied.")

    # Another separator before the deep link section
    st.divider()

    # Only build a deep link if at least one ticker is chosen
    if tickers:
        # Build a query string like "?stocks=AAPL,MSFT,NVDA"
        deep_link = "?" + urlencode({"stocks": ",".join(tickers)})
        # Text input field to show the generated deep link for copy-paste
        st.text_input("Deep Link (copy):", value=deep_link)

# ------------------------------
# Data loading and cleaning
# ------------------------------

# If no tickers are selected, inform the user and stop the app
if not tickers:
    st.info("Select at least one ticker from the sidebar to begin.")
    st.stop()                                # Halts the script to avoid further errors

# Load price data for the selected tickers and settings
prices = load_prices(tickers, start, end, interval)

# If the DataFrame is empty, show a warning and stop
if prices.empty:
    st.warning("No price data returned. Try different dates/interval or symbols.")
    st.stop()

# Align dates & drop rows with any missing values across tickers
prices = prices.dropna(how="any")

# If all rows were dropped due to missing data, inform user and stop
if prices.empty:
    st.warning(
        "No rows left after cleaning. Try a wider date range, "
        "a different interval (e.g., 1wk), or other tickers."
    )
    st.stop()

# ------------------------------
# KPIs (Key Performance Indicators)
# ------------------------------

# Subheader for the KPI section
st.subheader("Key Metrics")

# Calculate performance statistics for each ticker
metrics_df = perf_stats(prices)

# Create a copy for pretty formatting of the values
metrics_fmt = metrics_df.copy()

# Format annualized return as percentage string with two decimals
metrics_fmt["Ann. Return"] = (metrics_fmt["Ann. Return"] * 100).map(
    lambda x: f"{x:,.2f}%"
)

# Format annualized volatility as percentage string with two decimals
metrics_fmt["Ann. Vol"] = (metrics_fmt["Ann. Vol"] * 100).map(
    lambda x: f"{x:,.2f}%"
)

# Format Sharpe ratio with two decimal places
metrics_fmt["Sharpe (rf=0)"] = metrics_fmt["Sharpe (rf=0)"].map(
    lambda x: f"{x:,.2f}"
)

# Format maximum drawdown as percentage string with two decimals
metrics_fmt["Max Drawdown"] = (metrics_fmt["Max Drawdown"] * 100).map(
    lambda x: f"{x:,.2f}%"
)

# Render the KPI table as an interactive DataFrame in Streamlit
st.dataframe(metrics_fmt, use_container_width=True)

# ------------------------------
# Charts
# We visualize: (1) indexed prices to compare relative performance,
# (2) rolling volatility as a risk measure, and (3) return correlations.
# These visualizations help users understand price dynamics and
# relationships between assets.
# ------------------------------

# Create two side-by-side columns for the first two charts
colA, colB = st.columns(2)

# ----- Chart 1: Indexed Prices -----
with colA:
    # Chart 1: Indexed prices (start value = 100) to allow direct comparison.
    st.markdown("**Indexed Prices (Start = 100)**")

    # Convert normalized price data into long format for Altair charts
    idx = norm_100(prices).reset_index().melt(
        "Date", var_name="Ticker", value_name="Index"
    )

    # Define the Altair line chart for indexed prices
    chart = (
        alt.Chart(idx)
        .mark_line()
        .encode(
            x="Date:T",                      # Date on x-axis (temporal type)
            y="Index:Q",                     # Indexed price level on y-axis (quantitative)
            color="Ticker:N",                # Different color per ticker (nominal)
            tooltip=["Date:T", "Ticker:N", "Index:Q"],  # Tooltip on hover
        )
        .interactive()                       # Enable interactive zooming and panning
        .properties(height=350)              # Chart height in pixels
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

# ----- Chart 2: Rolling Volatility -----
with colB:
    # Chart 2: Rolling annualized volatility using a 20-period window.
    # Shows how risk evolves over time.
    st.markdown("**Rolling Volatility (20-period)**")

    # Recalculate daily returns from price data
    rets = prices.pct_change()

    # Compute rolling 20-period annualized volatility (in percentage)
    roll_vol = (
        rets.rolling(20).std() * np.sqrt(252) * 100
    ).reset_index().melt(
        "Date", var_name="Ticker", value_name="VolPct"
    )

    # Define the Altair line chart for rolling volatility
    chart2 = (
        alt.Chart(roll_vol.dropna())        # Drop rows with missing volatility
        .mark_line()
        .encode(
            x="Date:T",                     # Date on x-axis
            y=alt.Y("VolPct:Q", title="Ann. Vol (%)"),  # Annualized volatility in %
            color="Ticker:N",               # Different color per ticker
            tooltip=[                       # Tooltip showing date, ticker, volatility
                "Date:T",
                "Ticker:N",
                alt.Tooltip("VolPct:Q", format=".2f"),
            ],
        )
        .interactive()                      # Enable interactivity
        .properties(height=350)             # Chart height
    )

    # Display the rolling volatility chart
    st.altair_chart(chart2, use_container_width=True)

# ----- Chart 3: Correlation Heatmap -----

# Chart 3: Heatmap of correlations of daily returns.
# Helps identify diversification potential between assets.
st.markdown("**Correlation (Daily Returns)**")

# Compute correlation matrix of daily returns
cor = rets.dropna().corr()

# Convert correlation matrix to long format for visualization
cor_long = cor.reset_index().melt(
    "index", var_name="col", value_name="corr"
).rename(columns={"index": "row"})

# Define Altair heatmap chart for the correlation matrix
heat = (
    alt.Chart(cor_long)
    .mark_rect()
    .encode(
        x="row:N",                          # Row ticker on x-axis (nominal)
        y="col:N",                          # Column ticker on y-axis (nominal)
        tooltip=[                           # Tooltip shows pair and correlation value
            "row:N",
            "col:N",
            alt.Tooltip("corr:Q", format=".2f"),
        ],
        color=alt.Color(                    # Color encodes magnitude and direction of correlation
            "corr:Q",
            scale=alt.Scale(scheme="blueorange"),
        ),
    )
    .properties(height=300)                 # Height of the heatmap
)

# Display the correlation heatmap
st.altair_chart(heat, use_container_width=True)

# ------------------------------
# Portfolio Comparison
# We build two portfolios:
# 1) Custom portfolio using user-defined weights
# 2) Equal-weight portfolio as benchmark
# Both portfolios are normalized to 100 to compare performance visually.
# ------------------------------

# Section header for portfolio comparison
st.header("📦 Portfolio")

# Convert raw weight inputs (percent) into a pandas Series with weights in [0,1]
weights = pd.Series({k: v / 100.0 for k, v in weight_inputs.items()})

# Compute the custom-weighted and equal-weighted portfolio series.
port_custom = portfolio_series(prices, weights)               # Portfolio using user-defined weights
port_equal = portfolio_series(                                # Equal-weight benchmark portfolio
    prices,
    pd.Series(0.0, index=prices.columns)                      # Passing zeros triggers equal-weight logic
)

# Combine both portfolio series into a single DataFrame for plotting
port_df = pd.concat(
    [
        norm_100(prices).mean(axis=1) * 0 + port_custom,      # Custom portfolio index
        port_equal,                                           # Equal-weight portfolio index
    ],
    axis=1,
)

# Name the two portfolio columns
port_df.columns = ["Custom", "Equal-Weight"]

# Convert portfolio DataFrame into long format for Altair
port_long = port_df.reset_index().melt(
    "Date", var_name="Series", value_name="Index"
)

# Define Altair line chart to compare portfolio performance
chart3 = (
    alt.Chart(port_long)
    .mark_line()
    .encode(
        x="Date:T",                         # Date on x-axis
        y="Index:Q",                        # Portfolio index level on y-axis
        color="Series:N",                   # Color by portfolio type (Custom vs Equal-Weight)
        tooltip=[                           # Tooltip with date, series, and index value
            "Date:T",
            "Series:N",
            alt.Tooltip("Index:Q", format=".2f"),
        ],
    )
    .interactive()                          # Enable interactive features
    .properties(height=380)                 # Height of portfolio comparison chart
)

# Display portfolio comparison chart
st.altair_chart(chart3, use_container_width=True)

# Show normalized weights actually used
if weights.sum() > 0:
    # Normalize user-specified weights if there is at least one positive weight
    w_used = (weights / weights.sum()).reindex(prices.columns).fillna(0)
else:
    # If no weights are specified, show equal-weight distribution
    w_used = pd.Series(1 / len(prices.columns), index=prices.columns)

# Display the normalized weights actually applied (summing to 100%).
w_tbl = (w_used * 100).to_frame("Weight %").style.format(
    {"Weight %": "{:.2f}"}
)

# Subheader for the weights table
st.subheader("Portfolio Weights Used")

# Show the final weights table in the app
st.dataframe(w_tbl, use_container_width=False)

st.subheader("Portfolio Weights Used")
st.dataframe(w_tbl, use_container_width=False)

# ------------------------------
# Machine Learning: Next-day Direction
# This section implements requirement 5:
# A simple ML model (logistic regression) that predicts whether
# the next day's return will be up or down for a selected stock.
# ------------------------------

st.header("🔮 Machine Learning: Next-day Direction")

# Let the user choose which ticker to use for the ML model
ml_ticker = st.selectbox(
    "Select ticker for ML prediction",
    options=tickers,
    index=0,
    help="Choose one stock to train a model that predicts whether the next day's return is up or down."
)

# Build the ML dataset for the selected ticker (features X and target y)
X, y = build_ml_dataset(prices, ml_ticker, lookback=5)

if X.empty or len(y) < 30:
    # If there is not enough historical data, show an info message
    st.info(
        "Not enough data to train a model for this ticker and time range. "
        "Try a longer time period or select a different ticker."
    )
else:
    # Time-based train/test split (no shuffling, to respect time order)
    split_idx = int(len(X) * 0.7)  # 70% of observations for training, 30% for testing
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Initialize logistic regression classifier
    model = LogisticRegression(max_iter=1000)

    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the held-out test set
    y_pred = model.predict(X_test)

    # Compute accuracy of the classifier on the test data
    acc = accuracy_score(y_test, y_pred)

    # Compute confusion matrix (rows: actual, columns: predicted)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Show accuracy in the app
    st.subheader("Model Performance")
    st.write(f"**Accuracy on test set:** {acc:.2%}")

    # Create a labeled confusion matrix DataFrame
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Down/Flat (0)", "Actual Up (1)"],
        columns=["Predicted Down/Flat (0)", "Predicted Up (1)"],
    )

    # Show confusion matrix in Streamlit
    st.write("**Confusion matrix:**")
    st.dataframe(cm_df, use_container_width=True)

    # Show model coefficients to understand feature importance
    coef_df = pd.DataFrame(
        model.coef_.T,
        index=X.columns,
        columns=["Coefficient"],
    )

    st.write("**Model coefficients (importance of each past return feature):**")
    st.dataframe(coef_df.style.format({"Coefficient": "{:.4f}"}), use_container_width=True)

# ------------------------------
# Download buttons
# ------------------------------
csv_prices = prices.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Download Prices (CSV)", data=csv_prices, file_name="prices.csv", mime="text/csv")

port_csv = port_df.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Download Portfolio Series (CSV)", data=port_csv, file_name="portfolio.csv", mime="text/csv")

st.caption("Tip: Use the Deep Link in the sidebar to share a pre-filled ticker selection.")


# ------------------------------
# Download options
# ------------------------------

# Convert price data to CSV bytes for download
csv_prices = prices.to_csv(index=True).encode("utf-8")

# Button to download the underlying price data as CSV
st.download_button(
    "⬇️ Download Prices (CSV)",
    data=csv_prices,
    file_name="prices.csv",
    mime="text/csv",
)

# Convert portfolio index data to CSV bytes for download
port_csv = port_df.to_csv(index=True).encode("utf-8")

# Button to download the portfolio index series as CSV
st.download_button(
    "⬇️ Download Portfolio Series (CSV)",
    data=port_csv,
    file_name="portfolio.csv",
    mime="text/csv",
)

# Final caption reminding users about the deep link functionality
st.caption(
    "Tip: Use the Deep Link in the sidebar to share a pre-filled ticker selection."
)
