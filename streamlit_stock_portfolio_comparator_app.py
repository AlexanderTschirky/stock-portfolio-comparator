# -----------------------------------------------------------------------------
# IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------

# This application was built using Google Gemini. We created the base code in cooperation with Google Gemini, modified and commented the code ourselves.

# We import the libraries we need.
import streamlit as st # Streamlit is the framework we use to build the web app.
import pandas as pd # Pandas is the tool we use for tabular data handling.
import yfinance as yf # yfinance is the library we use tu fetch stock data from the Yahoo Finance API.
import numpy as np # Numpy is the library we use for mathematical calculations.
import altair as alt # Altair is used for advanced charts.
from sklearn.ensemble import RandomForestRegressor # We need Sklearn for the Machine Learning part.
from sklearn.metrics import mean_absolute_error # We need Sklearn for the Machine Learning part.

# This must be the first Streamlit command. It sets up the page title and layout.
st.set_page_config(page_title="SMI Stock & Portfolio Comparator", layout="wide") # This sets up the page title in the browser

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_KPI(df, risk_free_rate=0.0): 
    """
    We want to calculate different KPI's of the stocks to later compare the stocks to each other in the application.
    We put it all in one function for efficiency and simplicity, as there are some variables we need for different KPI's.
    """
    
    summary = pd.DataFrame(index=df.columns) # We first create an empty data frame to store our results.
    
    # 1. Daily Returns
    returns = df.pct_change().dropna() # We calculate the precentage change in price from yesterday to today. The command ".dropna()" removes the first row, which has no yesterday to compare to.
    
    # 2. Annualized Return
    summary['Ann. Return'] = returns.mean() * 252 # We calculate the average daily return and scale it up to a year. We asssume 252 trading days in a year.

    # 3. Cumulative Return
    summary['Cumulative Return'] = (1 + returns).prod() - 1 # We calculate the total percentage gain/loss over the selected period.
    
    # 4. Annualized Volatility (Risk)
    summary['Ann. Volatility'] = returns.std() * np.sqrt(252) # We calculate the annualized volatility. We assume 252 trading days in a year.
    
    # 5. Sharpe Ratio
    # We use the user-defined risk_free_rate passed to the function
    summary['Sharpe Ratio'] = (summary['Ann. Return'] - risk_free_rate) / summary['Ann. Volatility'] 

    # 6. Sortino Ratio
    downside_returns = returns.copy() # We copy the returns into the new variable "downside_returns" to further process the data.
    downside_returns[downside_returns > 0] = np.nan # For the Sortino Ratio, we neglect upside volatility, therefore we do not consider positive returns for the calculation.
    annual_downside_vol = downside_returns.std() * np.sqrt(252) # We calculate the annual volatility only for negative days.
    # We use the user-defined risk_free_rate passed to the function
    summary['Sortino Ratio'] = (summary['Ann. Return'] - risk_free_rate) / annual_downside_vol # We calculate the Sortino Ratio.
    
    # 7. Max Drawdown
    # The "Worst Case Scenario": buying at the peak and selling at the bottom.
    # We calculate the cumulative return (growth of $1).
    cumulative_returns_series = (1 + returns).cumprod() # We calculate the cumulative return series, so that we get a value for each day.
    running_max = cumulative_returns_series.cummax() # We store the highest value seen so far as the "running_max"
    drawdown = (cumulative_returns_series / running_max) - 1 # We get a drawdown-value for every day.
    summary['Max Drawdown'] = drawdown.min() # We define the miminum value (the most negative value) of the "drawdown" data frame as the "Max Drawdown".

    # 8. Value at Risk (VaR)
    summary['Value at Risk (95%)'] = returns.quantile(0.05) # We define the 5th percentile of daily returns as the VaR at the 95%-level.
    
    return summary

def prepare_regression_data(series, window=21, horizon=1):
    """
    For the Machine Learning part, we want to predict next-day absolute return (volatility) of a stock.
    We use the absolute returns of the last trading month (21 days) for this.
    """
    # Robust check: If input is already a DataFrame, use it directly. If Series, convert.
    if isinstance(series, pd.DataFrame):
        df = series.copy()
        if len(df.columns) > 0:
            df.columns = ['Close']
    else:
        df = series.to_frame(name='Close') 
        
    df['Abs_Return'] = df['Close'].pct_change().abs() # We calculate the absolute daily returns
    
    # TARGET CREATION:
    if horizon == 1:
        df['Target'] = df['Abs_Return'].shift(-1)
    else:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        df['Target'] = df['Abs_Return'].rolling(window=indexer).mean()

    # Features: Recent volatility (Lag 1 to Lag 21)
    for i in range(1, window + 1): # We start a loop which will run 21 times
        df[f'Vol_Lag_{i}'] = df['Abs_Return'].shift(i) # We create new columns for the volatility of each day  
    
    df = df.dropna() # We remove any row that has missing data to avoid a crash of the model
    feature_cols = [f'Vol_Lag_{i}' for i in range(1, window + 1)] # This creates a list of the column names we created
    return df[feature_cols], df['Target'] # The function returns two separate tables, one containing the lag-columns, one containing the "Target". 

# -----------------------------------------------------------------------------
# LOADING DATA FUNCTION
# -----------------------------------------------------------------------------
@st.cache_data 
def load_data(ticker_list, start, end): 
    if not ticker_list:
        return pd.DataFrame() 
    
    safe_start = pd.Timestamp.today() - pd.DateOffset(years=2)
    download_start = min(start, safe_start)
    
    data = yf.download(ticker_list, start=download_start, end=end, auto_adjust=True) 
    
    if len(ticker_list) == 1:
        return data['Close'].to_frame(name=ticker_list[0]) 
    
    return data['Close']

# -----------------------------------------------------------------------------
# PAGE STRUCTURE & NAVIGATION
# -----------------------------------------------------------------------------

# SIDEBAR: NAVIGATION ONLY
with st.sidebar: 
    st.title("Navigation")
    page = st.radio("Go to", ["KPI Visualizer", "Risk & Correlation", "Volatility Forecasting"])
    st.markdown("---")
    st.info("ℹ️ Note: All analysis controls (Stocks, Dates, Weights) are now located at the top of the main page.")

# MAIN PAGE TITLE
st.title("📈 SMI Stock & Portfolio Comparator")

# -----------------------------------------------------------------------------
# TOP CONTROLS (MOVED FROM SIDEBAR)
# -----------------------------------------------------------------------------
# We create an expander to hold all the settings so they don't take up too much space initially.
with st.expander("⚙️ Analysis Settings & Portfolio", expanded=True):
    
    smi_companies = { 
        "^SSMI": "🇨🇭 Swiss Market Index (Benchmark)", 
        "ROG.SW": "Roche", 
        "NESN.SW": "Nestlé",
        "NOVN.SW": "Novartis",
        "UBSG.SW": "UBS Group",
        "ZURN.SW": "Zurich Insurance",
        "CFR.SW": "Richemont",
        "ABBN.SW": "ABB",
        "SIKA.SW": "Sika",
        "LONN.SW": "Lonza",
        "ALC.SW": "Alcon",
        "GIVN.SW": "Givaudan",
        "HOLN.SW": "Holcim",
        "SCMN.SW": "Swisscom",
        "PGHN.SW": "Partners Group",
        "SLHN.SW": "Swiss Life",
        "GEBN.SW": "Geberit",
        "SOON.SW": "Sonova",
        "SREN.SW": "Swiss Re",
        "KNIN.SW": "Kuehne + Nagel",
        "LOGN.SW": "Logitech"
    }

    # Row 1: Stock Selection & Dates
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    
    with col_c1:
        selectable_tickers = [t for t in smi_companies.keys() if t != "^SSMI"] 
        tickers = st.multiselect( 
            "Select Stocks", 
            options=selectable_tickers, 
            format_func=lambda x: f"{smi_companies[x]} ({x})", 
            default=["NESN.SW", "NOVN.SW", "UBSG.SW"] 
        )

    with col_c2:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01")) 
    
    with col_c3:
        end_date = st.date_input("End Date", value=pd.to_datetime("today")) 

    # Row 2: Portfolio Builder & Market Assumptions
    st.markdown("---")
    col_p1, col_p2 = st.columns([2, 1])
    
    weights = {} 
    risk_free_rate_val = 0.0

    with col_p1:
        st.subheader("⚖️ Portfolio Weights")
        if tickers:
            st.caption("Assign percentage weights. Must sum to 100%.")
            
            # Create columns for weights to save vertical space
            # We split the tickers into chunks of 3 for display
            cols = st.columns(3)
            default_weight = round(100.0 / len(tickers), 2) 
            
            for i, t in enumerate(tickers):
                col_idx = i % 3
                with cols[col_idx]:
                    name = smi_companies[t]
                    weights[t] = st.number_input(f"{name} (%)", min_value=0.0, max_value=100.0, value=default_weight, step=1.0)

            current_total = sum(weights.values())
            st.write(f"**Total Allocation:** {current_total:.1f}%") 
            
            if abs(current_total - 100.0) > 0.1: 
                st.error("⚠️ Total must be exactly 100%") 
            else:
                st.success("✅ Portfolio Ready") 
        else:
            st.info("Select stocks to build a portfolio.")

    with col_p2:
        st.subheader("🏦 Assumptions")
        rf_input = st.number_input(
            "Risk Free Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Used for Sharpe/Sortino Ratios."
        )
        risk_free_rate_val = rf_input / 100.0

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------
try:
    # 1. PREPARE TICKER LIST
    tickers_to_load = list(set(tickers + ["^SSMI"])) 

    # 2. CALL THE FUNCTION
    full_history_df = load_data(tickers_to_load, pd.Timestamp(start_date), pd.Timestamp(end_date)) 
    
    # 3. CHECK IF DATA IS EMPTY
    if full_history_df.empty:
        st.warning("No data found. Please check your date range.") 
    else:
        # -----------------------------------------------------------------------------
        # DATA PRE-PROCESSING & PORTFOLIO CALCULATION (ON FULL HISTORY)
        # -----------------------------------------------------------------------------
        cleaned_df = full_history_df.dropna()
        
        valid_portfolio = False 
        current_total = sum(weights.values())
        
        if tickers and not cleaned_df.empty and abs(current_total - 100.0) <= 0.1: 
            valid_portfolio = True 
    
            selected_tickers = cleaned_df[tickers]
            daily_returns = selected_tickers.pct_change()
            final_weights = [weights[t] / 100.0 for t in tickers] 
            
            portfolio_ret = daily_returns.dot(final_weights) 
            
            # Construct Price Series
            my_portfolio_price = (1 + portfolio_ret).cumprod() * 100 
            my_portfolio_price.iloc[0] = 100 
            
            cleaned_df["💼 My Portfolio"] = my_portfolio_price 

        # -----------------------------------------------------------------------------
        # CREATE DISPLAY DATAFRAME (FILTERED BY USER DATE)
        # -----------------------------------------------------------------------------
        display_start = pd.Timestamp(start_date).tz_localize(None)
        display_end = pd.Timestamp(end_date).tz_localize(None)
        display_df = cleaned_df.loc[display_start:display_end]

        # -----------------------------------------------------------------------------
        # PAGE 1: KPI VISUALIZER
        # -----------------------------------------------------------------------------
        if page == "KPI Visualizer":
            st.subheader("📊 KPI Visualizer over Time")
            
            # Raw Data Preview
            with st.expander("📄 View Last 21 Trading Days"): 
                 preview_df = display_df.rename(columns=lambda x: smi_companies.get(x, x)) 
                 st.dataframe(preview_df.tail(21))
                 
            csv_data = preview_df.to_csv().encode('utf-8') 
                 
            st.download_button( 
                label="⬇️ Download Raw Price Data (CSV)", 
                data=csv_data, 
                file_name="stock_price_data.csv",
                mime="text/csv"
            )

            if not display_df.empty:
                metric_options = [ 
                    "Cumulative Return (Indexed to 100)",
                    "Annualized Return (30-Day Rolling)",
                    "Volatility (30-Day Rolling)",
                    "Sharpe Ratio (30-Day Rolling)",
                    "Sortino Ratio (30-Day Rolling)",
                    "Drawdown (Historical)",
                    "Value at Risk 95% (30-Day Rolling)"
                ]
                
                selected_metric = st.selectbox("Select Metric to Plot", metric_options)
                
                # Use display_df for visual analysis
                returns = display_df.pct_change().dropna() 
                window = 30 
                
                if selected_metric == "Cumulative Return (Indexed to 100)":
                    plot_data = display_df / display_df.iloc[0] * 100
                    
                elif selected_metric == "Annualized Return (30-Day Rolling)":
                    plot_data = returns.rolling(window=window).mean() * 252
                
                elif selected_metric == "Volatility (30-Day Rolling)":
                    plot_data = returns.rolling(window=window).std() * np.sqrt(252)
                    
                elif selected_metric == "Sharpe Ratio (30-Day Rolling)":
                    rolling_return = returns.rolling(window=window).mean() * 252
                    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
                    plot_data = (rolling_return - risk_free_rate_val) / rolling_vol
                
                elif selected_metric == "Sortino Ratio (30-Day Rolling)":
                    downside = returns.copy()
                    downside[downside > 0] = np.nan
                    rolling_downside_vol = downside.rolling(window=window).std() * np.sqrt(252)
                    rolling_return = returns.rolling(window=window).mean() * 252
                    plot_data = (rolling_return - risk_free_rate_val) / rolling_downside_vol
                    
                elif selected_metric == "Drawdown (Historical)":
                    cumulative_rets = (1 + returns).cumprod()
                    running_max = cumulative_rets.cummax()
                    plot_data = (cumulative_rets / running_max) - 1
                    
                elif selected_metric == "Value at Risk 95% (30-Day Rolling)":
                    plot_data = returns.rolling(window=window).quantile(0.05)


                plot_data = plot_data.rename(columns=lambda x: smi_companies.get(x, x)) 
                st.line_chart(plot_data) 
                
            else:
                st.info("Not enough shared data points to plot a comparison. Try adjusting dates.") 

        # -----------------------------------------------------------------------------
        # PAGE 2: RISK & CORRELATION
        # -----------------------------------------------------------------------------
        elif page == "Risk & Correlation":
            st.subheader("📉 Risk & Return Analysis")
            
            metrics_df = calculate_KPI(display_df, risk_free_rate=risk_free_rate_val) 
            
            metrics_df = metrics_df.rename(index=lambda x: smi_companies.get(x, x)) 

            metrics_df.index.name = "Stock" 
            scatter_data = metrics_df.reset_index() 
            
            col_mapping = {
                'Ann. Return': 'Annualized Return',
                'Cumulative Return': 'Cumulative Return',
                'Ann. Volatility': 'Annualized Volatility',
                'Sharpe Ratio': 'Sharpe Ratio',
                'Sortino Ratio': 'Sortino Ratio',
                'Max Drawdown': 'Max Drawdown',
                'Value at Risk (95%)': 'Value at Risk 95%'
            }
            
            scatter_data = scatter_data.rename(columns=col_mapping) 

            st.markdown("##### Compare Metrics (Scatter Plot)") 
            col_x, col_y = st.columns(2) 
            
            chart_opts = list(col_mapping.values()) 
            
            with col_x:
                x_axis = st.selectbox("X-Axis", chart_opts, index=chart_opts.index('Annualized Volatility'))
            with col_y:
                y_axis = st.selectbox("Y-Axis", chart_opts, index=chart_opts.index('Annualized Return'))
                
            x_format = ".2f" if "Ratio" in x_axis else "%"
            y_format = ".2f" if "Ratio" in y_axis else "%"
            
            chart = alt.Chart(scatter_data).mark_circle(size=100).encode(
                x=alt.X(x_axis, title=x_axis, axis=alt.Axis(format=x_format)),
                y=alt.Y(y_axis, title=y_axis, axis=alt.Axis(format=y_format)),
                color='Stock',
                tooltip=['Stock'] + chart_opts
            ).interactive() 
            
            st.altair_chart(chart, use_container_width=True) 
            
            formatted_metrics = metrics_df.style.format({
                'Ann. Return': '{:.2%}',
                'Cumulative Return': '{:.2%}',
                'Ann. Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Sortino Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                'Value at Risk (95%)': '{:.2%}'
            })
            
            st.markdown("##### Detailed Metrics Table") 
            st.dataframe(formatted_metrics) 

            # Correlation Section
            st.markdown("---")
            st.subheader("🔗 Correlation Matrix")
            st.write("This heatmap shows how stocks move together. +1 means they move perfectly in sync (blue), -1 means they move in opposite directions (red).")

            if not display_df.empty:
                corr_returns = display_df.pct_change().dropna()
                corr_matrix = corr_returns.corr()
                
                if len(corr_matrix.columns) > 1:
                    corr_matrix_renamed = corr_matrix.rename(index=lambda x: smi_companies.get(x, x), columns=lambda x: smi_companies.get(x, x))
                    corr_data = corr_matrix_renamed.reset_index()
                    corr_data = corr_data.rename(columns={corr_data.columns[0]: 'Stock A'})
                    corr_data = corr_data.melt(id_vars='Stock A')
                    corr_data.columns = ['Stock A', 'Stock B', 'Correlation']

                    heatmap = alt.Chart(corr_data).mark_rect().encode(
                        x=alt.X('Stock A', title=None),
                        y=alt.Y('Stock B', title=None),
                        color=alt.Color('Correlation', scale=alt.Scale(domain=[-1, 1], scheme='redblue')),
                        tooltip=['Stock A', 'Stock B', alt.Tooltip('Correlation', format='.2f')]
                    ).properties(
                        height=500
                    )

                    text = heatmap.mark_text(baseline='middle').encode(
                        text=alt.Text('Correlation', format='.2f'),
                        color=alt.condition(
                            (alt.datum.Correlation > 0.5) | (alt.datum.Correlation < -0.5),
                            alt.value('white'),
                            alt.value('black')
                        )
                    )

                    st.altair_chart(heatmap + text, use_container_width=True)
                else:
                    st.info("Select at least 2 stocks (or 1 stock + Benchmark) to view correlations.")

        # -----------------------------------------------------------------------------
        # PAGE 3: VOLATILITY FORECASTING
        # -----------------------------------------------------------------------------
        elif page == "Volatility Forecasting":
            st.subheader("🤖 Machine Learning: Volatility Prediction") 
            
            st.write("""
            This model predicts the **Exact Volatility** (Average Absolute Daily Return) over a specific time horizon.
            It uses the past 21 days of volatility to learn patterns using a Random Forest Regressor.
            """) 
            
            ml_opts = list(cleaned_df.columns) 
            col_ml_1, col_ml_2 = st.columns(2)
            
            with col_ml_1:
                ml_ticker = st.selectbox("Select Stock to Predict", ml_opts, format_func=lambda x: smi_companies.get(x, x)) 
            
            with col_ml_2:
                horizon_dict = {"Next Day": 1, "Next Week (5 Days)": 5, "Next Month (21 Days)": 21}
                horizon_label = st.selectbox("Select Forecast Horizon", list(horizon_dict.keys()))
                horizon_val = horizon_dict[horizon_label]

            if ml_ticker:
                subset_series = cleaned_df[ml_ticker].dropna()
                X, y = prepare_regression_data(subset_series, window=21, horizon=horizon_val)
                
                if len(X) > 50: 
                    split_index = int(len(X) * 0.8) 
                    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42) 
                    model.fit(X_train, y_train) 
                    
                    preds = model.predict(X_test) 
                    mae = mean_absolute_error(y_test, preds) 
                    
                    st.markdown(f"#### Volatility Forecast for **{smi_companies.get(ml_ticker, ml_ticker)}** ({horizon_label})") 
                    
                    last_21_days = X.iloc[-1:].values 
                    next_val_pred = model.predict(last_21_days)[0] 
                    
                    col1, col2 = st.columns(2) 
                    col1.metric(f"Predicted Volatility ({horizon_label})", f"{next_val_pred:.2%}")
                    col2.metric("Mean Absolute Error (Test Set)", f"{mae:.2%}")

                    results_df = pd.DataFrame({
                        'Date': y_test.index, 
                        'Actual Volatility': y_test.values, 
                        'Predicted Volatility': preds 
                    }).set_index('Date') 
                    
                    st.write(f"**Predicted vs. Actual Volatility ({horizon_label}):**") 
                    st.line_chart(results_df) 
                    st.caption("The lower the ratio of MAE to Volatility, the more accurate our model is.")
                    
                else:
                    st.warning("Not enough data. Try a longer date range.") 

except Exception as e:
    st.error(f"An error occurred: {e}")
