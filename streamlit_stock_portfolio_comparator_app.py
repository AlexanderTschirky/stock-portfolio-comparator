# -----------------------------------------------------------------------------
# SNIPPET 1: IMPORTS & CONFIGURATION
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf  # We need this now to fetch financial data later

st.set_page_config(page_title="Stock Comparator", layout="wide")

st.title("📊 Stock & Portfolio Comparator")

# -----------------------------------------------------------------------------
# SNIPPET 2: SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
# We use 'with st.sidebar' to place everything inside this block on the left side.
with st.sidebar:
    st.header("⚙️ Controls")

    # 1. TICKER SELECTION
    # We define a list of tickers that the user can choose from.
    # (In the future, we could allow typing any ticker, but a list is easier for now).
    available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
    
    # st.multiselect(label, options, default)
    # 'tickers' will be a LIST of strings, e.g., ['AAPL', 'MSFT']
    tickers = st.multiselect(
        "Select Tickers", 
        options=available_tickers,
        default=["AAPL", "MSFT"] # Pre-selected when app loads
    )

    # 2. DATE SELECTION
    # We put the start and end dates side-by-side using columns.
    col1, col2 = st.columns(2)
    
    with col1:
        # Default start date: Jan 1, 2020
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        
    with col2:
        # Default end date: Today
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# -----------------------------------------------------------------------------
# DEBUGGING / VERIFICATION
# -----------------------------------------------------------------------------
# This is just to show you what the variables contain right now.
# We will delete this part later.
st.write("---")
st.write("You have selected these tickers:", tickers)
st.write(f"From {start_date} to {end_date}")
