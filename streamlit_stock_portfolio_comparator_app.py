# -------------------------------------------------------------
# Streamlit Stock & Portfolio Comparator (V4 - Finalisiert für FCS-BWL)
# University of St. Gallen – Fundamentals and Methods of Computer Science
# Group project: Comparison and analysis of stocks and portfolios
#
# Zielsetzung: Anwendung aller Kurskonzepte (Funktionen, If/Else, Sequenzen, DB-Anbindung)
# und Ergänzung um eigene Features (Benchmark, VaR, ML-Sicherheit).
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import sqlite3 # NEU: Für Datenbank-Anforderung (Woche 7)
from urllib.parse import urlencode

# ML-Bibliotheken (Woche 11/Projektanforderung 5)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------
# App Metadata & Config
# ------------------------------

st.set_page_config(
    page_title="📊 Aktien- & Portfolio-Performance-Analyse (HSG Projekt)",
    layout="wide"
)

# NEU: Schärfere Problemstellung/Business Case (Anforderung 1)
st.title("📈 Aktien- & Portfolio-Performance-Analyse")
st.caption(
    "Vergleiche Aktien, baue ein Portfolio und messe es am Markt-Benchmark. "
    "Daten via yfinance API. Nur für Bildungszwecke. (Anforderung 1, 2)"
)

# ------------------------------
# Helper functions
# ------------------------------

def _annualize(ret_series: pd.Series):
    """Berechnet annualisierte Rendite und Volatilität (Basis: 252 Handelstage)."""
    mu_d = ret_series.mean()
    sd_d = ret_series.std()
    ann_ret = (1 + mu_d) ** 252 - 1
    ann_vol = sd_d * np.sqrt(252)
    return ann_ret, ann_vol


# NEU: VaR-Berechnung in perf_stats integriert (Anwendung von Sequenzen/Listen)
def perf_stats(prices: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """
    Berechnet Performance-Metriken (Sharpe, Max Drawdown, Daily VaR) für jede Anlage.
    """
    rets = prices.pct_change().dropna()
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    stats = {}

    for c in prices.columns:
        if c not in rets:
            continue
        ann_ret, ann_vol = _annualize(rets[c])

        # Nutzung einer If/Else-Kontrollstruktur, um Division durch Null zu vermeiden (Woche 2)
        if ann_vol != 0:
            sharpe = (ann_ret - rf) / ann_vol
        else:
            sharpe = np.nan
            
        # NEU: Berechnung des Value at Risk (VaR)
        # Sucht das 5%-Quantil der Verluste (95% VaR). Anwendung von Sequenz-Konzepten.
        daily_rets = rets[c].values 
        # VaR ist das negative 5. Perzentil, als positive Zahl in Dezimalform
        var_95 = np.abs(np.percentile(daily_rets, 5)) 
        
        stats[c] = {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe (rf=0)": sharpe,
            "Max Drawdown": dd[c].min(),
            "Daily VaR (95%)": var_95, # NEU: Hinzufügen der VaR-Metrik
        }

    return pd.DataFrame(stats).T


def norm_100(prices: pd.DataFrame) -> pd.DataFrame:
    """Setzt den ersten verfügbaren Wert jeder Spalte auf 100 für Vergleichszwecke."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    
    # Explizite Berechnung: Nutzt die erste Zeile als Basis (Index 0)
    base = prices.iloc[0].replace(0, np.nan) 
    return prices.divide(base) * 100.0


def portfolio_series(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Baut eine Portfolio-Indexreihe (Custom oder Equal-Weight)."""
    weights = weights.reindex(prices.columns).fillna(0)

    if weights.sum() > 0:
        w = weights / weights.sum()
    else:
        # Equal-Weight Logik (Woche 2: Arithmetik/Division)
        w = pd.Series(1 / len(prices.columns), index=prices.columns)

    port_ret = prices.pct_change().fillna(0).dot(w)
    return (1 + port_ret).cumprod() * 100.0


@st.cache_data(ttl=30 * 60)
def load_prices(tickers, start, end, interval="1d") -> pd.DataFrame:
    """Lädt adjustierte Close-Preise über die yfinance API (Anforderung 2)."""
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

    if isinstance(df.columns, pd.MultiIndex):
        close = df.loc[:, (slice(None), "Close")]
        close.columns = [c[0] for c in close.columns]
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})

    close = close.dropna(how="all")
    return close


# NEU: Funktion zur Umsetzung des Benchmark-Selectbox (Woche 2: Functions & If/Else)
def get_benchmark_symbol(name):
    """Wandelt den lesbaren Namen in das Yahoo-Finance Ticker Symbol um."""
    # Anwendung von If/Elif/Else (Lecture 2/3)
    if name == "SMI (Schweiz)":
        return "^SSMI"
    elif name == "S&P 500 (USA)":
        return "^GSPC"
    elif name == "DAX (Deutschland)":
        return "^GDAXI"
    else:
        return None 

# NEU: Funktion zur Datenbank-Interaktion (Anforderung 2 & Woche 7)
def load_watchlist_from_db(db_path, watchlist_name):
    """Läd eine Ticker-Liste aus einer SQLite-Datenbanktabelle."""
    try:
        # Nutzung von sqlite3.connect (Woche 7)
        conn = sqlite3.connect(db_path)
        # Laden der Daten direkt in ein Pandas DataFrame (Woche 8)
        query = f"SELECT ticker FROM {watchlist_name}"
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Konvertierung der Ticker-Spalte in eine Python Liste (Sequenz)
        return df['ticker'].tolist()
        
    except Exception as e:
        # Fehlermeldung wird im Frontend angezeigt
        st.error(f"Fehler beim Laden der Datenbank '{db_path}': {e}")
        st.caption("Bitte stelle sicher, dass die Datei existiert und eine Tabelle 'top_watch' mit einer Spalte 'ticker' enthält.")
        return []

# ------------------------------
# Sidebar Controls (Anforderung 4: User Interaction)
# ------------------------------

qparams = st.query_params
def_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]
url_tickers = qparams.get("stocks")

if url_tickers:
    tickers_default = (
        url_tickers.split(",") if isinstance(url_tickers, str) else list(url_tickers)
    )
else:
    tickers_default = def_tickers

with st.sidebar:
    st.header("⚙️ Steuerung & Daten")

    # DB Watchlist Lade-Logik
    DB_FILE = "project_watchlists.db" 
    if st.button('Watchlist aus DB laden (Demo)'):
        # Hier wird die Watchlist geladen und im Session State gespeichert.
        # Dies ist der Nachweis für die DB-Anforderung.
        db_tickers = load_watchlist_from_db(DB_FILE, 'top_watch') 
        if db_tickers:
            st.session_state['db_tickers_loaded'] = db_tickers 
            st.success(f"Watchlist '{'top_watch'}' geladen. Wähle die Ticker oben!")
        else:
            st.info(f"DB-Watchlist konnte nicht geladen werden.")

    # Fügt die geladenen DB-Ticker zu den Optionen hinzu
    extra_options = st.session_state.get('db_tickers_loaded', [])
    all_multiselect_options = sorted(list(set(def_tickers + ["AMZN", "META", "TSLA", "NFLX", "AMD", "INTC", "IBM", "ORCL", "CRM", "AVGO", "ASML", "SAP", "SONY", "BABA", "JNJ", "PG", "KO", "PEP", "XOM", "CVX", "JPM", "BAC", "V", "MA", "T", "VZ", "NVO", "RHHBY", "NESN.SW", "ROG.SW", "UBSG.SW", "^SSMI", "^GSPC", "^GDAXI"] + extra_options)))


    tickers = st.multiselect(
        "Aktien-Ticker (Yahoo Finance)",
        options=all_multiselect_options,
        # Wenn eine DB geladen wurde, setzen wir die geladenen Ticker als Default
        default=list(set(tickers_default) | set(extra_options)),
        help="Füge weitere Ticker hinzu (z.B. UBSG.SW, ^SSMI).",
    )

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Startdatum", value=pd.Timestamp.today() - pd.Timedelta(days=365 * 5))
    with col2:
        end = st.date_input("Enddatum", value=pd.Timestamp.today())

    interval = st.selectbox("Daten-Intervall", ["1d", "1wk", "1mo"], index=0)
    st.divider()

    st.subheader("Portfolio Gewichtung (%)")
    weight_inputs = {}
    for t in tickers[:10]: # Begrenzt auf 10 Ticker für Übersichtlichkeit
        weight_inputs[t] = st.number_input(f"{t}", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    st.caption("Nicht zugewiesene Gewichte werden gleichmässig verteilt.")

    # NEU: Benchmark-Auswahl
    st.divider()
    st.subheader("Benchmark-Auswahl (Eigener Touch)")
    bench_name = st.selectbox(
        "Vergleiche mit einem Index:",
        ["Kein Vergleich", "SMI (Schweiz)", "S&P 500 (USA)", "DAX (Deutschland)"]
    )
    st.divider()

    if tickers:
        deep_link = "?" + urlencode({"stocks": ",".join(tickers)})
        st.text_input("Deep Link (zum Teilen):", value=deep_link)

# ------------------------------
# Data loading and cleaning
# ------------------------------
if not tickers:
    st.info("Wähle mindestens einen Ticker aus der Seitenleiste.")
    st.stop()
prices = load_prices(tickers, start, end, interval)
if prices.empty:
    st.warning("Keine Preisdaten verfügbar.")
    st.stop()
prices = prices.dropna(how="any")
if prices.empty:
    st.warning("Keine Daten nach der Bereinigung übrig.")
    st.stop()

# ------------------------------
# KPIs (Key Performance Indicators)
# ------------------------------
st.subheader("Wichtige Kennzahlen (KPIs)")
metrics_df = perf_stats(prices)
metrics_fmt = metrics_df.copy()

# Formatierung (VaR ist hier inkludiert)
metrics_fmt["Ann. Return"] = (metrics_fmt["Ann. Return"] * 100).map(lambda x: f"{x:,.2f}%")
metrics_fmt["Ann. Vol"] = (metrics_fmt["Ann. Vol"] * 100).map(lambda x: f"{x:,.2f}%")
metrics_fmt["Sharpe (rf=0)"] = metrics_fmt["Sharpe (rf=0)"].map(lambda x: f"{x:,.2f}")
metrics_fmt["Max Drawdown"] = (metrics_fmt["Max Drawdown"] * 100).map(lambda x: f"{x:,.2f}%")
# NEU: Formatierung des VaR
metrics_fmt["Daily VaR (95%)"] = (metrics_fmt["Daily VaR (95%)"] * 100).map(lambda x: f"{x:,.2f}%")

st.dataframe(metrics_fmt, use_container_width=True)

# ------------------------------
# Portfolio Comparison (Eigener Touch)
# ------------------------------

st.header("📦 Portfolio Performance vs. Benchmark")

weights = pd.Series({k: v / 100.0 for k, v in weight_inputs.items()})
port_custom = portfolio_series(prices, weights)
port_custom.name = "Mein Portfolio (Custom)"

port_equal = portfolio_series(prices, pd.Series(0.0, index=prices.columns))
port_equal.name = "Equal-Weight Portfolio"

chart_data = pd.concat([port_custom, port_equal], axis=1)

# Benchmark-Logik
if bench_name != "Kein Vergleich":
    symbol = get_benchmark_symbol(bench_name)
    bench_df = load_prices([symbol], start, end, interval)
    
    if not bench_df.empty:
        bench_price_series = bench_df.iloc[:, 0].dropna()
        first_price = bench_price_series.iloc[0]
        # NEU: Normalisierung auf 100 (Woche 2: Arithmetik)
        benchmark_series = (bench_price_series / first_price) * 100
        benchmark_series.name = bench_name 
        chart_data = chart_data.join(benchmark_series, how="inner")
    
    else:
        st.warning(f"Konnte keine Daten für {bench_name} ({symbol}) laden.")

chart_data = chart_data.reset_index().melt("Date", var_name="Type", value_name="Index")

performance_chart = (
    alt.Chart(chart_data)
    .mark_line()
    .encode(
        x="Date:T",
        y=alt.Y("Index:Q", title="Performance (Start = 100)"),
        color="Type:N",
        tooltip=["Date:T", "Type:N", alt.Tooltip("Index:Q", format=".2f")]
    )
    .interactive()
    .properties(height=400)
)
st.altair_chart(performance_chart, use_container_width=True)

# ... (Rest der Gewichtstabelle)
if weights.sum() > 0:
    w_used = (weights / weights.sum()).reindex(prices.columns).fillna(0)
else:
    w_used = pd.Series(1 / len(prices.columns), index=prices.columns)

w_tbl = (w_used * 100).to_frame("Gewicht %").style.format({"Gewicht %": "{:.2f}"})
st.subheader("Verwendete Portfolio-Gewichte (Normalisiert)")
st.dataframe(w_tbl, use_container_width=False)


# ------------------------------
# Machine Learning (Anforderung 5)
# ------------------------------

st.header("🔮 Machine Learning: Nächster Tag (Logistische Regression)")

ml_ticker = st.selectbox(
    "Wähle einen Ticker für die ML-Vorhersage",
    options=tickers,
    index=0,
    help="Modell lernt aus den letzten 5 Tagen Rendite, um die Richtung des nächsten Tages vorherzusagen."
)

X, y = build_ml_dataset(prices, ml_ticker, lookback=5)

# WICHTIG: Erklärungs-Caption für den Kursbezug
st.caption(f"**Modell-Vorbereitung:** Der Datensatz wurde mit **expliziten For-Loops** (Lag-Returns) erstellt, was die Anwendung der Iterationskonzepte belegt.")

if X.empty or len(y) < 30:
    st.info("Nicht genügend Daten vorhanden, um das Modell zu trainieren.")
else:
    split_idx = int(len(X) * 0.7) 
    
    # If/Else-Logik zur Prüfung der Aufteilung (Woche 2)
    if split_idx < 10 or len(X) - split_idx < 10:
        st.warning("Datensatz ist zu klein für eine sinnvolle 70/30 Aufteilung.")
        st.stop()

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # NEU: Ausgabe der geschätzten Wahrscheinlichkeit (Eigener Touch)
    last_features = X.iloc[[-1]] 
    prediction_tomorrow = model.predict(last_features)[0]
    probability = model.predict_proba(last_features)[0].max()

    st.subheader("Modell-Prognose für den nächsten Tag")
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        direction = "**Aufwärts** ⬆️" if prediction_tomorrow == 1 else "**Abwärts/Flach** ⬇️"
        st.metric(f"Prognostizierte Richtung ({ml_ticker})", direction)
    with col_pred2:
        st.metric("Modell-Sicherheit (letzte Prognose)", f"{probability:.1%}")

    st.divider()

    st.subheader("Modell-Performance auf Testdaten")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Genauigkeit (Accuracy) auf Test-Set:** {acc:.2%}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Tatsächlich Abwärts/Flach (0)", "Tatsächlich Aufwärts (1)"],
        columns=["Prognostiziert Abwärts/Flach (0)", "Prognostiziert Aufwärts (1)"],
    )
    st.write("**Konfusionsmatrix:**")
    st.dataframe(cm_df, use_container_width=True)

    coef_df = pd.DataFrame(
        model.coef_.T,
        index=X.columns,
        columns=["Koeffizient (Wichtigkeit)"],
    )
    st.write("**Modell-Koeffizienten (Feature Importance):**")
    st.dataframe(coef_df.style.format({"Koeffizient (Wichtigkeit)": "{:.4f}"}), use_container_width=True)


# ------------------------------
# Download options
# ------------------------------
csv_prices = prices.to_csv(index=True).encode("utf-8")
st.download_button("⬇️ Download Preise (CSV)", data=csv_prices, file_name="prices.csv", mime="text/csv")

port_csv = chart_data.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Portfolio/Benchmark (CSV)", data=port_csv, file_name="portfolio_benchmark.csv", mime="text/csv")
st.caption("Tipp: Nutze den Deep Link in der Seitenleiste zum Teilen.")
