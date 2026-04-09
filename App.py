"""
Stock Price Predictor Pro — Streamlit Dashboard
Run with:  streamlit run app.py
Requires:  pip install streamlit yfinance plotly tensorflow scikit-learn pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#  Page Config 
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS 
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #4CAF50;
        margin: 4px 0;
    }
    .metric-card.red  { border-left-color: #f44336; }
    .metric-card.blue { border-left-color: #2196F3; }
    .metric-card.gold { border-left-color: #FFC107; }
    .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #fff; }
    .signal-buy  { color: #00e676; font-weight: bold; font-size: 18px; }
    .signal-sell { color: #ff1744; font-weight: bold; font-size: 18px; }
    .signal-hold { color: #90a4ae; font-weight: bold; font-size: 18px; }
    h1 { background: linear-gradient(90deg, #4CAF50, #2196F3);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


#  Utility Functions 
@st.cache_data(ttl=300)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26, adjust=False).mean()

    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * std20
    df["BB_Lower"] = df["SMA_20"] - 2 * std20
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift()).abs()
    lpc = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    df["Return_1d"]    = df["Close"].pct_change(1)
    df["Return_5d"]    = df["Close"].pct_change(5)
    df["Volatility"]   = df["Return_1d"].rolling(20).std()
    df["Volume_SMA"]   = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rsi_buy   = df["RSI"] < 35
    rsi_sell  = df["RSI"] > 65
    macd_up   = (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
    macd_dn   = (df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))
    bb_buy    = df["Close"] < df["BB_Lower"]
    bb_sell   = df["Close"] > df["BB_Upper"]

    buy_sc  = rsi_buy.astype(int) + macd_up.astype(int) + bb_buy.astype(int)
    sell_sc = rsi_sell.astype(int) + macd_dn.astype(int) + bb_sell.astype(int)

    df["Signal"] = 0
    df.loc[buy_sc  >= 2, "Signal"] =  1
    df.loc[sell_sc >= 2, "Signal"] = -1
    return df


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "BB_Upper", "BB_Lower", "BB_Width",
    "MACD", "MACD_Signal", "MACD_Hist",
    "RSI", "ATR",
    "Return_1d", "Return_5d", "Volatility", "Volume_Ratio"
]


def make_sequences(X_sc, y_sc, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X_sc)):
        Xs.append(X_sc[i - lookback:i])
        ys.append(y_sc[i])
    return np.array(Xs), np.array(ys)


def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def forecast_lstm(model, last_seq, scaler_y, lookback, n_days):
    preds, seq = [], last_seq.copy()
    close_idx = FEATURE_COLS.index("Close")
    for _ in range(n_days):
        inp = seq[-lookback:].reshape(1, lookback, -1)
        ps  = model.predict(inp, verbose=0)[0, 0]
        pp  = scaler_y.inverse_transform([[ps]])[0, 0]
        preds.append(pp)
        next_row = seq[-1].copy()
        next_row[close_idx] = ps
        seq = np.vstack([seq, next_row])
    return preds


# Sidebar 

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stocks-growth.png", width=60)
    st.title(" Configuration")

    ticker  = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    period  = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1)
    lookback = st.slider("LSTM Lookback (days)", 30, 120, 60, step=10)
    forecast_days = st.slider("Forecast Horizon (days)", 3, 30, 7)
    train_model = st.button(" Train Models", type="primary", use_container_width=True)

    st.divider()
    st.caption(" Charts show last 180 trading days")
    st.caption(" For educational use only — not financial advice")


# Main Layout 

st.title(" Stock Price Predictor Pro")
st.caption("Live data · LSTM deep learning · Technical analysis · Buy/Sell signals")

# Load & process data 

with st.spinner(f" Loading {ticker} data…"):
    try:
        raw_df = load_data(ticker, period)
        if raw_df.empty:
            st.error("No data returned. Check the ticker symbol.")
            st.stop()
        df = add_indicators(raw_df)
        df = generate_signals(df)
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

# Key Metrics Strip

latest = df.iloc[-1]
prev   = df.iloc[-2]
chg    = latest["Close"] - prev["Close"]
chg_pct = chg / prev["Close"] * 100
signal_map = {1: ("🟢 BUY",  "signal-buy"), -1: ("🔴 SELL", "signal-sell"), 0: ("⚪ HOLD", "signal-hold")}
sig_text, sig_class = signal_map[int(latest["Signal"])]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Close Price", f"${latest['Close']:.2f}", f"{chg:+.2f} ({chg_pct:+.1f}%)")
with c2:
    st.metric("RSI (14)", f"{latest['RSI']:.1f}")
with c3:
    st.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")
with c4:
    st.metric("ATR (14)", f"${latest['ATR']:.2f}")
with c5:
    st.markdown(f"**Today's Signal**<br><span class='{sig_class}'>{sig_text}</span>",
                unsafe_allow_html=True)

st.divider()

#  Tabs

tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Chart", "🤖 ML Models", "🔮 Forecast", "📋 Data & Signals"])

# 
# TAB 1  Technical Analysis Chart
# 
with tab1:
    st.subheader(f"{ticker}  Technical Analysis")

    plot_df = df.tail(180).copy()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.21, 0.21],
        vertical_spacing=0.03,
        subplot_titles=("Price + Bollinger Bands + Signals", "MACD", "RSI")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],   close=plot_df["Close"],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(100,149,237,0.6)", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], name="BB Lower",
                             line=dict(color="rgba(100,149,237,0.6)", dash="dot", width=1),
                             fill="tonexty", fillcolor="rgba(100,149,237,0.06)"), row=1, col=1)

    # MAs
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_20"], name="SMA 20",
                             line=dict(color="#FFC107", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["SMA_50"], name="SMA 50",
                             line=dict(color="#9C27B0", width=1.5)), row=1, col=1)

    # Signals
    buys  = plot_df[plot_df["Signal"] ==  1]
    sells = plot_df[plot_df["Signal"] == -1]
    fig.add_trace(go.Scatter(x=buys.index,  y=buys["Low"]  * 0.98, mode="markers",
                             name="Buy",  marker=dict(symbol="triangle-up",   size=11, color="#00e676")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.02, mode="markers",
                             name="Sell", marker=dict(symbol="triangle-down", size=11, color="#ff1744")), row=1, col=1)

    # MACD
    macd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in plot_df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_Hist"], name="Hist",
                         marker_color=macd_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"],       name="MACD",
                             line=dict(color="#2196F3", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_Signal"], name="Signal",
                             line=dict(color="#FF9800", width=1.5)), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI",
                             line=dict(color="#CE93D8", width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#66BB6A", row=3, col=1)

    fig.update_layout(template="plotly_dark", height=800,
                      xaxis_rangeslider_visible=False, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)


# TAB 2  ML Models
with tab2:
    st.subheader("🤖 ML Model Training & Comparison")

    if not train_model:
        st.info("👈 Click **Train Models** in the sidebar to train LSTM, Random Forest & Linear Regression.")
    else:
        X = df[FEATURE_COLS].values
        y = df["Target"].values
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        #  Linear Regression 
        with st.spinner("Training Linear Regression…"):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)

        # Random Forest 
        with st.spinner("Training Random Forest…"):
            rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                       random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_preds = rf.predict(X_test)

        #  LSTM 
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_sc = scaler_X.fit_transform(X)
        y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_seq, y_seq = make_sequences(X_sc, y_sc, lookback)
        split_seq = int(len(X_seq) * 0.80)
        X_tr, X_ts = X_seq[:split_seq], X_seq[split_seq:]
        y_tr, y_ts = y_seq[:split_seq], y_seq[split_seq:]

        with st.spinner("Training LSTM (this may take a minute)…"):
            lstm_model = build_lstm((lookback, len(FEATURE_COLS)))
            early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            history = lstm_model.fit(
                X_tr, y_tr,
                epochs=60, batch_size=32,
                validation_split=0.1,
                callbacks=[early],
                verbose=0
            )
            lstm_preds_sc = lstm_model.predict(X_ts, verbose=0).flatten()
            lstm_preds  = scaler_y.inverse_transform(lstm_preds_sc.reshape(-1, 1)).flatten()
            lstm_actual = scaler_y.inverse_transform(y_ts.reshape(-1, 1)).flatten()

        # Store in session state for Forecast tab
        st.session_state["lstm_model"]  = lstm_model
        st.session_state["scaler_X"]    = scaler_X
        st.session_state["scaler_y"]    = scaler_y
        st.session_state["X_sc_last"]   = X_sc[-lookback:]

        st.success(" All models trained!")

        # Metrics
        def mk_metrics(y_true, y_pred):
            return {
                "MAE":  mean_absolute_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "R²":   r2_score(y_true, y_pred),
            }

        mets = {
            "Linear Regression": mk_metrics(y_test, lr_preds),
            "Random Forest":     mk_metrics(y_test, rf_preds),
            "LSTM":              mk_metrics(lstm_actual, lstm_preds),
        }
        metrics_df = pd.DataFrame(mets).T.reset_index().rename(columns={"index": "Model"})

        col_m1, col_m2, col_m3 = st.columns(3)
        best_mae  = metrics_df.loc[metrics_df["MAE"].idxmin(), "Model"]
        best_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
        best_r2   = metrics_df.loc[metrics_df["R²"].idxmax(), "Model"]
        col_m1.metric("Best MAE",  best_mae,  help="Lower is better")
        col_m2.metric("Best RMSE", best_rmse, help="Lower is better")
        col_m3.metric("Best R²",   best_r2,   help="Higher is better")

        st.dataframe(
            metrics_df.style.format({"MAE": "${:.2f}", "RMSE": "${:.2f}", "R²": "{:.4f}"}),
            use_container_width=True, hide_index=True
        )

        #  Comparison Chart 
        test_dates      = df.index[split_idx:].tolist()
        lstm_test_dates = df.index[split_idx + lookback:].tolist()

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=test_dates, y=y_test, name="Actual",
                                     line=dict(color="white", width=2)))
        fig_cmp.add_trace(go.Scatter(x=test_dates, y=rf_preds, name="Random Forest",
                                     line=dict(color="#FF9800", width=1.5)))
        fig_cmp.add_trace(go.Scatter(x=test_dates, y=lr_preds, name="Linear Reg.",
                                     line=dict(color="#2196F3", width=1.5)))
        fig_cmp.add_trace(go.Scatter(x=lstm_test_dates, y=lstm_preds, name="LSTM",
                                     line=dict(color="#00e676", width=2)))
        fig_cmp.update_layout(
            title="Predictions vs Actual (Test Set)",
            template="plotly_dark", height=420,
            xaxis_title="Date", yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        #  Training Loss
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history.history["loss"], name="Train Loss",
                                      line=dict(color="#2196F3")))
        fig_loss.add_trace(go.Scatter(y=history.history["val_loss"], name="Val Loss",
                                      line=dict(color="#FF5722")))
        fig_loss.update_layout(title="LSTM Training Loss", template="plotly_dark",
                               xaxis_title="Epoch", yaxis_title="Huber Loss", height=350)
        st.plotly_chart(fig_loss, use_container_width=True)

        #  Feature Importance 
        feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": rf.feature_importances_})
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)
        fig_imp = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
            marker=dict(color=feat_df["Importance"], colorscale="Viridis")
        ))
        fig_imp.update_layout(title="RF Feature Importances (Top 15)",
                              template="plotly_dark", height=450)
        st.plotly_chart(fig_imp, use_container_width=True)


# TAB 3  Forecast
with tab3:
    st.subheader(f"🔮 {forecast_days}-Day LSTM Price Forecast")

    if "lstm_model" not in st.session_state:
        st.info("👈 Train the models first (click **Train Models** in the sidebar).")
    else:
        lstm_model_fc = st.session_state["lstm_model"]
        scaler_y_fc   = st.session_state["scaler_y"]
        last_seq_fc   = st.session_state["X_sc_last"]

        with st.spinner("Generating forecast…"):
            future_prices = forecast_lstm(
                lstm_model_fc, last_seq_fc, scaler_y_fc, lookback, forecast_days
            )
        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days
        )

        hist_tail = df["Close"].tail(90)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail.values,
            mode="lines", name="Historical",
            line=dict(color="#90caf9", width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode="lines+markers", name="Forecast",
            line=dict(color="#ffd54f", width=2, dash="dash"),
            marker=dict(size=8, color="#ffd54f")
        ))
        upper = [p * 1.02 for p in future_prices]
        lower = [p * 0.98 for p in future_prices]
        fig_fc.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(255,213,79,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±2% Band"
        ))

        # Vertical line at forecast start
        fig_fc.add_vline(x=str(df.index[-1].date()),
                         line_dash="dash", line_color="gray", opacity=0.6)

        fig_fc.update_layout(
            title=f"{ticker} — {forecast_days}-Day Ahead Forecast",
            template="plotly_dark", height=480,
            xaxis_title="Date", yaxis_title="Price (USD)"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table
        forecast_df = pd.DataFrame({
            "Date":           [d.date() for d in future_dates],
            "Predicted Price": [f"${p:.2f}" for p in future_prices],
            "Change vs Today": [f"{(p - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100:+.1f}%" for p in future_prices],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        overall_direction = " Bullish" if future_prices[-1] > df["Close"].iloc[-1] else " Bearish"
        overall_pct = (future_prices[-1] - df["Close"].iloc[-1]) / df["Close"].iloc[-1] * 100
        st.markdown(f"**{forecast_days}-day outlook: {overall_direction} ({overall_pct:+.1f}%)**")


# TAB 4 Data & Signals

with tab4:
    st.subheader(" Recent Data & Signal History")

    sig_col = st.columns(3)
    sig_col[0].metric("🟢 Buy signals",  int((df["Signal"] ==  1).sum()))
    sig_col[1].metric("🔴 Sell signals", int((df["Signal"] == -1).sum()))
    sig_col[2].metric("⚪ Hold signals", int((df["Signal"] ==  0).sum()))

    display_df = df[["Open", "High", "Low", "Close", "Volume",
                     "SMA_20", "SMA_50", "RSI", "MACD", "ATR", "Signal"]].tail(50).copy()
    display_df.index = display_df.index.date

    def style_signal(val):
        if val ==  1: return "color: #00e676; font-weight: bold"
        if val == -1: return "color: #ff1744; font-weight: bold"
        return "color: #90a4ae"

    styled = display_df.style \
        .format({c: "${:.2f}" for c in ["Open","High","Low","Close","SMA_20","SMA_50","MACD","ATR"]}) \
        .format({"Volume": "{:,.0f}", "RSI": "{:.1f}"}) \
        .applymap(style_signal, subset=["Signal"])

    st.dataframe(styled, use_container_width=True, height=500)

    csv = display_df.to_csv().encode("utf-8")
    st.download_button("⬇ Download CSV", csv, f"{ticker}_signals.csv", "text/csv")
