import sys
from pathlib import Path

# Get absolute path of the project root and add to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
#import streamlit as st
# st.write("Project root:", str(project_root))  # temporary debug line

from Src.feature_Engineering import build_features, FEATURE_COLS
from Src.prediction import load_model_and_scaler, predict_next_day, get_rsi_signal, get_ma_signal

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F3864;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #1F3864;
    }
    .predict-up {
        background: #e6f9f0;
        border: 2px solid #00b050;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #00b050;
    }
    .predict-down {
        background: #fff0f0;
        border: 2px solid #e00000;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #e00000;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1F3864;
        border-bottom: 2px solid #1F3864;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_features(df):
    df = df.copy()
    df['MA7']          = df['Close'].rolling(7).mean()
    df['MA21']         = df['Close'].rolling(21).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Volatility']   = df['Daily_Return'].rolling(7).std()
    df['Price_Range']  = df['High'] - df['Low']
    df['Momentum']     = df['Close'] - df['Close'].shift(10)

    # RSI
    delta     = df['Close'].diff()
    gain      = delta.where(delta > 0, 0)
    loss      = -delta.where(delta < 0, 0)
    avg_gain  = gain.rolling(14).mean()
    avg_loss  = loss.rolling(14).mean()
    rs        = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df


def make_prediction(df, model, scaler):
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'MA7', 'MA21', 'Daily_Return',
                    'Volatility', 'Price_Range', 'Momentum', 'RSI']
    latest = df[feature_cols].iloc[-1:]
    try:
        latest_scaled = scaler.transform(latest)
        pred     = model.predict(latest_scaled)[0]
        prob     = model.predict_proba(latest_scaled)[0]
        confidence = max(prob) * 100
    except Exception:
        pred       = model.predict(latest)[0]
        prob       = model.predict_proba(latest)[0]
        confidence = max(prob) * 100
    return pred, confidence


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    from pathlib import Path
    base_dir   = Path(__file__).resolve().parent.parent
    model_path = base_dir / "Models" / "stock_model.pkl"
    scaler_path = base_dir / "Models" / "scaler.pkl"
    
    model  = joblib.load(str(model_path))
    scaler = joblib.load(str(scaler_path))
    return model, scaler


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
        [data-testid="stImage"] img {
            border-radius: 50% !important;
            border: 3px solid #1F3864 !important;
            width: 80px !important;
            height: 80px !important;
            object-fit: cover !important;
        }
    </style>
    """, unsafe_allow_html=True)
    # Resolve logo path relative to project root (works when running via `streamlit run`)
    logo_path = project_root / "stocksenseimg.png"
    if not logo_path.exists():
        # Fallback: look next to this file (App folder)
        logo_path = Path(__file__).resolve().parent / "stocksenseimg.png"
    if logo_path.exists():
        st.image(str(logo_path), width=80)
    st.title("StockSense AI")
    st.markdown("---")

    st.subheader("⚙️ Settings")

    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="RELIANCE.NS",
        help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, AAPL, TSLA"
    ).upper()

    period_options = {
        "6 Months"  : 180,
        "1 Year"    : 365,
        "2 Years"   : 730,
        "5 Years"   : 1825
    }
    selected_period = st.selectbox("Select Time Period", list(period_options.keys()), index=1)
    days = period_options[selected_period]

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days)

    st.markdown("---")
    st.subheader("📌 Popular Indian Stocks")
    quick_stocks = {
        "Reliance"  : "RELIANCE.NS",
        "TCS"       : "TCS.NS",
        "Infosys"   : "INFY.NS",
        "HDFC Bank" : "HDFCBANK.NS",
        "Wipro"     : "WIPRO.NS"
    }
    for name, sym in quick_stocks.items():
        if st.button(f"📊 {name}", width='stretch'):
            ticker = sym

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ─────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────
st.markdown('<p class="main-title">📈 StockSense AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Stock Market Analytics & Trend Prediction System</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker}..."):
    df_raw = load_stock_data(ticker, start_date.strftime('%Y-%m-%d'),
                             end_date.strftime('%Y-%m-%d'))

if df_raw.empty:
    st.error("❌ No data found. Please check the ticker symbol and try again.")
    st.stop()

df = add_features(df_raw.copy())

# ─────────────────────────────────────────
# KEY METRICS ROW
# ─────────────────────────────────────────
st.markdown('<p class="section-header">📊 Key Metrics</p>', unsafe_allow_html=True)

latest       = df.iloc[-1]
prev         = df.iloc[-2]
price_change = latest['Close'] - prev['Close']
pct_change   = (price_change / prev['Close']) * 100
color        = "🟢" if price_change >= 0 else "🔴"

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Current Price",  f"₹{latest['Close']:.2f}",
            f"{price_change:+.2f} ({pct_change:+.2f}%)")
col2.metric("Day High",       f"₹{latest['High']:.2f}")
col3.metric("Day Low",        f"₹{latest['Low']:.2f}")
col4.metric("Volume",         f"{int(latest['Volume']):,}")
col5.metric("RSI",            f"{latest['RSI']:.1f}",
            "Overbought" if latest['RSI'] > 70 else
            "Oversold"   if latest['RSI'] < 30 else "Neutral")
col6.metric("Volatility",     f"{latest['Volatility']:.2f}%")

st.markdown("---")

# ─────────────────────────────────────────
# PREDICTION BOX
# ─────────────────────────────────────────
st.markdown('<p class="section-header">🤖 ML Prediction</p>', unsafe_allow_html=True)

try:
    model, scaler = load_model()
    prediction, confidence = make_prediction(df, model, scaler)

    pred_col, info_col = st.columns([1, 2])

    with pred_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="predict-up">
                📈 PRICE LIKELY TO GO UP<br>
                <span style="font-size:1rem; font-weight:400">
                    Confidence: {confidence:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="predict-down">
                📉 PRICE LIKELY TO GO DOWN<br>
                <span style="font-size:1rem; font-weight:400">
                    Confidence: {confidence:.1f}%
                </span>
            </div>""", unsafe_allow_html=True)

    with info_col:
        st.info("""
        **How this prediction works:**
        - Trained on 5 years of Reliance Industries historical data
        - Uses Random Forest with 12 technical features
        - Features include: RSI, Moving Averages, Volatility, Momentum
        - Predicts next trading day's price direction (Up or Down)

        ⚠️ This is for educational purposes only — not financial advice!
        """)

except Exception as e:
    st.warning(f"Model not loaded: {e}. Showing dashboard without prediction.")

st.markdown("---")

# ─────────────────────────────────────────
# CHART 1 — PRICE + MOVING AVERAGES
# ─────────────────────────────────────────
st.markdown('<p class="section-header">📉 Price Trend & Moving Averages</p>',
            unsafe_allow_html=True)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
    name='Close Price', line=dict(color='royalblue', width=1.5)))
fig1.add_trace(go.Scatter(x=df['Date'], y=df['MA7'],
    name='MA 7',  line=dict(color='orange', width=1.5, dash='dot')))
fig1.add_trace(go.Scatter(x=df['Date'], y=df['MA21'],
    name='MA 21', line=dict(color='red',    width=1.5, dash='dash')))
fig1.update_layout(
    title=f"{ticker} – Closing Price with Moving Averages",
    xaxis_title="Date", yaxis_title="Price (INR ₹)",
    hovermode='x unified', height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig1, width='stretch')

# ─────────────────────────────────────────
# CHART 2 — CANDLESTICK
# ─────────────────────────────────────────
st.markdown('<p class="section-header">🕯️ Candlestick Chart (Last 90 Days)</p>',
            unsafe_allow_html=True)

df_candle = df.tail(90)
fig2 = go.Figure(data=[go.Candlestick(
    x=df_candle['Date'],
    open=df_candle['Open'], high=df_candle['High'],
    low=df_candle['Low'],   close=df_candle['Close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])
fig2.update_layout(
    title=f"{ticker} – Candlestick Chart (Last 90 Trading Days)",
    xaxis_title="Date", yaxis_title="Price (INR ₹)",
    xaxis_rangeslider_visible=False, height=420
)
st.plotly_chart(fig2, width='stretch')

# ─────────────────────────────────────────
# CHART 3 — RSI + VOLUME (side by side)
# ─────────────────────────────────────────
st.markdown('<p class="section-header">📊 RSI & Volume Analysis</p>',
            unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['RSI'],
        name='RSI', line=dict(color='purple', width=1.5),
        fill='tozeroy', fillcolor='rgba(128,0,128,0.1)'))
    fig3.add_hline(y=70, line_dash="dash", line_color="red",
                   annotation_text="Overbought (70)")
    fig3.add_hline(y=30, line_dash="dash", line_color="green",
                   annotation_text="Oversold (30)")
    fig3.update_layout(title="RSI (14-Day)",
                       yaxis_title="RSI Value",
                       height=350)
    st.plotly_chart(fig3, width='stretch')

with chart_col2:
    colors = ['green' if r >= 0 else 'red'
              for r in df['Daily_Return']]
    fig4 = go.Figure(go.Bar(x=df['Date'], y=df['Volume'],
        marker_color='steelblue', opacity=0.7, name='Volume'))
    fig4.update_layout(title="Trading Volume",
                       yaxis_title="Volume (Shares)",
                       height=350)
    st.plotly_chart(fig4, width='stretch')

# ─────────────────────────────────────────
# CHART 4 — DAILY RETURNS DISTRIBUTION
# ─────────────────────────────────────────
st.markdown('<p class="section-header">📈 Daily Returns Analysis</p>',
            unsafe_allow_html=True)

ret_col1, ret_col2 = st.columns(2)

with ret_col1:
    fig5 = go.Figure()
    colors_ret = ['green' if v >= 0 else 'red' for v in df['Daily_Return']]
    fig5.add_trace(go.Bar(x=df['Date'], y=df['Daily_Return'],
        marker_color=colors_ret, opacity=0.7, name='Daily Return'))
    fig5.add_hline(y=0, line_color='black', line_width=1)
    fig5.update_layout(title="Daily Returns (%)",
                       yaxis_title="Return (%)", height=350)
    st.plotly_chart(fig5, width='stretch')

with ret_col2:
    fig6 = px.histogram(df, x='Daily_Return', nbins=50,
        color_discrete_sequence=['steelblue'],
        title="Distribution of Daily Returns",
        labels={'Daily_Return': 'Daily Return (%)'})
    fig6.update_layout(height=350)
    st.plotly_chart(fig6, width='stretch')

# ─────────────────────────────────────────
# STATS TABLE
# ─────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">📋 Statistical Summary</p>',
            unsafe_allow_html=True)

stats = pd.DataFrame({
    'Metric': [
        'Starting Price', 'Current Price', 'All-Time High (in period)',
        'All-Time Low (in period)', 'Average Close Price',
        'Average Daily Return', 'Max Single Day Gain',
        'Max Single Day Loss', 'Average Volume'
    ],
    'Value': [
        f"₹{df['Close'].iloc[0]:.2f}",
        f"₹{df['Close'].iloc[-1]:.2f}",
        f"₹{df['Close'].max():.2f}",
        f"₹{df['Close'].min():.2f}",
        f"₹{df['Close'].mean():.2f}",
        f"{df['Daily_Return'].mean():.4f}%",
        f"{df['Daily_Return'].max():.2f}%",
        f"{df['Daily_Return'].min():.2f}%",
        f"{int(df['Volume'].mean()):,} shares"
    ]
})
st.dataframe(stats, width='stretch', hide_index=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem'>
    StockSense AI — BCA Final Year Project &nbsp;|&nbsp;
    Built with Python & Streamlit &nbsp;|&nbsp;
    Data from Yahoo Finance &nbsp;|&nbsp;
    ⚠️ Not financial advice
</div>
""", unsafe_allow_html=True)
import requests as req

# ─────────────────────────────────────────
# PREDICTION HISTORY (from Flask API)
# ─────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-header">🕘 Prediction History</p>',
            unsafe_allow_html=True)

try:
    response = req.get("http://localhost:5000/history", timeout=3)
    if response.status_code == 200:
        history_data = response.json().get("history", [])

        if history_data:
            hist_df = pd.DataFrame(history_data)
            hist_df = hist_df[['created_at', 'ticker',
                                'prediction', 'confidence',
                                'current_price', 'rsi']]
            hist_df.columns = ['Date & Time', 'Ticker',
                                'Prediction', 'Confidence (%)',
                                'Price (₹)', 'RSI']

            # Color code UP/DOWN with better contrast
            def highlight(val):
                if val == 'UP':
                    return 'background-color: #00b050; color: white; font-weight: bold; text-align: center'
                else:
                    return 'background-color: #e00000; color: white; font-weight: bold; text-align: center'

            st.dataframe(
                hist_df.style.applymap(
                    highlight, subset=['Prediction']
                ),
                width='stretch',
                hide_index=True
            )

            # Stats
            stats_res = req.get("http://localhost:5000/stats", timeout=3).json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Predictions", stats_res['total_predictions'])
            c2.metric("UP Predictions",    stats_res['up_predictions'])
            c3.metric("DOWN Predictions",  stats_res['down_predictions'])
        else:
            st.info("No prediction history yet. Make a prediction above!")

except Exception:
    st.info("Start the Flask API to see prediction history.")