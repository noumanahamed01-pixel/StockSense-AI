from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.database import init_db, save_prediction, get_history, get_stats

app = Flask(__name__)
CORS(app)  # allows Streamlit to call this API

# ─────────────────────────────────────────
# Load model on startup
# ─────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
model_path  = BASE_DIR / "Models" / "stock_model.pkl"
scaler_path = BASE_DIR / "Models" / "scaler.pkl"

model  = joblib.load(str(model_path))
scaler = joblib.load(str(scaler_path))

# Initialize database
init_db()

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA7', 'MA21', 'Daily_Return',
    'Volatility', 'Price_Range', 'Momentum', 'RSI'
]


# ─────────────────────────────────────────
# Helper — fetch & engineer features
# ─────────────────────────────────────────
def prepare_data(ticker):
    df = yf.download(ticker, period="3mo", auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Feature engineering
    df['MA7']          = df['Close'].rolling(7).mean()
    df['MA21']         = df['Close'].rolling(21).mean()
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Volatility']   = df['Daily_Return'].rolling(7).std()
    df['Price_Range']  = df['High'] - df['Low']
    df['Momentum']     = df['Close'] - df['Close'].shift(10)

    delta    = df['Close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/', methods=['GET'])
def health_check():
    """Check if API is running."""
    return jsonify({
        "status"  : "running",
        "message" : "StockSense AI API is live!",
        "version" : "1.0"
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "ticker": "RELIANCE.NS" }
    Returns: prediction, confidence, current price, RSI
    """
    data   = request.get_json()
    ticker = data.get('ticker', 'RELIANCE.NS').upper()

    try:
        df            = prepare_data(ticker)
        latest        = df[FEATURE_COLS].iloc[-1:]
        latest_scaled = scaler.transform(latest)

        pred       = model.predict(latest_scaled)[0]
        prob       = model.predict_proba(latest_scaled)[0]
        confidence = round(float(max(prob)) * 100, 2)
        prediction = "UP" if pred == 1 else "DOWN"

        current_price = round(float(df['Close'].iloc[-1]), 2)
        rsi           = round(float(df['RSI'].iloc[-1]), 2)

        # Save to database
        save_prediction(ticker, prediction, confidence, current_price, rsi)

        return jsonify({
            "status"        : "success",
            "ticker"        : ticker,
            "prediction"    : prediction,
            "confidence"    : confidence,
            "current_price" : current_price,
            "rsi"           : rsi,
            "signal"        : "BUY" if prediction == "UP" else "SELL"
        })

    except Exception as e:
        return jsonify({
            "status" : "error",
            "message": str(e)
        }), 500


@app.route('/history', methods=['GET'])
def history():
    """
    GET /history
    Returns last 20 predictions from database.
    """
    rows = get_history(limit=20)
    return jsonify({
        "status" : "success",
        "count"  : len(rows),
        "history": rows
    })


@app.route('/stats', methods=['GET'])
def stats():
    """
    GET /stats
    Returns summary stats of all predictions.
    """
    data = get_stats()
    return jsonify({
        "status": "success",
        **data
    })


# ─────────────────────────────────────────
# Run the API
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("Starting StockSense AI API...")
    print("API running at http://localhost:5000")
    app.run(debug=False, port=5000, use_reloader=False)