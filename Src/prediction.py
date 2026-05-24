import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from Src.feature_Engineering import FEATURE_COLS


def load_model_and_scaler():
    """
    Load the saved ML model and scaler from models/ folder.
    Returns (model, scaler)
    """
    base_dir    = Path(__file__).resolve().parent.parent
    model_path  = base_dir / "Models" / "stock_model.pkl"
    scaler_path = base_dir / "Models" / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_next_day(df, model, scaler):
    """
    Predict next day's price direction using the latest row of data.

    Returns:
        prediction  : 1 (UP) or 0 (DOWN)
        confidence  : confidence % of the prediction
        signal      : 'BUY' or 'SELL' string
    """
    latest = df[FEATURE_COLS].iloc[-1:]

    try:
        latest_scaled = scaler.transform(latest)
        prediction    = model.predict(latest_scaled)[0]
        probability   = model.predict_proba(latest_scaled)[0]
    except Exception:
        # If scaler doesn't apply (e.g. Random Forest without scaling)
        prediction  = model.predict(latest)[0]
        probability = model.predict_proba(latest)[0]

    confidence = round(max(probability) * 100, 2)
    signal     = "BUY 📈"  if prediction == 1 else "SELL 📉"

    return int(prediction), confidence, signal


def get_rsi_signal(rsi_value):
    """
    Interpret RSI value into a human-readable signal.
    """
    if rsi_value > 70:
        return "Overbought ⚠️ — Price may fall soon"
    elif rsi_value < 30:
        return "Oversold ✅ — Price may rise soon"
    else:
        return "Neutral — No strong RSI signal"


def get_ma_signal(ma7, ma21):
    """
    Interpret moving average crossover signal.
    """
    if ma7 > ma21:
        return "Bullish 📈 — MA7 above MA21 (uptrend)"
    elif ma7 < ma21:
        return "Bearish 📉 — MA7 below MA21 (downtrend)"
    else:
        return "Neutral — MA7 equals MA21"