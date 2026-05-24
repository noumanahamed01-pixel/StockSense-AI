import pandas as pd
import numpy as np


def add_moving_averages(df):
    """Add 7-day and 21-day Simple Moving Averages."""
    df['MA7']  = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    return df


def add_daily_return_and_volatility(df):
    """Add daily percentage return and 7-day rolling volatility."""
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Volatility']   = df['Daily_Return'].rolling(window=7).std()
    return df


def add_price_range_and_momentum(df):
    """Add price range (High - Low) and 10-day momentum."""
    df['Price_Range'] = df['High'] - df['Low']
    df['Momentum']    = df['Close'] - df['Close'].shift(10)
    return df


def add_rsi(df, period=14):
    """
    Add Relative Strength Index (RSI).
    RSI > 70 = Overbought (price may fall)
    RSI < 30 = Oversold  (price may rise)
    """
    delta    = df['Close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs       = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_target(df):
    """
    Add binary target column:
    1 = next day price goes UP
    0 = next day price goes DOWN
    """
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df


def build_features(df):
    """
    Full feature engineering pipeline.
    Applies all features and drops NaN rows.
    Usage:
        from src.feature_engineering import build_features
        df = build_features(df)
    """
    df = add_moving_averages(df)
    df = add_daily_return_and_volatility(df)
    df = add_price_range_and_momentum(df)
    df = add_rsi(df)
    df = add_target(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# List of feature columns used for model training
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA7', 'MA21', 'Daily_Return',
    'Volatility', 'Price_Range', 'Momentum', 'RSI'
]