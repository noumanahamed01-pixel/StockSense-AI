import pandas as pd
import numpy as np
import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    Returns a cleaned DataFrame.
    """
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean_data(df):
    """
    Clean raw stock data:
    - Drop fully empty rows
    - Forward fill missing values
    - Remove duplicates
    - Fix data types
    """
    # Drop rows where all values are missing
    df.dropna(how='all', inplace=True)

    # Forward fill remaining missing values
    df.ffill(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Fix data types
    price_cols = ['Open', 'High', 'Low', 'Close']
    df[price_cols] = df[price_cols].astype(float)
    df['Volume']   = df['Volume'].astype(int)
    df['Date']     = pd.to_datetime(df['Date'])

    df.reset_index(drop=True, inplace=True)
    return df


def get_clean_stock_data(ticker, start_date, end_date):
    """
    Full pipeline: fetch + clean in one call.
    Usage:
        from src.data_preprocessing import get_clean_stock_data
        df = get_clean_stock_data('RELIANCE.NS', '2020-01-01', '2024-12-31')
    """
    df = fetch_stock_data(ticker, start_date, end_date)
    df = clean_data(df)
    return df