import sqlite3
from pathlib import Path
from datetime import datetime

# Database file path
DB_PATH = Path(__file__).resolve().parent / "predictions.db"


def init_db():
    """
    Create the predictions table if it doesn't exist.
    Call this once when the API starts.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT    NOT NULL,
            prediction   TEXT    NOT NULL,
            confidence   REAL    NOT NULL,
            current_price REAL   NOT NULL,
            rsi          REAL,
            created_at   TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized!")


def save_prediction(ticker, prediction, confidence, current_price, rsi):
    """
    Save a prediction to the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
            (ticker, prediction, confidence, current_price, rsi, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        prediction,
        round(confidence, 2),
        round(current_price, 2),
        round(rsi, 2) if rsi else None,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_history(limit=20):
    """
    Fetch last N predictions from the database.
    Returns a list of dicts.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # returns dict-like rows
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_stats():
    """
    Get summary stats from prediction history.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'UP'")
    up_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'DOWN'")
    down_count = cursor.fetchone()[0]

    conn.close()

    return {
        "total_predictions" : total,
        "up_predictions"    : up_count,
        "down_predictions"  : down_count
    }