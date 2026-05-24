# 📈 StockSense AI
### Stock Market Analytics and Trend Prediction System
> **BCA Final Year Project** — A data science project that analyzes stock market data and predicts short-term price direction (up or down) using machine learning.

---

## 🚀 Live Demo

Run locally with:
```bash
cd app
streamlit run app.py
```

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Dashboard Preview](#-dashboard-preview)
- [ML Models Used](#-ml-models-used)
- [Results](#-results)
- [Project Workflow](#-project-workflow)
- [Disclaimer](#-disclaimer)

---

## 📖 About the Project

StockSense AI is a machine learning project that analyzes stock market data and predicts whether the next day’s price will go up or down.

In this project, I collected historical stock data from Yahoo Finance, cleaned it, and created features like moving averages, RSI, and daily returns. Then I trained machine learning models such as Logistic Regression and Random Forest to make predictions.

Finally, I built a simple web application using Streamlit where users can view stock trends and predictions in an interactive way.

---

## ✨ Features

- 📊 **Live stock data** fetched directly from Yahoo Finance
- 📉 **Interactive charts** — Closing price, Moving Averages, Candlestick, RSI, Volume
- 🤖 **ML Prediction**— Predicts whether the next day's price will go up or down
- 📈 **Technical Indicators** — MA7, MA21, RSI, Daily Return, Volatility, Momentum
- 🔍 **Statistical Summary** — Key metrics for the selected stock
- 🌐 **Any stock supported** — just enter the ticker symbol (e.g. TCS.NS, INFY.NS, AAPL)
- 📱 **Responsive UI** — works on desktop and mobile browsers

---

## 📁 Project Structure

```
StockSense-AI/
│
├── data/
│   ├── raw_data.csv                  # Raw data from Yahoo Finance
│   └── processed_data.csv            # Cleaned + feature engineered data
│
├── notebooks/
│   ├── 01_data_collection.ipynb      # Phase 1 - Fetch stock data
│   ├── 02_data_cleaning.ipynb        # Phase 2 - Preprocess data
│   ├── 03_eda_visualization.ipynb    # Phase 3 - EDA & charts
│   ├── 04_feature_engineering.ipynb  # Phase 4 - Create features
│   └── 05_model_training.ipynb       # Phase 5 - Train ML models
│
├── models/
│   ├── stock_model.pkl               # Trained Random Forest model
│   └── scaler.pkl                    # StandardScaler for features
│
├── app/
│   └── app.py                        # Streamlit dashboard
│
├── src/
│   ├── data_preprocessing.py         # Reusable data functions
│   ├── feature_engineering.py        # Feature engineering functions
│   └── prediction.py                 # Prediction helper functions
│
├── report/
│   └── project_report.md             # Full project documentation
│
├── requirements.txt                  # All Python dependencies
└── README.md                         # This file
```

---

## 🛠️ Tech Stack

| Category | Tool / Library |
|----------|---------------|
| Language | Python 3.10+ |
| Data Collection | yfinance |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn |
| Web Dashboard | Streamlit |
| Model Saving | joblib |
| IDE | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

---

## ⚙️ Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/yourusername/StockSense-AI.git
cd StockSense-AI
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run the Streamlit Dashboard

```bash
cd app
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Run the Notebooks (optional — to retrain model)

Open Jupyter or VS Code and run notebooks in order:

```
01_data_collection.ipynb      → fetches raw data
02_data_cleaning.ipynb        → cleans data
03_eda_visualization.ipynb    → generates charts
04_feature_engineering.ipynb  → creates features
05_model_training.ipynb       → trains & saves model
```

---

## 🖥️ Dashboard Preview

| Section | Description |
|---------|-------------|
| **Key Metrics** | Live price, Day High/Low, Volume, RSI, Volatility |
| **ML Prediction** | UP/DOWN prediction with confidence % |
| **Price Chart** | Close price + MA7 + MA21 overlay |
| **Candlestick** | Last 90 days OHLC chart |
| **RSI Chart** | With overbought/oversold zones |
| **Volume Chart** | Daily trading volume bar chart |
| **Returns Chart** | Daily returns + distribution histogram |
| **Stats Table** | Full statistical summary of the stock |

---

## 🤖 ML Models Used

### 1. Logistic Regression (Baseline)
- Simple linear classification model
- Fast to train and easy to interpret
- Used as a baseline for comparison

### 2. Random Forest Classifier (Final Model ✅)
- Ensemble of 100 decision trees
- Handles non-linear patterns well
- More robust against overfitting
- Selected as the final model due to better performance

### Features Used for Training

| Feature | Description |
|---------|-------------|
| Open, High, Low, Close | Raw OHLC price data |
| Volume | Number of shares traded |
| MA7 | 7-day Simple Moving Average |
| MA21 | 21-day Simple Moving Average |
| Daily_Return | Percentage price change per day |
| Volatility | 7-day rolling std of daily returns |
| Price_Range | High minus Low for the day |
| Momentum | Close minus Close 10 days ago |
| RSI | 14-day Relative Strength Index |

### Target Variable

```
Target = 1  →  Next day price goes UP
Target = 0  →  Next day price goes DOWN
```

---

## 📊 Results

| Model | Accuracy |
|-------|---------|
| Logistic Regression | 51.43% |
| **Random Forest** | **52.14% 🏆** |

### Top 3 Most Important Features

1. **Low Price** — Strongest predictor of next day movement
2. **Volatility** — Risk level of the stock
3. **Daily Return** — Percentage change in price that day

> **Note:** ~52% accuracy is considered good for stock market prediction. Random guessing gives 50%, so our model consistently beats chance. Professional quant funds target 55–60%.

---

## 📋 Project Workflow

```
Step 1: Data Collection
        ↓
   yfinance API → Yahoo Finance → raw_data.csv

Step 2: Data Preprocessing
        ↓
   Handle missing values → Fix data types → processed_data.csv

Step 3: EDA & Visualization
        ↓
   Price trends → Volume → Correlation heatmap → Candlestick

Step 4: Feature Engineering
        ↓
   MA7, MA21, RSI, Momentum, Volatility, Target column

Step 5: Model Building
        ↓
   Logistic Regression + Random Forest → Evaluate → Save best model

Step 6: Dashboard Deployment
        ↓
   Streamlit web app → Live prediction + Interactive charts
```

---

## ⚠️ Disclaimer

> This project is built for **educational purposes only** as part of a BCA Final Year Project.
> The predictions made by this system are based on historical data patterns and machine learning models.
> **This is NOT financial advice.** Do not make real investment decisions based on this tool.
> Always consult a certified financial advisor before investing.

---

## 👨‍💻 Author

**[Your Full Name]**
BCA Final Year Student
[Your College Name]
[Your Email Address]
[Your GitHub Profile URL]

---

## 📄 License

This project is licensed under the MIT License.

---

*Made with ❤️ using Python & Streamlit*
