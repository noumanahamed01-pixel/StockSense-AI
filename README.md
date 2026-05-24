📈 StockSense AI
Stock Market Analytics and Trend Prediction System
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F89939?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>
<p align="center">
  <b>BCA Final Year Project &nbsp;|&nbsp; 2025 – 2026</b>
</p>

# #📌 Table of Contents

About the Project
Live Demo
Features
System Architecture
Project Structure
Tech Stack
Installation
How to Run
API Endpoints
Machine Learning
Results
Screenshots
Disclaimer
Author


# # 📖 About the Project
StockSense AI is a full-stack data science project that analyzes historical stock market data for Reliance Industries (RELIANCE.NS) and predicts whether the stock price will go UP or DOWN the next trading day using machine learning.
The project covers the complete data science workflow:
Data Collection → Preprocessing → EDA → Feature Engineering → ML Model → API → Dashboard
It is built with a Streamlit frontend, a Flask REST API backend, and a SQLite database that stores every prediction made.

🚀 Live Demo
To run the project locally:
bash# Terminal 1 — Start Flask API
cd "STOCKSENSE - AI/api"
python api.py

# Terminal 2 — Start Streamlit
cd "STOCKSENSE - AI/app"
streamlit run app.py
Then open http://localhost:8501 in your browser.

# ✨ Features
FeatureDescription📊 Fetches latest available stock market data using Yahoo Finance using yfinance🤖 ML PredictionPredicts next-day price direction (UP or DOWN) with confidence %📈 Price Trend ChartClosing price with MA7 and MA21 overlay🕯️ Candlestick ChartLast 90 days of OHLC data📉 RSI ChartWith overbought (>70) and oversold (<30) zones📦 Volume ChartDaily trading volume bar chart🔄 Daily ReturnsReturns over time and distribution histogram📋 Stats TableKey statistics summary for any stock🗃️ Prediction HistoryColor-coded table of all past predictions from SQLite🌐 Any StockWorks with any NSE, BSE, or global ticker symbol

🏗️ System Architecture
┌─────────────────────────────────────────────────────────┐
│                    USER (Browser)                        │
│                 http://localhost:8501                    │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              STREAMLIT DASHBOARD (Frontend)              │
│  • Live charts    • Metrics    • Prediction display      │
│  • History table  • Sidebar controls                     │
└──────────────┬──────────────────────────────────────────┘
               │  HTTP POST /predict
               │  HTTP GET  /history
┌──────────────▼──────────────────────────────────────────┐
│              FLASK REST API (Backend :5000)              │
│  GET  /          → Health check                          │
│  POST /predict   → Run model + save to DB + return JSON  │
│  GET  /history   → Return last 20 predictions            │
│  GET  /stats     → Return UP/DOWN counts                 │
└──────────┬──────────────────┬───────────────────────────┘
           │                  │
┌──────────▼──────┐  ┌────────▼─────────────────────────┐
│  Random Forest  │  │    SQLite Database                │
│  52.14% accuracy│  │    predictions.db                 │
│  stock_model.pkl│  │    (id, ticker, prediction,       │
│  scaler.pkl     │  │     confidence, price, rsi,       │
└─────────────────┘  │     created_at)                   │
                     └───────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│              Yahoo Finance (Data Source)                 │
│              via yfinance Python library                 │
└─────────────────────────────────────────────────────────┘

# #📁 Project Structure
STOCKSENSE - AI/
├── .venv/                          ← Python virtual environment
└── STOCKSENSE - AI/                ← Main project folder
    │
    ├── data/
    │   ├── raw_data.csv            ← Raw downloaded stock data
    │   └── processed_data.csv      ← Cleaned + feature engineered data
    │
    ├── notebooks/
    │   ├── 01_data_collection.ipynb
    │   ├── 02_data_cleaning.ipynb
    │   ├── 03_eda_visualization.ipynb
    │   ├── 04_feature_engineering.ipynb
    │   └── 05_model_training.ipynb
    │
    ├── models/
    │   ├── stock_model.pkl         ← Trained Random Forest model
    │   └── scaler.pkl              ← Fitted StandardScaler
    │
    ├── api/
    │   ├── api.py                  ← Flask REST API
    │   ├── database.py             ← SQLite helper functions
    │   └── predictions.db          ← SQLite database file
    │
    ├── app/
    │   └── app.py                  ← Streamlit dashboard
    │
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── feature_engineering.py
    │   └── prediction.py
    │
    ├── report/
    │   └── project_report.docx
    │
    ├── requirements.txt
    └── README.md

# 🛠️ Tech Stack
CategoryTechnologyVersionLanguagePython3.10+Data Collectionyfinance1.2.0Data ProcessingPandas2.3.3Numerical ComputingNumPy2.4.3Static ChartsMatplotlib3.10.8Statistical VizSeaborn0.13.2Interactive ChartsPlotly6.6.0Machine Learningscikit-learn1.8.0Model Savingjoblib1.5.3Web DashboardStreamlit1.55.0REST APIFlask + flask-corsLatestDatabaseSQLite3Built-inAPI Callsrequests2.32.5NotebooksJupyter1.1.1

# ⚙️ Installation
Step 1 — Clone the repository
bashgit clone https://github.com/noumanahamed01-pixel/StockSense-AI.git
cd StockSense-AI
Step 2 — Create a virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
Step 3 — Install all dependencies
bashpip install -r requirements.txt
Step 4 — Verify installation
bashpip list
You should see streamlit, flask, yfinance, scikit-learn in the list.

# ▶️ How to Run

⚠️ Always start Flask API first, then Streamlit.

Terminal 1 — Flask API
bashcd "STOCKSENSE - AI"
..\venv\Scripts\activate
cd api
python api.py
Expected output:
Database initialized!
Starting StockSense AI API...
 * Running on http://127.0.0.1:5000
Terminal 2 — Streamlit Dashboard
bashcd "STOCKSENSE - AI"
..\venv\Scripts\activate
cd app
streamlit run app.py
Expected output:
Local URL:   http://localhost:8501
Network URL: http://10.x.x.x:8501
Open in browser
http://localhost:8501

🔌 API Endpoints
Base URL: http://localhost:5000
MethodEndpointDescriptionGET/Health check — confirms API is runningPOST/predictRun ML model, save to DB, return predictionGET/historyReturn last 20 predictions from databaseGET/statsReturn total UP and DOWN prediction counts
Example — Predict endpoint
bashcurl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"ticker": "RELIANCE.NS"}'
Response:
json{
  "status": "success",
  "ticker": "RELIANCE.NS",
  "prediction": "UP",
  "confidence": 53.1,
  "current_price": 1347.80,
  "rsi": 41.6,
  "signal": "BUY"
}
Example — History endpoint
bashcurl http://localhost:5000/history
Response:
json{
  "status": "success",
  "count": 5,
  "history": [
    {
      "id": 1,
      "ticker": "RELIANCE.NS",
      "prediction": "UP",
      "confidence": 53.1,
      "current_price": 1347.80,
      "rsi": 41.6,
      "created_at": "2025-04-18 10:31:32"
    }
  ]
}

# #🤖 Machine Learning
Dataset
ParameterValueStockReliance Industries (RELIANCE.NS)ExchangeNational Stock Exchange (NSE India)PeriodJanuary 2020 – December 2024Raw Records1,257 trading daysFinal Records699 (after feature engineering)Features12 input features + 1 target
Features Engineered
FeatureFormulaWhat it tells the modelMA77-day rolling mean of CloseShort-term price trendMA2121-day rolling mean of CloseMedium-term price trendDaily_Return(Close - prev Close) / Close × 100% change per dayVolatility7-day rolling std of Daily_ReturnRisk levelPrice_RangeHigh - LowIntraday price swingMomentumClose - Close (10 days ago)Price accelerationRSI14-day Relative Strength IndexOverbought / Oversold
Target Variable
Target = 1  →  Next day price goes UP
Target = 0  →  Next day price goes DOWN
Models Compared
ModelAccuracyNotesLogistic Regression51.43%Baseline modelRandom Forest52.14% 🏆Final model — saved as stock_model.pkl
Top 3 Important Features
RankFeatureImportance1Low Price10.5%2Volatility10.1%3Daily Return9.7%

# 📊 Results
╔══════════════════════════════════════════════╗
║         STOCKSENSE AI — MODEL RESULTS        ║
╠══════════════════════════════════════════════╣
║  Stock          :  RELIANCE.NS               ║
║  Training Data  :  559 records               ║
║  Testing Data   :  140 records               ║
║  Features Used  :  12                        ║
║  LR Accuracy    :  51.43%                    ║
║  RF Accuracy    :  52.14%  ← Winner          ║
║  Model Saved    :  stock_model.pkl           ║
╚══════════════════════════════════════════════╝

Note: 52% accuracy beats random guessing (50%). Stock market prediction is one of the hardest problems in data science. This project focuses on demonstrating the complete ML workflow rather than building a real trading system.


# # 📸 Screenshots

Add screenshots of your running dashboard here after uploading to GitHub.
# Dashboard Home
[Dashboard](Data/dashboard_home.jpeg)

---

# ⚠️ Disclaimer

This project is built for educational purposes only as part of a BCA Final Year Project.
The predictions made by this system are based on historical data and machine learning models.
This is NOT financial advice. Do not make real investment decisions based on this tool.
Always consult a certified financial advisor before investing.


👨‍💻 Author
M Numan Ahamed

🎓 BCA Final Year Student
🏫 Sri Maata Degree College
📧 noumanahamed01@gmail.com
🐙 https://github.com/noumanahamed01-pixel
💼 https://www.linkedin.com/in/m-numan-ahamed-53a937314/

Project Guide: Pruthvi raj Sir (Lecturer)

🙏 Acknowledgements

Yahoo Finance for free stock market data
yfinance Python library
Streamlit for easy web app development
scikit-learn for machine learning tools
Flask for the REST API framework


# 📄 License
This project is licensed under the MIT License — free to use, modify, and share with attribution.

<p align="center">
  Made with ❤️ using Python & Streamlit
  <br><br>
  <b>StockSense AI — BCA Final Year Project 2025-2026</b>
</p>