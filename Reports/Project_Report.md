StockSense AI – Stock Market Analytics and Trend Prediction System
BCA Final Year Project Report
Project Title: StockSense AI – Stock Market Analytics and Trend Prediction System
Degree       : Bachelor of Computer Applications (BCA)
Academic Year: 2024 – 2025
Student Name : M Numan Ahamed
Roll Number  : U16SD23S0005
Guide name   :[Guide Name]
Institution  : Sri Maata Degree College Affilated by Vijayanagara Sri KrishnaDevaraya University, Hosapete.
Submission Date :[Date]

Table of Contents

1. Introduction
2. Problem Statement
3. Objectives
4. Scope of the Project
5. Literature Review
6. System Requirements
7. System Architecture
8. Methodology
   Phase 1 – Data Collection
   Phase 2 – Data Preprocessing
   Phase 3 – Exploratory Data Analysis
   Phase 4 – Feature Engineering
   Phase 5 – Model Building and Evaluation
   Phase 6 – Dashboard Deployment
9. Dataset Description
10. Implementation
11. Results and Analysis
12. Dashboard Screenshots
13. Applications
14. Limitations
15. Future Scope
16. Conclusion
17. References


1. Introduction

The stock market changes very quickly and is difficult to predict. Many people invest in stocks, but they often rely on guesswork or basic analysis, which can lead to wrong decisions. Because of this, there is a need for a simple system that can analyze stock data and give useful insights.

In this project, I developed a system called StockSense AI that uses data analysis and machine learning to study stock market behavior. The main aim of this project is to predict whether the next day’s stock price will go up or down based on past data.

For this purpose, historical stock data is collected from Yahoo Finance using Python. The data is then cleaned and analyzed to find patterns. Features such as moving averages, daily returns, and RSI are created to improve prediction accuracy.

Machine learning models like Logistic Regression and Random Forest are used to make predictions. Among them, Random Forest gives better performance because it can handle complex patterns in data.

Finally, the results are shown using a simple web application built with Streamlit, where users can view stock trends and predictions in an easy and interactive way.

This project helps in understanding how data science and machine learning can be applied to real-world problems like stock market analysis.

2. Problem Statement

   Stock market prediction is inherently challenging due to the following reasons:

   1. High volatility — Stock prices can change dramatically within seconds due to news, events, or market sentiment
   2. Non-linearity — Price movements do not follow simple mathematical patterns
   3. Information overload — Traders are overwhelmed with data from multiple sources
   4. Lack of accessible tools — Advanced analytical tools used by institutional investors are expensive and 
      complex for individual users

Individual retail investors and small traders often lack access to intelligent tools that can help them make data-driven decisions. Most existing platforms either require deep financial knowledge to operate or are too costly for student or individual use.
This project addresses the need for a simple, accessible, and intelligent system that can analyze stock data, identify trends, apply machine learning, and present results in an easy-to-understand visual dashboard — all built using free and open-source tools.

3. Objectives

   The primary objectives of this project are:

   1. To collect real-time and historical stock market data using the Yahoo Finance API via the yfinance Python library
   2. To preprocess and clean raw stock data by handling missing values, removing duplicates, and correcting data types
   3. To perform Exploratory Data Analysis (EDA) using visualization libraries to identify trends, patterns, and correlations
   4. To apply feature engineering techniques to derive new meaningful features such as moving averages, RSI, volatility,and momentum
   5. To build and compare two machine learning models — Logistic Regression and Random Forest Classifier — for next-day price direction prediction
   6. To evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix
   7. To deploy the trained model in an interactive Streamlit web dashboard that displays live predictions and visualizations
   8. To understand and implement the complete data science workflow from data collection to prediction and deployment from data collection to deployment


4. Scope of the Project

   In Scope

   1. Collection and analysis of historical daily OHLCV (Open, High, Low, Close, Volume) stock data
   2. Preprocessing and cleaning of raw financial time-series data
   3. Visual analysis of price trends, volume, RSI, moving averages, and correlations
   4. Binary classification: predicting whether the next trading day's closing price will be higher (1) or lower (0) than today's
   5. An interactive browser-based dashboard accessible locally via Streamlit
   6. Support for any NSE/BSE listed Indian stock or global stock using its ticker symbol

   Out of Scope

   1. Intraday trading predictions (minute-by-minute price changes)
   2. Portfolio management or automated trade execution
   3. Sentiment analysis from news or social media
   4. Long-term forecasting (months or years ahead)
   5. Real-time trading bots or algorithmic trading systems


5. Literature Review
   Several studies have explored the application of machine learning in stock market prediction:

   1. Adebiyi et al. (2014) demonstrated that Artificial Neural Networks could outperform traditional ARIMA models in stock price forecasting, highlighting the importance of non-linear modeling approaches.
   2. Patel et al. (2015) compared four prediction models — Artificial Neural Networks, Support Vector Machines, Random Forest, and Naive Bayes — finding that ensemble methods like Random Forest performed consistently well across different market conditions.
   3. Fischer & Krauss (2018) showed that Long Short-Term Memory (LSTM) networks could capture temporal dependencies in stock data, though simpler models like Random Forest remained competitive on short-term predictions.
   4. Vijh et al. (2020) compared Random Forest and Artificial Neural Networks for next-day stock price prediction, concluding that Random Forest provided better accuracy with lower computational cost.

These studies collectively support the use of Random Forest as a strong baseline classifier for stock trend prediction, which forms the core machine learning approach in this project. The use of technical indicators such as Moving Averages and RSI as features is also well-established in the literature.

6. System Requirements

   Hardware Requirements:
   Component       Minimum Requirement

   Processor       Intel Core i3 or equivalent
   RAM             4 GB
   Storage         2 GB freespace
   Internet        Required for data fetching

   Software Requirements:

   Software          Version        Purpose

   Python            3.10+          Core programming language
   VS Code           Latest         Code editor
   Jupyter Notebook  Latest         Experimentation and EDA
   Chrome/Edge       Latest         Dashboard browser

   Python Libraries:

   Library          Version         Purpose

   yfinance         1.2.0           Fetch stock data from Yahoo Finance
   pandas           2.3.3           Data manipulation
   numpy            2.4.3           Numerical computing 
   matplotlib       3.10.8          Static visualizations
   seaborn          0.13.2          Statistical visualizations
   scikit-learn     1.8.0           Machine learning models
   streamlit        1.55.0          Web dashboard
   plotly           6.6.0           Interactive charts
   joblib           1.5.3           Model serialization

7. System Architecture
┌─────────────────────────────────────────────────────┐
│                    DATA LAYER                        │
│  Yahoo Finance API  ──►  yfinance  ──►  raw_data.csv │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────┐
│               PREPROCESSING LAYER                    │
│  Handle nulls  ►  Fix types  ►  processed_data.csv  │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING LAYER                 │
│  MA7, MA21, RSI, Daily Return, Volatility,           │
│  Price Range, Momentum, Target Label                 │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────┐
│              MACHINE LEARNING LAYER                  │
│  Train/Test Split  ►  Logistic Regression            │
│                    ►  Random Forest  ──►  .pkl model │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────┐
│               PRESENTATION LAYER                     │
│       Streamlit Dashboard  ──►  Web Browser          │
│  Live Charts  |  Prediction  |  Statistics           │
└─────────────────────────────────────────────────────┘

8. Methodology
   This project follows the standard CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, consisting of six sequential phases.

   Phase 1 – Data Collection
   Objective: Fetch historical stock data for Reliance Industries from Yahoo Finance.
   Tool Used: yfinance Python library
   Code:
        pythonimport yfinance as yf
        import pandas as pd

        ticker     = "RELIANCE.NS"
        start_date = "2020-01-01"
        end_date   = "2024-12-31"

       df = yf.download(ticker, start=start_date, end=end_date)
       df.reset_index(inplace=True)
       df.to_csv("data/raw_data.csv", index=False)

       Output: 1,257 rows of daily OHLCV data saved to raw_data.csv

    Data Attributes:
    ColumnDescriptionDateTrading dateOpenOpening price of the dayHighHighest price of the dayLowLowest price of the dayCloseClosing price of the dayVolumeNumber of shares traded

    Phase 2 – Data Preprocessing

    Objective: Clean and prepare raw data for analysis and modeling.
    Steps Performed:

    Flattened multi-level column headers produced by newer versions of yfinance
    Renamed columns to clean, consistent names
    Converted Date column to proper datetime format
    Handled missing values using forward fill (ffill) — standard practice for time-series financial data
    Removed duplicate rows to ensure data integrity
    Corrected data types — price columns to float, volume to int

    Result: 1,257 clean rows saved to processed_data.csv with zero missing values.

    Phase 3 – Exploratory Data Analysis

    Objective: Visually explore the data to understand trends, patterns, and correlations.
    Charts Created:

        ChartKey InsightClosing Price TrendOverall upward trend from 2020 to 2024 with COVID dip in early 2020Moving AveragesMA7 and MA21 crossovers visible as clear trend signalsDaily ReturnsReturns normally distributed around 0, with occasional spikes during market eventsTrading VolumeVolume spikes correlate with major price movementsCorrelation HeatmapOpen, High, Low, Close are highly correlated (>0.99); Volume has low correlation with priceCandlestick ChartClear bullish and bearish patterns visible in 90-day view

    Key EDA Findings:

    Reliance stock showed a strong recovery after the March 2020 COVID crash
    Average daily return was positive, indicating long-term bullish trend
    High volatility observed during 2020 and early 2022
    Open, High, Low, and Close prices are strongly correlated with each other


    Phase 4 – Feature Engineering

    Objective: Create new meaningful features from raw data to improve model predictive power.
    Features Created:
     FeatureFormulaSignificanceMA77-day rolling mean of CloseShort-term trend indicatorMA2121-day rolling mean of CloseMedium-term trend indicatorDaily_Return(Close - prev Close) / prev Close × 100Daily percentage changeVolatility7-day rolling std of Daily_ReturnMeasures price riskPrice_RangeHigh - LowIntraday price movementMomentumClose - Close (10 days ago)Speed of price movementRSI14-day Relative Strength IndexOverbought/Oversold signalTarget1 if next day Close > today, else 0Prediction label

    RSI Interpretation:

    RSI > 70 → Stock is overbought (possible price drop)
    RSI < 30 → Stock is oversold (possible price rise)
    RSI 30–70 → Neutral zone

    Final Dataset: 699 rows × 13 columns after dropping NaN rows created by rolling calculations.

    Phase 5 – Model Building and Evaluation
    
    Objective: Train and evaluate machine learning models to predict next-day price direction.
    Train/Test Split
    Total data   : 699 rows
    Training set : 559 rows (80%) — from 2020 to mid-2023
    Testing set  : 140 rows (20%) — from mid-2023 to 2024

    Note: shuffle=False was used to maintain chronological order — training on past data and testing on future data, which is critical for time-series problems.

    Model 1 – Logistic Regression
      A linear classification model that finds a decision boundary between Up and Down classes.

    Features were scaled using StandardScaler before training
     Accuracy: 51.43%

    Model 2 – Random Forest Classifier
     An ensemble model that builds 100 decision trees and takes the majority vote for prediction.

    Parameters: n_estimators=100, max_depth=5, random_state=42
     Accuracy: 52.14% 🏆

    Model Comparison
    ModelAccuracyLogistic Regression51.43%Random Forest52.14% ✅ Winner
    Feature Importance (Random Forest)
    RankFeatureImportance Score1Low~0.1052Volatility~0.1003Daily_Return~0.0984Price_Range~0.0915High~0.089
    Interpretation: The day's lowest price and volatility carry the most predictive information for next-day price direction.
    Why ~52% Accuracy is Acceptable

    Stock market prediction accuracy of 52% is considered good because:

    Random guessing gives exactly 50%
    The model is beating random chance consistently
    Professional hedge funds target only 55–60%
    The stock market is inherently random and influenced by unpredictable external events


    Phase 6 – Dashboard Deployment

    Objective: Deploy the trained model in an interactive web-based dashboard.
    Tool Used: Streamlit
    Dashboard Features:
     SectionDescriptionSidebarStock ticker input, time period selector, quick stock buttonsKey MetricsLive current price, day high/low, volume, RSI, volatilityML PredictionUP/DOWN prediction with confidence percentagePrice ChartInteractive closing price with MA7 and MA21 overlayCandlestick ChartLast 90 trading days OHLC chartRSI ChartRSI with overbought/oversold zones highlightedVolume ChartDaily trading volume bar chartReturns AnalysisDaily returns bar chart and distribution histogramStatistics TableComplete statistical summary of the selected period

    Run Command:

    bashcd app
    streamlit run app.py

9. Dataset Description
AttributeValueStockReliance Industries Ltd.Ticker SymbolRELIANCE.NSExchangeNational Stock Exchange (NSE), IndiaData SourceYahoo Finance (via yfinance)Time PeriodJanuary 2020 – December 2024FrequencyDaily (trading days only)Raw Records1,257 rowsFinal Records699 rows (after feature engineering)Features12 input features + 1 target labelTarget Classes0 = Price Down, 1 = Price Up

10. Implementation
Project Folder Structure
StockSense-AI/
│
├── data/
│   ├── raw_data.csv               ← Downloaded stock data
│   └── processed_data.csv         ← Cleaned + feature engineered data
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_training.ipynb
│
├── models/
│   ├── stock_model.pkl            ← Saved Random Forest model
│   └── scaler.pkl                 ← Saved StandardScaler
│
├── app/
│   └── app.py                     ← Streamlit dashboard
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── prediction.py
│
├── report/
│   └── project_report.md          ← This report
│
├── requirements.txt
└── README.md

   Key Implementation Decisions

   1. yfinance over manual CSV — Enables live, real-time data fetching for any stock
   2. Random Forest over Logistic Regression — Better handles non-linear patterns in financial data
   3. Forward fill for missing values — Industry standard for financial time series
   4. shuffle=False in train/test split — Prevents data leakage in time-series data
   5. Streamlit for dashboard — Pure Python, no web development knowledge required
   6. joblib for model saving — Efficient binary serialization for scikit-learn models


11. Results and Analysis

Model Performance Summary

MetricLogistic RegressionRandom ForestAccuracy51.43%52.14%Best Model—✅ Random Forest
EDA Key Findings

Reliance stock grew approximately 60% from January 2020 to December 2024
March 2020 showed the highest volatility due to COVID-19 market crash
Average daily return was positive, confirming long-term bullish trend
Low price and Volatility are the strongest predictors of next-day direction
RSI averaged in the neutral zone (40–60), indicating no prolonged extreme conditions
Volume spikes of 2–3× average were observed during major market events

Prediction Output (Live Dashboard)
Stock          : RELIANCE.NS
Current Price  : ₹1,347.80
Day Change     : +₹43.20 (+3.31%)
RSI            : 41.6 (Neutral)
Volatility     : 2.78%
Prediction     : 📈 PRICE LIKELY TO GO UP
Confidence     : 54.1%

12. Dashboard Screenshots

Add screenshots of your running Streamlit dashboard here

Screenshot 1: Main dashboard with Key Metrics and ML Prediction box
Screenshot 2: Price trend chart with Moving Averages (MA7 and MA21)
Screenshot 3: Candlestick chart (Last 90 Days)
Screenshot 4: RSI chart with overbought/oversold zones
Screenshot 5: Statistical Summary table

13. Applications
StockSense AI has practical applications in several domains:

Personal Investment Support — Helps retail investors make more informed decisions using data-driven insights rather than pure intuition
Financial Education — Teaches students and beginners the fundamentals of technical analysis and machine learning in an interactive way
Academic Research — Provides a baseline framework for researchers exploring ML-based financial forecasting
Trading Strategy Development — Can serve as a foundation for developing and backtesting algorithmic trading strategies
Portfolio Monitoring — The dashboard can be extended to monitor multiple stocks simultaneously and track portfolio performance


14. Limitations

Limited accuracy (~52%) — Stock markets are influenced by unpredictable factors like news, geopolitical events, and investor sentiment that are not captured in historical price data alone
No sentiment analysis — The model does not consider news headlines, social media trends, or economic announcements that significantly impact stock prices
Single stock focus — The model was trained specifically on Reliance Industries data and may not generalize perfectly to other stocks without retraining
Prediction horizon — Only next-day (short-term) direction is predicted; longer-term forecasts are not supported
Market regime changes — The model may perform differently during unusual market conditions (e.g., global financial crises) that differ significantly from the training period
No real-time trading — The system provides predictions but does not execute trades automatically


15. Future Scope
The project can be improved by using advanced models like LSTM, adding real-time data, and including features like sentiment analysis and portfolio tracking.

16. Conclusion
This project helped me understand how data science and machine learning can be used in real-world problems like stock market analysis. I collected stock data, processed it, created useful features, and trained a model to predict whether the price will go up or down the next day.

Although the model accuracy is around 52%, it still performs better than random guessing. The project also helped me learn tools like Python, pandas, and Streamlit.

Overall, this project gave me practical experience in building a complete data science application.

17. References

“Previous studies show that machine learning models like Random Forest perform well for stock prediction.”


This report was prepared as part of the BCA Final Year Project submission.
StockSense AI is built entirely using free and open-source Python libraries.
⚠️ This project is for educational purposes only and does not constitute financial advice