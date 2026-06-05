

# 📈 AI Trading Signal Generator

> An end-to-end AI-powered trading system that uses an LSTM with Attention Mechanism to generate BUY, SELL, and HOLD signals from real stock market data, supported by technical indicators, backtesting, and a live interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat-square&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🚀 Overview

This project is a **machine learning-based trading intelligence system** that analyzes historical stock data and generates predictive trading signals.

It combines:
- Deep Learning (LSTM + Attention)
- Technical Analysis Indicators
- Backtesting Engine
- Interactive Web Dashboard (Streamlit)

---

## 🎯 Key Features

- 📊 Real-time stock data via `yfinance`
- 🧠 LSTM neural network with Attention mechanism
- 📉 10+ engineered technical indicators (RSI, MACD, Bollinger Bands, EMA, etc.)
- 📈 BUY / SELL / HOLD signal classification
- 🧪 Full historical backtesting engine
- 💰 Portfolio performance vs Buy & Hold benchmark
- 📊 Interactive Plotly charts
- 🌐 Streamlit dashboard (any stock ticker input)

---

## 🏗️ System Architecture

Raw Stock Data (OHLCV) ↓ Feature Engineering (RSI, MACD, Bollinger Bands, Stochastic, EMA) ↓ Sequence Builder (60-day windows) ↓ LSTM Neural Network (2 layers, 128 hidden units) ↓ Attention Mechanism ↓ Fully Connected Layer ↓ Trading Signal Output (BUY / SELL / HOLD + Confidence) ↓ Backtesting Engine ↓ Performance Evaluation vs Buy & Hold

---

## 📊 Technical Indicators Used

| Indicator        | Purpose |
|------------------|--------|
| Close Price      | Market reference price |
| Volume           | Trading activity strength |
| RSI              | Momentum strength indicator |
| MACD             | Trend direction detection |
| Bollinger Bands  | Volatility measurement |
| Stochastic Osc.  | Overbought/oversold conditions |
| EMA Fast         | Short-term trend |
| EMA Slow         | Long-term trend |

---

## 🖥️ Live Dashboard

Run locally:

```bash
streamlit run app.py

Features:

Live BUY / SELL / HOLD signal

Confidence scores

Portfolio vs Buy & Hold comparison

Price charts with signal markers

Backtesting summary



---

🚀 Installation & Setup

git clone https://github.com/Eddiegah/trading-signal-ai.git
cd trading-signal-ai

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

python src/train.py
streamlit run app.py


---

📁 Project Structure

trading-signal-ai/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── backtest.py
│
└── results/
    ├── best_model.pth
    ├── scaler.pkl
    ├── features.json
    └── metrics.json


---

🧰 Tech Stack

Python 3.11

PyTorch (LSTM + Attention)

yfinance (stock data)

Technical Indicators (TA)

scikit-learn (preprocessing & evaluation)

Streamlit (dashboard)

Plotly (visualization)

NumPy & Pandas



---

🧠 Why LSTM + Attention?

LSTM captures sequential dependencies in time-series data, while attention improves learning by focusing on the most important time steps.

This helps the model:

Focus on key market movements

Reduce noise sensitivity

Improve prediction accuracy



---

⚠️ Disclaimer

This project is for educational and research purposes only. It is NOT financial advice.


---

📄 License

MIT License


---

👤 Author

Eddiegah

GitHub: https://github.com/Eddiegah
