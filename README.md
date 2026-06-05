

# 📈 AI Trading Signal Generator

> An LSTM-powered trading system that generates BUY, SELL, and HOLD signals for any stock ticker, backed by technical indicators, backtesting, and a live interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project builds an end-to-end AI trading signal system using a **Long Short-Term Memory (LSTM) neural network with attention**, trained on real historical stock data.

The model learns from 10 technical indicators (RSI, MACD, Bollinger Bands, EMA, etc.) and predicts whether to BUY, SELL, or HOLD a given stock.

It includes:
- A full backtesting engine
- A live Streamlit dashboard
- Portfolio simulation vs buy-and-hold strategy

---

## 🎯 Key Features

- 📥 Automatic stock data download using `yfinance`
- 🧠 10+ engineered technical indicators
- 📊 LSTM + Attention deep learning model
- 📈 BUY / SELL / HOLD classification signals
- 🧪 Backtesting engine with performance tracking
- 📉 Buy-and-hold benchmark comparison
- 📊 Interactive Plotly charts
- 🌐 Streamlit dashboard for live ticker analysis

---

## 🏗️ Architecture

Raw Stock Data (OHLCV) ↓ Technical Indicator Engineering (RSI, MACD, Bollinger Bands, Stochastic, EMA) ↓ 60-Day Sequence Windows ↓ LSTM (2 layers, 128 hidden units) ↓ Attention Mechanism ↓ Fully Connected Layer ↓ BUY / SELL / HOLD + Confidence Score ↓ Backtesting Engine + Portfolio Simulation

---

## 📊 Features Used

| Feature        | Description |
|----------------|-------------|
| Close          | Closing price |
| Volume         | Trading volume |
| RSI            | Relative Strength Index |
| MACD           | Moving Average Convergence Divergence |
| MACD Signal    | Signal line |
| Bollinger High | Upper Bollinger Band |
| Bollinger Low  | Lower Bollinger Band |
| Stochastic     | Momentum indicator |
| EMA Fast       | Fast exponential moving average |
| EMA Slow       | Slow exponential moving average |

---

## 🖥️ Live Dashboard

Run the app locally:

```bash
streamlit run app.py

Features:

Live BUY / SELL / HOLD signal

Confidence scores

Portfolio vs Buy & Hold comparison

Price chart with signal markers

Full backtest summary



---

🚀 Quick Start

git clone https://github.com/Eddiegah/trading-signal-ai.git
cd trading-signal-ai

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

python src/train.py
streamlit run app.py


---

📁 Project Structure

trading-signal-ai/
├── app.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── backtest.py
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

Technical analysis indicators (TA)

scikit-learn (preprocessing & evaluation)

Streamlit (dashboard)

Plotly (visualization)

NumPy & Pandas



---

💡 Why LSTM with Attention?

LSTM captures time-series dependencies, while attention improves performance by focusing on the most important time steps.

This helps the model:

Focus on critical market movements

Reduce noise sensitivity

Improve prediction accuracy



---

⚠️ Disclaimer

This project is for educational and research purposes only. It is NOT financial advice. Do not make investment decisions based solely on this model.


---

📄 License

MIT License


---

👤 Author

Built with 📈 by Eddiegah
