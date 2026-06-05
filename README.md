
# 📈 AI Trading Signal Generator

> An end-to-end AI-powered trading system that uses an LSTM with Attention Mechanism to generate BUY, SELL, and HOLD signals from real stock market data, supported by technical indicators, backtesting, and a live interactive dashboard.

---

## 🚀 Overview

This project is a **machine learning-based trading intelligence system** that analyzes historical stock data and generates predictive trading signals.

It combines:
- Deep Learning (LSTM + Attention)
- Technical Analysis Indicators
- Backtesting Engine
- Interactive Web Dashboard (Streamlit)

The system is designed to simulate real-world trading decisions and evaluate performance against a buy-and-hold strategy.

---

## 🎯 Key Features

- 📊 Real-time stock data via `yfinance`
- 🧠 LSTM neural network with Attention mechanism
- 📉 10+ engineered technical indicators (RSI, MACD, Bollinger Bands, EMA, etc.)
- 📈 BUY / SELL / HOLD signal classification
- 🧪 Full historical backtesting engine
- 💰 Portfolio performance vs Buy & Hold benchmark
- 📊 Interactive visualizations using Plotly
- 🌐 Streamlit-based live dashboard (any stock ticker input)

---

## 🏗️ System Architecture

Raw Stock Data (OHLCV) ↓ Feature Engineering (RSI, MACD, Bollinger Bands, Stochastic, EMA) ↓ Sequence Builder (60-day windows) ↓ LSTM Neural Network (2 layers, 128 hidden units) ↓ Attention Mechanism (focus on key time steps) ↓ Fully Connected Layer ↓ Trading Signal Output (BUY / SELL / HOLD + Confidence) ↓ Backtesting Engine ↓ Performance Evaluation vs Buy & Hold

---

## 📊 Technical Indicators Used

| Indicator        | Purpose |
|------------------|--------|
| Close Price      | Market reference price |
| Volume           | Trading activity strength |
| RSI              | Momentum strength indicator |
| MACD             | Trend direction detection |
| MACD Signal      | Signal smoothing |
| Bollinger Bands  | Volatility measurement |
| Stochastic Osc.  | Overbought/oversold conditions |
| EMA Fast         | Short-term trend |
| EMA Slow         | Long-term trend |

---

## 🖥️ Live Dashboard

Launch the Streamlit app:

```bash
streamlit run app.py

Dashboard Features:

Real-time trading signal generation

Confidence score visualization

Portfolio vs Buy & Hold comparison

Interactive price charts with signal markers

Full backtesting summary



---

🚀 Installation & Setup

# Clone repository
git clone https://github.com/Eddiegah/trading-signal-ai.git
cd trading-signal-ai

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Run dashboard
streamlit run app.py


---

📁 Project Structure

trading-signal-ai/
│
├── app.py                     # Streamlit dashboard
├── requirements.txt
│
├── src/
│   ├── data.py               # Data loading & feature engineering
│   ├── model.py              # LSTM + Attention model
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Inference pipeline
│   └── backtest.py           # Backtesting engine
│
└── results/
    ├── best_model.pth
    ├── scaler.pkl
    ├── features.json
    └── metrics.json


---

🧰 Tech Stack

🐍 Python 3.11

🔥 PyTorch (Deep Learning)

📈 LSTM + Attention Architecture

📊 yfinance (Market Data)

📉 Technical Analysis Indicators (TA)

🧮 scikit-learn (Preprocessing & Metrics)

🌐 Streamlit (Web Dashboard)

📊 Plotly (Visualization)

📦 NumPy & Pandas



---

🧠 Why LSTM + Attention?

Traditional LSTMs treat all past time steps equally.

The Attention Mechanism improves performance by:

Focusing on the most relevant market events

Reducing noise from irrelevant time periods

Improving prediction stability and accuracy


This makes the model more aligned with real financial decision-making.


---

⚠️ Disclaimer

This project is built for educational and research purposes only.

It does NOT constitute financial advice. Always conduct your own research before making investment decisions.


---

📄 License

This project is licensed under the MIT License.


---

👤 Author

Eddiegah

GitHub: github.com/Eddiegah




⭐ If you like this project, consider giving it a star on GitHub!
