import sys
import os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import shap
import torch
from predict import load_model_and_scaler, predict_ticker
from backtest import run_backtest

st.set_page_config(
    page_title="AI Trading Signals",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Signal Generator")
st.markdown("Enter any stock ticker and get AI-powered BUY, SELL, or HOLD signals with full explainability.")
st.markdown("---")


@st.cache_resource
def get_model():
    return load_model_and_scaler()


with st.spinner("Loading model..."):
    model, scaler, features = get_model()

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input(
        "Stock Ticker", value="AAPL",
        placeholder="e.g. AAPL, TSLA, GOOGL, MSFT").upper()
with col2:
    analyse = st.button("Generate Signal", type="primary")

if analyse:
    with st.spinner(f"Analysing {ticker}..."):
        try:
            result = predict_ticker(ticker, model, scaler, features)
            backtest = run_backtest(
                result["df"], result["all_preds"])

            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            signal = result["signal"]
            color  = result["color"]

            with col1:
                if signal == "BUY":
                    st.success(f"## 🟢 {signal}")
                elif signal == "SELL":
                    st.error(f"## 🔴 {signal}")
                else:
                    st.info(f"## ⚪ {signal}")

            with col2:
                st.metric("Confidence", f"{result['confidence']}%")
            with col3:
                st.metric("Total Return", f"{backtest['total_return']}%")
            with col4:
                st.metric("Buy & Hold Return",
                          f"{backtest['buy_and_hold']}%")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Signal Probabilities")
                prob_df = pd.DataFrame({
                    "Signal": ["SELL", "HOLD", "BUY"],
                    "Probability": [
                        result["sell_prob"],
                        result["hold_prob"],
                        result["buy_prob"]
                    ],
                    "Color": ["red", "gray", "green"]
                })
                fig = px.bar(
                    prob_df, x="Signal", y="Probability",
                    color="Signal",
                    color_discrete_map={
                        "SELL": "red", "HOLD": "gray", "BUY": "green"},
                    title="Signal Probability Distribution"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Portfolio vs Buy & Hold")
                dates = backtest["dates"]
                fig2  = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=dates,
                    y=backtest["portfolio_values"],
                    name="AI Strategy",
                    line=dict(color="cyan", width=2)
                ))
                initial = 10000
                prices  = result["df"]["Close"].values
                prices  = prices[len(prices) - len(dates):]
                bh = [initial * (p / float(prices[0]))
                      for p in prices]
                fig2.add_trace(go.Scatter(
                    x=dates, y=bh,
                    name="Buy & Hold",
                    line=dict(color="orange", width=2,
                              dash="dash")
                ))
                fig2.update_layout(
                    title="Portfolio Performance",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)"
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.subheader("Price Chart with Signals")
            df_plot = result["df"].copy()
            preds   = result["all_preds"]
            df_plot = df_plot.iloc[len(df_plot) - len(preds):]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot["Close"],
                name="Price",
                line=dict(color="white", width=1)
            ))

            for signal_val, color, name in [
                    (2, "green", "BUY"),
                    (0, "red", "SELL")]:
                mask = preds == signal_val
                idx  = df_plot.index[mask]
                vals = df_plot["Close"].values[mask]
                fig3.add_trace(go.Scatter(
                    x=idx, y=vals, mode="markers",
                    name=name,
                    marker=dict(color=color, size=6)
                ))

            fig3.update_layout(
                title=f"{ticker} Price with AI Signals",
                xaxis_title="Date",
                yaxis_title="Price ($)"
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("---")
            st.subheader("Backtest Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Portfolio Value",
                          f"${backtest['final_value']:,.2f}")
            with col2:
                st.metric("Number of Trades",
                          backtest["num_trades"])
            with col3:
                st.metric("Starting Capital", "$10,000")

            st.caption(
                "Model: LSTM with Attention trained on technical indicators, "
                "Explainability via SHAP values")

        except Exception as e:
            st.error(f"Error analysing {ticker}: {str(e)}")
            st.info("Try a different ticker like AAPL, MSFT, GOOGL, or TSLA")