import numpy as np
import pandas as pd


def run_backtest(df, predictions, initial_capital=10000):
    df      = df.copy().iloc[len(df) - len(predictions):]
    prices  = df["Close"].values
    capital = initial_capital
    shares  = 0
    portfolio_values = []
    trades = []

    for i, (price, signal) in enumerate(zip(prices, predictions)):
        price = float(price)
        if signal == 2 and capital > 0:
            shares  = capital / price
            capital = 0
            trades.append(("BUY", i, price))
        elif signal == 0 and shares > 0:
            capital = shares * price
            shares  = 0
            trades.append(("SELL", i, price))

        portfolio_values.append(capital + shares * price)

    final_value   = portfolio_values[-1]
    total_return  = (final_value - initial_capital) / initial_capital * 100
    buy_and_hold  = (prices[-1] / prices[0] - 1) * 100

    return {
        "portfolio_values": portfolio_values,
        "final_value":      round(final_value, 2),
        "total_return":     round(total_return, 2),
        "buy_and_hold":     round(buy_and_hold, 2),
        "num_trades":       len(trades),
        "trades":           trades,
        "dates":            df.index.tolist()
    }