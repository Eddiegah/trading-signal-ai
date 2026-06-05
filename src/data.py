import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

SEQUENCE_LENGTH = 60
FEATURES = [
    "Close", "Volume", "momentum_rsi", "trend_macd",
    "trend_macd_signal", "volatility_bbh", "volatility_bbl",
    "momentum_stoch", "trend_ema_fast", "trend_ema_slow"
]


def download_data(ticker, period="5y"):
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, period=period, auto_adjust=True)
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df


def add_features(df):
    df = df.copy()
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )
    return df


def create_labels(df, threshold=0.02):
    df = df.copy()
    future_return = df["Close"].shift(-5) / df["Close"] - 1
    df["label"] = 1
    df.loc[future_return > threshold, "label"] = 2
    df.loc[future_return < -threshold, "label"] = 0
    df.dropna(inplace=True)
    return df


def prepare_sequences(df, features=FEATURES, seq_len=SEQUENCE_LENGTH):
    df = df.copy()
    available = [f for f in features if f in df.columns]
    data = df[available].values
    labels = df["label"].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i - seq_len:i])
        y.append(labels[i])

    return np.array(X), np.array(y), scaler, available


def get_full_pipeline(ticker, period="5y"):
    df = download_data(ticker, period)
    df = add_features(df)
    df = create_labels(df)
    X, y, scaler, features = prepare_sequences(df)
    return X, y, scaler, features, df