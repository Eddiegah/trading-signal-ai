import torch
import numpy as np
import pickle
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TradingLSTM
from data import download_data, add_features, create_labels, prepare_sequences

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./results"
LABELS     = {0: "SELL", 1: "HOLD", 2: "BUY"}
COLORS     = {0: "red", 1: "gray", 2: "green"}


def load_model_and_scaler():
    with open(f"{OUTPUT_DIR}/features.json") as f:
        features = json.load(f)
    with open(f"{OUTPUT_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = TradingLSTM(input_size=len(features))
    model.load_state_dict(torch.load(
        f"{OUTPUT_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()
    return model, scaler, features


def predict_ticker(ticker, model, scaler, features):
    df       = download_data(ticker, period="2y")
    df       = add_features(df)
    df       = create_labels(df)
    X, y, _, _ = prepare_sequences(df, features=features)

    X_tensor = torch.FloatTensor(X).to(DEVICE)
    with torch.no_grad():
        logits = model(X_tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

    latest_pred  = int(preds[-1])
    latest_probs = probs[-1]

    return {
        "signal":      LABELS[latest_pred],
        "color":       COLORS[latest_pred],
        "confidence":  round(float(latest_probs[latest_pred]) * 100, 2),
        "sell_prob":   round(float(latest_probs[0]) * 100, 2),
        "hold_prob":   round(float(latest_probs[1]) * 100, 2),
        "buy_prob":    round(float(latest_probs[2]) * 100, 2),
        "all_preds":   preds,
        "all_probs":   probs,
        "df":          df
    }