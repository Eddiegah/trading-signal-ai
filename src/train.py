import torch
import torch.nn as nn
import numpy as np
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model import TradingLSTM
from data import get_full_pipeline
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ──────────────────────────────────────────────
TICKER      = "AAPL"
EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR  = "./results"
# ────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print("Preparing data...")

    X, y, scaler, features, df = get_full_pipeline(TICKER)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, shuffle=False)

    def to_loader(X, y, shuffle=True):
        ds = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    test_loader  = to_loader(X_test,  y_test,  shuffle=False)

    model     = TradingLSTM(input_size=X.shape[2]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    best_val_acc = 0

    print(f"\nTraining on {TICKER} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch.to(DEVICE)).argmax(dim=1).cpu()
                val_preds.extend(preds.numpy())
                val_true.extend(y_batch.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(1 - val_acc)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f"{OUTPUT_DIR}/best_model.pth")
            print(f"  Saved best model")

    model.load_state_dict(torch.load(
        f"{OUTPUT_DIR}/best_model.pth", map_location=DEVICE))
    model.eval()

    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(DEVICE)).argmax(dim=1).cpu()
            test_preds.extend(preds.numpy())
            test_true.extend(y_batch.numpy())

    acc = accuracy_score(test_true, test_preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(
        test_true, test_preds,
        target_names=["SELL", "HOLD", "BUY"]))

    import pickle
    with open(f"{OUTPUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{OUTPUT_DIR}/features.json", "w") as f:
        json.dump(features, f)

    metrics = {
        "ticker": TICKER,
        "test_accuracy": round(acc, 4),
        "best_val_accuracy": round(best_val_acc, 4)
    }
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()