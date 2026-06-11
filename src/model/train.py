"""
src/model/train.py
------------------
Train the AirQualityMLP and log everything to MLflow.

What gets logged:
  - Params: lr, hidden_dims, dropout, epochs, batch_size
  - Metrics per epoch: train_loss, val_loss, val_accuracy, val_f1
  - Artifacts: model checkpoint, scaler, confusion matrix plot

Run:
    python -m src.model.train
    python -m src.model.train --lr 1e-3 --hidden 128 64 --dropout 0.3
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import FEATURE_COLS, LABEL_COL
from src.model.mlp import AirQualityMLP

PROCESSED_DIR = Path("data/processed")
MLFLOW_EXPERIMENT = "air-quality-risk-scorer"


def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df[LABEL_COL].values, dtype=torch.long)
    return X, y


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # torch.no_grad() disables gradient tracking during inference —
    # reduces memory usage and speeds up evaluation significantly.
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(X_batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, accuracy, f1


def main(
    lr: float = 1e-3,
    hidden_dims: list[int] = [128, 64],
    dropout: float = 0.3,
    epochs: int = 30,
    batch_size: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size
    )

    model = AirQualityMLP(
        input_dim=len(FEATURE_COLS),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "lr": lr,
            "hidden_dims": str(hidden_dims),
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        })

        best_val_f1 = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                },
                step=epoch,
            )

            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Restore best weights and log model artifact
        model.load_state_dict(best_model_state)
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(PROCESSED_DIR / "scaler.pkl"), artifact_path="model")
        mlflow.log_metric("best_val_f1", best_val_f1)

        # Save checkpoint for serving (DVC tracks this file)
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), models_dir / "best_model.pt")

        # Write metrics JSON (DVC tracks this file)
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        metrics = {
            "best_val_f1": round(best_val_f1, 4),
            "final_val_loss": round(val_loss, 4),
            "final_val_accuracy": round(val_acc, 4),
            "epochs": epochs,
        }
        with open(metrics_dir / "train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nBest val F1: {best_val_f1:.4f}")
        print(f"Saved → models/best_model.pt")
        print(f"Saved → metrics/train_metrics.json")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args.lr, args.hidden, args.dropout, args.epochs, args.batch_size)