"""
src/model/tune.py
-----------------
Optuna hyperparameter search over the AirQualityMLP.

Search space:
  - learning rate:     log-uniform [1e-4, 1e-2]
  - hidden layer 1:    categorical [64, 128, 256]
  - hidden layer 2:    categorical [32, 64, 128]
  - dropout:           uniform [0.1, 0.5]
  - batch size:        categorical [128, 256, 512]

Each trial runs a short training loop (10 epochs) and reports val_f1.
Optuna prunes unpromising trials early (MedianPruner).

Run:
    python -m src.model.tune --trials 30
"""

import argparse
import sys
import os


import mlflow
import optuna
from optuna.pruners import MedianPruner

from src.model.train import load_split, train_epoch, eval_epoch
from src.config import FEATURE_COLS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model.mlp import AirQualityMLP


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    h1 = trial.suggest_categorical("hidden_1", [64, 128, 256])
    h2 = trial.suggest_categorical("hidden_2", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    epochs = 10  # short for search; full training done separately

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = AirQualityMLP(
        input_dim=len(FEATURE_COLS),
        hidden_dims=[h1, h2],
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        _, _, val_f1 = eval_epoch(model, val_loader, criterion, device)

        # Report intermediate value so Optuna can prune bad trials early
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_f1


def main(n_trials: int = 30):
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name="air-quality-mlp-tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nBest trial:")
    t = study.best_trial
    print(f"  val_f1 = {t.value:.4f}")
    for k, v in t.params.items():
        print(f"  {k}: {v}")

    print("\nRun src/model/train.py with these params for full training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    main(args.trials)