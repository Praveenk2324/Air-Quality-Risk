"""
src/data/preprocess.py
----------------------
Transform raw OpenAQ readings into a training-ready feature matrix.

Steps:
  1. Pivot long-format rows → wide format (one row per location × hour)
  2. Engineer cyclical time features (hour, month as sin/cos)
  3. Assign WHO risk tier label from PM2.5 AQI breakpoints
  4. Stratified train / val / test split (70/15/15)
  5. Fit StandardScaler on train only, save for inference

Output: data/processed/{train,val,test}.csv + scaler.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, LABEL_COL

RAW_PATH = Path("data/raw/latest.csv")
PROCESSED_DIR = Path("data/processed")

# WHO AQI breakpoints for PM2.5 (µg/m³)
PM25_BREAKPOINTS = [
    (0,     12.0,   0),   # Good
    (12.1,  35.4,   1),   # Moderate
    (35.5,  150.4,  2),   # Unhealthy
    (150.5, np.inf, 3),   # Hazardous
]

# Minimum rows per class needed for stratified split to work.
# If any class falls below this, we fall back to a non-stratified split.
MIN_SAMPLES_PER_CLASS = 10


def assign_risk_tier(pm25: float) -> int:
    """Map PM2.5 concentration to WHO risk tier 0–3."""
    for lo, hi, tier in PM25_BREAKPOINTS:
        if lo <= pm25 <= hi:
            return tier
    return 3


def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-format sensor readings to wide format.

    Input:  one row per (location, timestamp, parameter)
    Output: one row per (location, hour) with columns pm25 / no2 / o3
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.floor("h")
    df = df[df["parameter"].isin(["pm25", "no2", "o3"])]

    wide = (
        df.groupby(["location_id", "hour", "parameter"])["value"]
        .median()
        .unstack("parameter")
        .reset_index()
    )
    wide.columns.name = None
    return wide


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour-of-day and month as sin/cos pairs.

    Why sin/cos: hour 23 and hour 0 are adjacent, but numerically far apart.
    Cyclical encoding preserves that adjacency for the model.
    """
    df = df.copy()
    hour  = df["hour"].dt.hour
    month = df["hour"].dt.month

    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def safe_split(X, y, test_size: float, random_state: int = 42):
    """
    Stratified split with fallback to non-stratified if any class is too sparse.

    The OpenAQ dataset for a single city can be heavily skewed — e.g. almost
    all readings may fall in tier 0 (Good) with only a handful in tier 3
    (Hazardous). sklearn's train_test_split will raise if a class has fewer
    samples than the number of splits, so we catch that and fall back.
    """
    class_counts = y.value_counts()
    min_count = class_counts.min()

    if min_count < MIN_SAMPLES_PER_CLASS:
        print(
            f"  WARNING: class {class_counts.idxmin()} has only {min_count} samples. "
            f"Falling back to non-stratified split."
        )
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH)
    print(f"  Raw shape: {raw.shape}")

    # ── 1. Pivot ──────────────────────────────────────────────────────────────
    wide = pivot_wide(raw)
    print(f"  After pivot: {wide.shape}")

    if wide.empty:
        raise RuntimeError(
            "Pivot produced an empty DataFrame. "
            "Check that latest.csv contains pm25/no2/o3 rows with valid values."
        )

    # ── 2. Drop rows with no PM2.5 (can't label without it) ──────────────────
    before = len(wide)
    wide = wide.dropna(subset=["pm25"])
    print(f"  Dropped {before - len(wide)} rows missing PM2.5 → {len(wide)} remain")

    if len(wide) < 50:
        print(
            f"  WARNING: only {len(wide)} labelled rows. "
            f"Consider fetching more data (increase --measurements-per-sensor)."
        )

    # ── 3. Fill missing pollutants with column median ─────────────────────────
    for col in ["no2", "o3"]:
        if col in wide.columns:
            n_missing = wide[col].isna().sum()
            if n_missing > 0:
                median_val = wide[col].median()
                wide[col] = wide[col].fillna(median_val)
                print(f"  Filled {n_missing} missing {col} values with median={median_val:.2f}")
        else:
            wide[col] = 0.0
            print(f"  WARNING: no {col} sensors found — filling column with 0.0")

    # ── 4. Time features + label ──────────────────────────────────────────────
    wide = add_time_features(wide)
    wide[LABEL_COL] = wide["pm25"].apply(assign_risk_tier)

    print(f"\n  Label distribution:")
    print(wide[LABEL_COL].value_counts().sort_index().to_string())

    features = wide[FEATURE_COLS + [LABEL_COL]].dropna()
    print(f"\n  Final feature matrix: {features.shape}")

    # ── 5. Train / val / test split ───────────────────────────────────────────
    X = features[FEATURE_COLS]
    y = features[LABEL_COL]

    X_tv, X_test, y_tv, y_test = safe_split(X, y, test_size=0.15)
    X_train, X_val, y_train, y_val = safe_split(
        X_tv, y_tv, test_size=0.15 / 0.85
    )

    print(f"\n  Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ── 6. Scale (fit on train only) ──────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val),       columns=FEATURE_COLS)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS)

    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    print(f"  Scaler saved → data/processed/scaler.pkl")

    # ── 7. Save splits ────────────────────────────────────────────────────────
    for split_name, X_s, y_s in [
        ("train", X_train_scaled, y_train),
        ("val",   X_val_scaled,   y_val),
        ("test",  X_test_scaled,  y_test),
    ]:
        out = X_s.copy()
        out[LABEL_COL] = y_s.values
        out.to_csv(PROCESSED_DIR / f"{split_name}.csv", index=False)
        print(f"  Saved data/processed/{split_name}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()