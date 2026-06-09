import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = Path('data/raw/latest.csv')
PROCESSED_DIR = Path('data/processed')

PM25_BREAKPOINTS = [
    (0, 12.0, 0),    # Good
    (12.1, 35.4, 1), # Moderate
    (35.5, 150.4, 2),# Unhealthy
    (150.5, np.inf, 3),
]

MIN_SAMPLES_PER_CLASS = 10

def assign_risk_tier(pm25: float) -> int:
    for lo, hi, tier in PM25_BREAKPOINTS:
        if lo <= pm25 <= hi:
            return tier
    return 3

def pivot_wide(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['hour'] = df['timestamp'].dt.floor('h')
    df = df[df['parameter'].isin(['pm25', 'no2', 'o3'])]

    wide = (
        df.groupby(['location_id', 'hour', 'parameter'])['value']
        .median()
        .unstack('parameter')
        .reset_index()
    )
    wide.columns.name = None
    return wide

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df['hour'].dt.hour
    month = df['hour'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    return df

def safe_split(X, y, test_size: float, random_state: int = 42):
    class_count = y.value_counts()
    min_count = class_count.min()

    if min_count < MIN_SAMPLES_PER_CLASS:
        print(
            f"  WARNING: class {class_count.idxmin()} has only {min_count} samples. "
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

    wide = pivot_wide(raw)
    print(f"   After pivot: {wide.shape}")

    if wide.empty:
        raise RuntimeError(
            "Pivot produced an empty DataFrame. "
            "Check that latest.csv contains pm25/no2/o3 rows with valid values."
        )
    
    before = len(wide)
    wide = wide.dropna(subset=["pm25"])
    print(f"  Dropped {before - len(wide)} rows missing PM2.5 → {len(wide)} remain")
 
    if len(wide) < 50:
        print(
            f"  WARNING: only {len(wide)} labelled rows. "
            f"Consider fetching more data (increase --measurements-per-sensor)."
        )
 
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
    
    wide = add_time_features(wide)
    wide[LABEL_COL] = wide["pm25"].apply(assign_risk_tier)
 
    print(f"\n  Label distribution:")
    print(wide[LABEL_COL].value_counts().sort_index().to_string())
 
    features = wide[FEATURE_COLS + [LABEL_COL]].dropna()
    print(f"\n  Final feature matrix: {features.shape}")