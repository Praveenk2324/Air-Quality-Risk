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
    min_count = class