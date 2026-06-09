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