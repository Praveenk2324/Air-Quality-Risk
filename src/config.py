"""
src/config.py
-------------
Single source of truth for feature definitions shared across
preprocess, train, tune, serving, and monitoring.

If you add or remove a feature, change it here only.
"""

FEATURE_COLS = [
    "pm25",
    "no2",
    "o3",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

LABEL_COL = "risk_tier"

RISK_LABELS = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy",
    3: "Hazardous",
}

NUM_CLASSES = len(RISK_LABELS)
INPUT_DIM = len(FEATURE_COLS)