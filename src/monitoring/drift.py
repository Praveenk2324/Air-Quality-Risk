"""
src/monitoring/drift.py
-----------------------
Generate an Evidently data + prediction drift report using modern API.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from src.config import FEATURE_COLS, LABEL_COL

REPORTS_DIR = Path("reports")


def load_reference() -> pd.DataFrame:
    return pd.read_csv("data/processed/train.csv")


def load_current() -> pd.DataFrame:
    return pd.read_csv("data/processed/val.csv")


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: Path,
    drift_threshold: float = 0.15,
) -> bool:
    """
    Generate HTML report using Presets. 
    Returns True if significant drift detected.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(
        metrics=[
            DataDriftPreset(drift_share=drift_threshold),
        ]
    )
    
    # 1. Capture the snapshot returned by .run()
    snapshot = report.run(
        reference_data=reference[FEATURE_COLS + [LABEL_COL]],
        current_data=current[FEATURE_COLS + [LABEL_COL]],
    )
    
    # 2. Call save_html on the snapshot, not the report
    snapshot.save_html(str(out_path))
    print(f"Drift report saved → {out_path}")

    # 3. Use .dict() on the snapshot to extract the values
    result = snapshot.dict()
    drift_result = result["metrics"][0]["result"]
    dataset_drift = drift_result["dataset_drift"]
    drift_share = drift_result["share_of_drifted_columns"]
    
    print(f"Dataset drift detected: {dataset_drift}")
    print(f"Share of drifted features: {drift_share:.2%}")

    return dataset_drift


def main(threshold: float = 0.15, out: str = "reports/drift_report.html"):
    reference = load_reference()
    current = load_current()
    drift_detected = run_drift_report(reference, current, Path(out), threshold)

    if drift_detected:
        print("\nDrift threshold exceeded — consider triggering retraining.")
        sys.exit(1)
    else:
        print("\nNo significant drift detected.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--out", default="reports/drift_report.html")
    args = parser.parse_args()
    main(args.threshold, args.out)