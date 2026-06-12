"""
src/serving/api.py
------------------
FastAPI REST scorer for the air quality risk model.

Endpoints:
  POST /predict          — single prediction with SHAP explanation
  POST /predict/batch    — batch predictions (no SHAP, faster)
  GET  /health           — liveness check

Key MLOps pattern:
  All inference runs under torch.no_grad() to disable gradient tracking.
  Without it, PyTorch builds a computation graph on every forward pass —
  wasting memory and adding ~15% latency with zero benefit at inference time.

Run:
    uvicorn src.serving.api:app --reload
"""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import mlflow.pytorch
import numpy as np
import shap
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model.mlp import AirQualityMLP
from src.config import FEATURE_COLS, RISK_LABELS, INPUT_DIM
MODEL_PATH = Path("models/best_model.pt")
SCALER_PATH = Path("data/processed/scaler.pkl")

# Global model state — loaded once at startup
_model: Optional[AirQualityMLP] = None
_scaler = None
_explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler once at startup, release at shutdown."""
    global _model, _scaler, _explainer

    _scaler = joblib.load(SCALER_PATH)

    _model = AirQualityMLP(input_dim=INPUT_DIM, hidden_dims=[64, 32])
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    _model.eval()  # sets BatchNorm and Dropout to inference mode

    # Build SHAP explainer using a small background dataset
    # KernelExplainer works model-agnostically — no need for tree/deep variants
    background = torch.zeros(50, len(FEATURE_COLS))
    _explainer = shap.KernelExplainer(
        lambda x: _model_predict_np(x), background.numpy()
    )

    print("Model and scaler loaded.")
    yield
    _model = None
    _scaler = None


app = FastAPI(
    title="Air Quality Health Risk Scorer",
    version="1.0.0",
    lifespan=lifespan,
)


def _model_predict_np(x_np: np.ndarray) -> np.ndarray:
    """Wrapper for SHAP — converts numpy array to tensor and back."""
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        probs = _model.predict_proba(x_tensor)
    return probs.numpy()


class PollutantReading(BaseModel):
    pm25: float = Field(..., ge=0, description="PM2.5 concentration µg/m³")
    no2: float = Field(..., ge=0, description="NO2 concentration µg/m³")
    o3: float = Field(..., ge=0, description="O3 concentration µg/m³")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    month: int = Field(..., ge=1, le=12, description="Month (1–12)")
    explain: bool = Field(False, description="Include SHAP feature attributions")


class RiskPrediction(BaseModel):
    risk_tier: int
    risk_label: str
    confidence: float
    probabilities: dict[str, float]
    shap_values: Optional[dict[str, float]] = None


def _encode_time(hour: int, month: int) -> list[float]:
    """Cyclical encoding — must match preprocess.py exactly."""
    import math
    return [
        math.sin(2 * math.pi * hour  / 24),
        math.cos(2 * math.pi * hour  / 24),
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
    ]


@app.post("/predict", response_model=RiskPrediction)
def predict(reading: PollutantReading):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    time_features = _encode_time(reading.hour_of_day, reading.month)
    raw_features = np.array(
        [[reading.pm25, reading.no2, reading.o3] + time_features]
    )
    scaled = _scaler.transform(raw_features)
    x_tensor = torch.tensor(scaled, dtype=torch.float32)

    # torch.no_grad() — disables autograd engine for inference
    # This is the pattern interviewers ask about explicitly
    with torch.no_grad():
        probs = _model.predict_proba(x_tensor)[0].numpy()

    tier = int(probs.argmax())
    confidence = float(probs[tier])

    shap_vals = None
    if reading.explain:
        sv = _explainer.shap_values(scaled)
        # sv shape: (n_classes, n_samples, n_features) — take predicted class
        shap_vals = dict(zip(FEATURE_COLS, sv[tier][0].tolist()))

    return RiskPrediction(
        risk_tier=tier,
        risk_label=RISK_LABELS[tier],
        confidence=confidence,
        probabilities={RISK_LABELS[i]: float(p) for i, p in enumerate(probs)},
        shap_values=shap_vals,
    )


@app.post("/predict/batch")
def predict_batch(readings: list[PollutantReading]):
    """Batch scoring — SHAP disabled for throughput."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = []
    for r in readings:
        tf = _encode_time(r.hour_of_day, r.month)
        rows.append([r.pm25, r.no2, r.o3] + tf)

    scaled = _scaler.transform(np.array(rows))
    x_tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        probs = _model.predict_proba(x_tensor).numpy()

    results = []
    for p in probs:
        tier = int(p.argmax())
        results.append({
            "risk_tier": tier,
            "risk_label": RISK_LABELS[tier],
            "confidence": float(p[tier]),
        })
    return results


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}