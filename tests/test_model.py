"""
tests/test_model.py
-------------------
Unit tests for the MLP and preprocessing logic.
Run: pytest tests/ -v
"""

import math
import numpy as np
import torch
import pytest

from src.model.mlp import AirQualityMLP
from src.data.preprocess import assign_risk_tier, add_time_features
import pandas as pd


class TestAirQualityMLP:
    def test_output_shape(self):
        model = AirQualityMLP(input_dim=7, hidden_dims=[64, 32], num_classes=4)
        x = torch.randn(16, 7)
        logits = model(x)
        assert logits.shape == (16, 4)

    def test_predict_proba_sums_to_one(self):
        model = AirQualityMLP()
        model.eval()
        x = torch.randn(8, 7)
        with torch.no_grad():
            probs = model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_no_grad_context(self):
        """Verify no gradient is computed during predict_proba."""
        model = AirQualityMLP()
        model.eval()
        x = torch.randn(4, 7)
        with torch.no_grad():
            probs = model.predict_proba(x)
        assert probs.requires_grad is False

    def test_dropout_disabled_in_eval(self):
        """Dropout should be deterministic in eval mode."""
        model = AirQualityMLP(dropout=0.9)
        model.eval()
        x = torch.randn(32, 7)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


class TestPreprocessing:
    def test_risk_tier_good(self):
        assert assign_risk_tier(5.0) == 0

    def test_risk_tier_moderate(self):
        assert assign_risk_tier(20.0) == 1

    def test_risk_tier_unhealthy(self):
        assert assign_risk_tier(80.0) == 2

    def test_risk_tier_hazardous(self):
        assert assign_risk_tier(300.0) == 3

    def test_time_features_range(self):
        df = pd.DataFrame({
            "hour": pd.to_datetime(["2024-01-15 08:00:00+00:00",
                                    "2024-06-21 20:00:00+00:00"])
        })
        df["hour"] = pd.to_datetime(df["hour"], utc=True)
        result = add_time_features(df)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert col in result.columns
            assert result[col].between(-1.0, 1.0).all()