"""Tests for probability signal export."""

from pathlib import Path

import numpy as np
import pandas as pd

from stock_trend.signals import build_signal_frame, export_proba_signals, predict_proba_up


class MockModel:
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


def test_predict_proba_up():
    model = MockModel()
    X = pd.DataFrame({"a": [1, 2, 3]})
    proba = predict_proba_up(model, X)
    assert len(proba) == 3
    assert proba[0] == 0.6


def test_build_and_export_signals(tmp_path: Path):
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    X = pd.DataFrame({"f1": range(5)})
    frame = build_signal_frame(
        dates,
        "000001.SZ",
        MockModel(),
        X,
        {"signal_mode": "probability", "proba_threshold": 0.55},
        model_type="random_forest",
    )
    assert list(frame.columns) == ["date", "symbol", "proba_up", "signal", "model_type"]
    assert (frame["signal"] == 1).all()

    out = export_proba_signals(frame, tmp_path / "proba_signals.parquet")
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert len(loaded) == 5
