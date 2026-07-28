"""Tests for model evaluation pipeline."""

import json

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from stock_trend.dataset import build_labeled_frame, time_series_split
from stock_trend.evaluate import evaluate_model, load_model_artifacts, run_evaluation
from stock_trend.features import get_feature_columns


def _sample_config() -> dict:
    return {
        "symbol": "000001.SZ",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "label": {"forward_days": 5, "threshold": 0.0},
        "features": {
            "rsi_window": 14,
            "ma_windows": [5, 20, 60],
            "use_macd": True,
        },
        "model": {"type": "random_forest", "test_size": 0.2, "random_state": 42},
        "output_dir": "outputs",
    }


def _sample_ohlcv(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_evaluate_model_returns_metrics():
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1.0, 0.5, 0.2, 0.1]})
    y = pd.Series([0, 0, 1, 1])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    assert "accuracy" in metrics
    assert "classification_report_text" in metrics
    assert metrics["test_samples"] == 4


def test_run_evaluation_with_saved_model(tmp_path, monkeypatch):
    config = _sample_config()
    ohlcv = _sample_ohlcv()
    frame = build_labeled_frame(ohlcv, config)
    feature_columns = get_feature_columns(frame)
    X_train, X_test, y_train, y_test, test_dates = time_series_split(
        frame, feature_columns, config["model"]["test_size"]
    )

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    joblib.dump(model, output_dir / "model.pkl")
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f)

    def fake_prepare_dataset(cfg, cache_dir=None):
        from stock_trend.dataset import DatasetSplit

        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_columns=feature_columns,
            ohlcv=ohlcv,
            test_dates=test_dates,
        )

    monkeypatch.setattr("stock_trend.evaluate.prepare_dataset", fake_prepare_dataset)

    metrics = run_evaluation(config, output_dir)

    assert metrics["test_samples"] == len(y_test)
    assert (output_dir / "eval_report.yaml").exists()
    assert (output_dir / "eval_report.txt").exists()


def test_load_model_artifacts_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model not found"):
        load_model_artifacts(tmp_path)
