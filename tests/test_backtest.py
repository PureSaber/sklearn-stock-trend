"""Tests for backtest logic."""

import numpy as np
import pandas as pd

from stock_trend.backtest import (
    build_signal_frame,
    compute_performance_metrics,
    generate_signals,
    run_backtest,
)


def _sample_ohlcv(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], dtype=float)
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


def test_build_signal_frame_applies_position_with_lag():
    ohlcv = _sample_ohlcv()
    dates = ohlcv["date"].iloc[2:]
    signals = np.array([1, 1, 0, 0, 1, 1, 1, 1])

    frame = build_signal_frame(dates, ohlcv, signals)

    assert len(frame) == len(dates)
    assert frame["position"].iloc[0] == 0
    assert frame["position"].iloc[1] == signals[0]
    assert frame["strategy_return"].iloc[2] == frame["daily_return"].iloc[2] * signals[1]


def test_generate_signals_probability():
    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])

    X = pd.DataFrame({"a": [0, 1, 2, 3]})
    signals = generate_signals(
        MockModel(),
        X,
        {"signal_mode": "probability", "proba_threshold": 0.55},
    )
    assert list(signals) == [1, 1, 0, 1]


def test_build_signal_frame_with_benchmark():
    ohlcv = _sample_ohlcv()
    dates = ohlcv["date"]
    signals = np.ones(len(dates), dtype=int)
    benchmark = ohlcv[["date", "close"]].copy()
    benchmark["close"] = benchmark["close"] * 0.5

    frame = build_signal_frame(dates, ohlcv, signals, benchmark=benchmark)
    assert "benchmark_return" in frame.columns
    assert frame["benchmark_return"].notna().sum() > 0


def test_apply_transaction_costs_reduces_return():
    from stock_trend.backtest import apply_transaction_costs

    frame = pd.DataFrame(
        {
            "position": [0.0, 1.0, 1.0, 0.0],
            "strategy_return": [0.0, 0.01, 0.02, 0.0],
        }
    )
    adjusted = apply_transaction_costs(frame, commission_rate=0.001, slippage=0.001)
    assert adjusted["strategy_return"].iloc[1] < 0.01
    assert adjusted["strategy_return"].iloc[3] < 0.0


def test_compute_performance_metrics_positive_trend():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015])
    metrics = compute_performance_metrics(returns)

    assert metrics["total_return"] > 0
    assert metrics["max_drawdown"] <= 0
    assert "sharpe_ratio" in metrics


def test_run_backtest_with_saved_model(tmp_path, monkeypatch):
    import json

    import joblib
    from sklearn.dummy import DummyClassifier

    from stock_trend.dataset import DatasetSplit, build_labeled_frame, time_series_split
    from stock_trend.features import get_feature_columns

    config = {
        "symbol": "000001.SZ",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "label": {"forward_days": 5, "threshold": 0.0},
        "features": {"ma_windows": [5, 20, 60], "rsi_window": 14, "use_macd": True},
        "model": {"type": "random_forest", "test_size": 0.2, "random_state": 42},
        "backtest": {"signal_mode": "hard", "save_plot": False},
        "output_dir": str(tmp_path / "outputs"),
    }

    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    close = pd.Series(range(100, 220), dtype=float)
    ohlcv = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        }
    )
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
        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_columns=feature_columns,
            ohlcv=ohlcv,
            test_dates=test_dates,
        )

    monkeypatch.setattr("stock_trend.backtest.prepare_dataset", fake_prepare_dataset)

    result = run_backtest(config, output_dir)

    assert (output_dir / "backtest_equity.csv").exists()
    assert (output_dir / "backtest_summary.yaml").exists()
    assert "strategy" in result["summary"]
    assert "buy_and_hold" in result["summary"]
