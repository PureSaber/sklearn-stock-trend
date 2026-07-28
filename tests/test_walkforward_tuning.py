"""Tests for walk-forward with per-fold tuning."""

import pandas as pd


def test_walkforward_tune_per_fold(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    close = pd.Series(range(100, 400), dtype=float)
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

    monkeypatch.setattr("stock_trend.walkforward.load_or_fetch", lambda *a, **k: ohlcv)
    monkeypatch.setattr("stock_trend.walkforward.load_or_fetch_benchmark", lambda *a, **k: None)

    tune_calls = []

    def fake_run_tuning(X_train, y_train, tuning_cfg, model_cfg):
        tune_calls.append(len(X_train))
        return {"max_depth": 4, "n_estimators": 10, "min_samples_leaf": 5}, 0.45

    monkeypatch.setattr("stock_trend.walkforward.run_tuning", fake_run_tuning)

    from stock_trend.walkforward import run_walkforward

    config = {
        "symbol": "000001.SZ",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "label": {"forward_days": 5, "threshold": 0.0},
        "features": {"ma_windows": [5, 20], "rsi_window": 14, "use_macd": False},
        "model": {
            "type": "random_forest",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 10,
            "max_depth": 4,
            "min_samples_leaf": 5,
        },
        "tuning": {
            "enabled": True,
            "n_splits": 2,
            "param_grid": {"max_depth": [4], "n_estimators": [10], "min_samples_leaf": [5]},
        },
        "backtest": {"signal_mode": "hard", "save_plot": False},
        "walkforward": {"n_splits": 3, "min_train_size": 80, "tune_per_fold": True},
    }

    result = run_walkforward(config, tmp_path / "outputs")

    assert len(tune_calls) > 0
    assert result["summary"]["tune_per_fold"] is True
    assert "tuning_best_params" in result["summary"]["folds"][0]
