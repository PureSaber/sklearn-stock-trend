"""Tests for walk-forward evaluation."""

import pandas as pd

from stock_trend.walkforward import run_walkforward


def test_run_walkforward_synthetic(tmp_path, monkeypatch):
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

    def fake_load_or_fetch(symbol, start_date, end_date, cache_dir=None, use_cache=True):
        return ohlcv

    monkeypatch.setattr("stock_trend.walkforward.load_or_fetch", fake_load_or_fetch)
    monkeypatch.setattr("stock_trend.walkforward.load_or_fetch_benchmark", lambda *a, **k: None)

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
        "backtest": {"signal_mode": "hard", "save_plot": False},
        "walkforward": {"n_splits": 3, "min_train_size": 80},
    }

    out = tmp_path / "outputs"
    result = run_walkforward(config, out)

    assert result["summary"]["oos_days"] > 0
    assert (out / "walkforward" / "walkforward_equity.csv").exists()
    assert (out / "walkforward" / "walkforward_summary.yaml").exists()
