"""Offline end-to-end pipeline smoke test (no network)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from stock_trend.backtest import run_backtest
from stock_trend.dataset import DatasetSplit
from stock_trend.train import run_training


def _synthetic_config(tmp_path: Path) -> dict:
    return {
        "symbol": "000001.SZ",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "label": {"forward_days": 5, "threshold": 0.0},
        "features": {"rsi_window": 14, "ma_windows": [5, 20], "use_macd": False},
        "tuning": {"enabled": False},
        "model": {
            "type": "random_forest",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 10,
            "max_depth": 4,
            "min_samples_leaf": 5,
        },
        "backtest": {
            "signal_mode": "probability",
            "proba_threshold": 0.5,
            "commission_rate": 0.0,
            "slippage": 0.0,
            "save_plot": False,
        },
        "signals": {"export": True},
        "output_dir": str(tmp_path / "out"),
    }


def _mock_dataset(n: int = 80) -> DatasetSplit:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, n))
    ohlcv = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1e6,
            "return_1d": pd.Series(close).pct_change(),
            "volatility_20d": pd.Series(close).pct_change().rolling(20).std(),
            "ma_5": pd.Series(close).rolling(5).mean(),
            "ma_20": pd.Series(close).rolling(20).mean(),
            "close_ma_5_ratio": pd.Series(close) / pd.Series(close).rolling(5).mean() - 1,
            "close_ma_20_ratio": pd.Series(close) / pd.Series(close).rolling(20).mean() - 1,
            "rsi_14": 50.0,
            "label": (pd.Series(close).shift(-5) / pd.Series(close) - 1 > 0).astype("Int64"),
        }
    )
    ohlcv = ohlcv.dropna().reset_index(drop=True)
    feature_cols = [c for c in ohlcv.columns if c not in {"date", "open", "high", "low", "close", "volume", "label"}]
    split = int(len(ohlcv) * 0.8)
    return DatasetSplit(
        X_train=ohlcv.iloc[:split][feature_cols],
        X_test=ohlcv.iloc[split:][feature_cols],
        y_train=ohlcv.iloc[:split]["label"].astype(int),
        y_test=ohlcv.iloc[split:]["label"].astype(int),
        feature_columns=feature_cols,
        test_dates=ohlcv.iloc[split:]["date"],
        ohlcv=ohlcv,
    )


@patch("stock_trend.train.prepare_dataset")
@patch("stock_trend.backtest.prepare_dataset")
@patch("stock_trend.backtest.load_or_fetch_benchmark")
def test_e2e_train_backtest_offline(mock_bench, mock_bt_ds, mock_train_ds, tmp_path):
    config = _synthetic_config(tmp_path)
    dataset = _mock_dataset()
    mock_train_ds.return_value = dataset
    mock_bt_ds.return_value = dataset
    mock_bench.return_value = None

    out = Path(config["output_dir"])
    metrics = run_training(config, out)
    assert metrics["test_samples"] > 0
    assert (out / "model.pkl").exists()

    result = run_backtest(config, out)
    assert "summary" in result
    assert (out / "proba_signals.parquet").exists()
    assert (out / "latest" / "report.html").exists()

    summary = yaml.safe_load((out / "backtest_summary.yaml").read_text(encoding="utf-8"))
    assert "strategy" in summary
