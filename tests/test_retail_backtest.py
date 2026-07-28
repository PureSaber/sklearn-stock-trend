"""Tests for discrete retail backtest."""

import pandas as pd

from stock_trend.retail_backtest import run_retail_backtest, simulate_retail_long_flat
from stock_trend.trading_costs import CostsConfig


def test_simulate_retail_respects_lot_size():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": [10.0] * 10,
            "signal": [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        }
    )
    costs = CostsConfig(initial_capital=10_000, lot_size=100, min_holding_days=1)
    result = simulate_retail_long_flat(frame, costs)
    assert (result["shares"].max() % 100) == 0 or result["shares"].max() == 0


def test_run_retail_backtest_returns_metrics():
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    close = [10.0 + i * 0.1 for i in range(20)]
    frame = pd.DataFrame({"date": dates, "close": close, "signal": [0] + [1] * 18 + [0]})
    backtest_cfg = {
        "retail_mode": True,
        "initial_capital": 10_000,
        "lot_size": 100,
        "min_commission": 5.0,
        "stamp_tax": 0.0005,
        "commission_rate": 0.0003,
        "slippage": 0.0001,
        "min_holding_days": 1,
    }
    result = run_retail_backtest(frame, backtest_cfg)
    assert "metrics" in result
    assert "total_return" in result["metrics"]
