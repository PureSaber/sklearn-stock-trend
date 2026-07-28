"""Tests for batch walk-forward runner."""

from stock_trend.multisymbol_walkforward import run_multisymbol_walkforward


def test_run_multisymbol_walkforward_mock(tmp_path, monkeypatch):
    def fake_wf(config, output_dir=None):
        return {
            "summary": {
                "oos_days": 100,
                "tune_per_fold": True,
                "strategy": {"total_return": 0.1, "sharpe_ratio": 0.5, "max_drawdown": -0.05},
                "buy_and_hold": {"total_return": 0.05},
                "excess_total_return": 0.05,
                "benchmark": {"total_return": 0.08},
                "excess_vs_benchmark": 0.02,
            }
        }

    monkeypatch.setattr("stock_trend.multisymbol_walkforward.run_walkforward", fake_wf)

    config = {"symbol": "000001.SZ", "model": {"type": "xgboost"}}
    df = run_multisymbol_walkforward(
        config,
        ["000001.SZ", "000858.SZ"],
        tmp_path / "wf",
        delay_seconds=0,
    )

    assert len(df) == 2
    assert (tmp_path / "wf" / "multisymbol_walkforward.csv").exists()
