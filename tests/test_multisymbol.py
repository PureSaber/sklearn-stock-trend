"""Tests for multisymbol batch runner."""

from stock_trend.multisymbol import load_symbols, run_multisymbol


def test_load_symbols_yaml(tmp_path):
    path = tmp_path / "symbols.yaml"
    path.write_text("symbols:\n  - '000001.SZ'\n  - '600519.SH'\n", encoding="utf-8")
    assert load_symbols(path) == ["000001.SZ", "600519.SH"]


def test_load_symbols_text(tmp_path):
    path = tmp_path / "symbols.txt"
    path.write_text("000001.SZ\n# comment\n600519.SH\n", encoding="utf-8")
    assert load_symbols(path) == ["000001.SZ", "600519.SH"]


def test_run_multisymbol_mock(tmp_path, monkeypatch):
    def fake_train(config, output_dir):
        return {
            "test_accuracy": 0.5,
            "train_accuracy": 0.6,
            "train_samples": 100,
            "test_samples": 20,
            "feature_count": 5,
            "tuning_best_params": None,
            "tuning_best_score": None,
            "classification_report_text": "",
        }

    def fake_backtest(config, output_dir=None):
        return {
            "summary": {
                "strategy": {"total_return": 0.05, "sharpe_ratio": 0.4, "max_drawdown": -0.03},
                "buy_and_hold": {"total_return": 0.1},
                "excess_total_return": -0.05,
                "benchmark": {"total_return": 0.08},
                "excess_vs_benchmark": -0.03,
            }
        }

    monkeypatch.setattr("stock_trend.multisymbol.run_training", fake_train)
    monkeypatch.setattr("stock_trend.multisymbol.run_backtest", fake_backtest)

    config = {"symbol": "000001.SZ", "model": {"type": "random_forest"}}
    df = run_multisymbol(config, ["000001.SZ", "600519.SH"], tmp_path / "ms")

    assert len(df) == 2
    assert (tmp_path / "ms" / "multisymbol_comparison.csv").exists()
