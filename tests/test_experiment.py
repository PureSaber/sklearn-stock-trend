"""Tests for batch experiment runner."""


import yaml

from stock_trend.experiment import _experiment_row, run_experiments


def test_experiment_row_structure():
    config = {
        "label": {"threshold": 0.02},
        "backtest": {"proba_threshold": 0.6, "signal_mode": "probability"},
        "features": {"use_macd": False},
    }
    train_m = {"test_accuracy": 0.5, "tuning_best_score": 0.45}
    eval_m = {"classification_report": {"macro avg": {"f1-score": 0.4}}}
    bt = {
        "summary": {
            "strategy": {"total_return": 0.1, "sharpe_ratio": 0.5, "max_drawdown": -0.05},
            "buy_and_hold": {"total_return": 0.2},
            "excess_total_return": -0.1,
            "benchmark": {"total_return": 0.15},
            "excess_vs_benchmark": -0.05,
        }
    }
    row = _experiment_row("test_exp", config, train_m, eval_m, bt)
    assert row["experiment"] == "test_exp"
    assert row["label_threshold"] == 0.02
    assert row["proba_threshold"] == 0.6
    assert row["excess_vs_benchmark"] == -0.05


def test_run_experiments_with_mock(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_data = {
        "symbol": "000001.SZ",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "label": {"forward_days": 5, "threshold": 0.0},
        "features": {"ma_windows": [5], "rsi_window": 14, "use_macd": False},
        "model": {"type": "random_forest", "test_size": 0.2, "random_state": 42},
        "backtest": {"signal_mode": "hard", "save_plot": False},
        "tuning": {"enabled": False},
        "output_dir": "outputs/x",
    }
    with (configs_dir / "a.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    def fake_run_training(config, output_dir):
        return {
            "test_accuracy": 0.5,
            "tuning_best_score": None,
            "train_samples": 10,
            "test_samples": 2,
            "train_accuracy": 0.6,
            "feature_count": 3,
            "classification_report_text": "",
        }

    def fake_run_evaluation(config, output_dir=None):
        return {
            "accuracy": 0.5,
            "test_samples": 2,
            "classification_report": {"macro avg": {"f1-score": 0.4}},
            "classification_report_text": "",
            "confusion_matrix": [[1, 0], [0, 1]],
        }

    def fake_run_backtest(config, output_dir=None):
        return {
            "summary": {
                "strategy": {"total_return": 0.05, "sharpe_ratio": 0.3, "max_drawdown": -0.02},
                "buy_and_hold": {"total_return": 0.1},
                "excess_total_return": -0.05,
            }
        }

    monkeypatch.setattr("stock_trend.experiment.run_training", fake_run_training)
    monkeypatch.setattr("stock_trend.experiment.run_evaluation", fake_run_evaluation)
    monkeypatch.setattr("stock_trend.experiment.run_backtest", fake_run_backtest)

    report_dir = tmp_path / "reports"
    df = run_experiments([configs_dir / "a.yaml"], report_dir)

    assert len(df) == 1
    assert (report_dir / "experiment_comparison.csv").exists()
    assert (report_dir / "experiment_comparison.yaml").exists()
