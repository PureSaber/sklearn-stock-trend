"""Tests for HTML report generation."""

from pathlib import Path

import yaml

from stock_trend.report import write_html_report


def test_write_html_report(tmp_path: Path):
    (tmp_path / "train_meta.yaml").write_text(
        yaml.safe_dump(
            {"symbol": "000001.SZ", "model_type": "random_forest", "train_accuracy": 0.7, "test_accuracy": 0.6}
        ),
        encoding="utf-8",
    )
    (tmp_path / "backtest_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "strategy": {"total_return": 0.05, "sharpe_ratio": 1.2},
                "buy_and_hold": {"total_return": 0.02},
            }
        ),
        encoding="utf-8",
    )

    report = write_html_report(tmp_path, title="Test Report")
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert "000001.SZ" in html
    assert "random_forest" in html
