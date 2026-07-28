"""CLI entry point smoke tests."""

import importlib
import subprocess
import sys


def test_console_scripts_importable():
    modules = [
        "stock_trend.train",
        "stock_trend.evaluate",
        "stock_trend.backtest",
        "stock_trend.walkforward",
        "stock_trend.multisymbol",
        "stock_trend.experiment",
    ]
    for mod in modules:
        importlib.import_module(mod)


def test_st_train_help():
    result = subprocess.run(
        [sys.executable, "-m", "stock_trend.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Train stock trend" in result.stdout
