"""Batch experiment runner for comparing label/threshold/feature variants."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from stock_trend.backtest import run_backtest
from stock_trend.config import load_config
from stock_trend.evaluate import run_evaluation
from stock_trend.train import run_training


def _experiment_row(name: str, config: dict, train_m: dict, eval_m: dict, bt: dict) -> dict:
    summary = bt["summary"]
    strat = summary["strategy"]
    bh = summary["buy_and_hold"]
    row = {
        "experiment": name,
        "label_threshold": config.get("label", {}).get("threshold", 0.0),
        "proba_threshold": config.get("backtest", {}).get("proba_threshold"),
        "signal_mode": config.get("backtest", {}).get("signal_mode", "hard"),
        "use_macd": config.get("features", {}).get("use_macd", True),
        "test_accuracy": train_m["test_accuracy"],
        "tuning_score": train_m.get("tuning_best_score"),
        "strategy_return": strat["total_return"],
        "buy_hold_return": bh["total_return"],
        "excess_vs_bh": summary["excess_total_return"],
        "strategy_sharpe": strat["sharpe_ratio"],
        "strategy_max_drawdown": strat["max_drawdown"],
    }
    if "benchmark" in summary:
        row["benchmark_return"] = summary["benchmark"]["total_return"]
        row["excess_vs_benchmark"] = summary["excess_vs_benchmark"]
    if eval_m.get("classification_report"):
        row["f1_macro"] = eval_m["classification_report"].get("macro avg", {}).get("f1-score")
    return row


def run_single_experiment(name: str, config: dict) -> dict:
    """Train, evaluate, and backtest one experiment configuration."""
    output_dir = Path(config.get("output_dir", f"outputs/experiments/{name}"))
    config = deepcopy(config)
    config["output_dir"] = str(output_dir)

    print(f"\n=== Experiment: {name} ===")
    train_metrics = run_training(config, output_dir)
    eval_metrics = run_evaluation(config, output_dir)
    backtest_result = run_backtest(config, output_dir)

    row = _experiment_row(name, config, train_metrics, eval_metrics, backtest_result)
    print(
        f"  test_acc={row['test_accuracy']:.3f} | "
        f"strategy={row['strategy_return']:.2%} | "
        f"excess_vs_bh={row['excess_vs_bh']:.2%}"
    )
    return row


def run_experiments(config_paths: list[Path], report_dir: Path) -> pd.DataFrame:
    """Run a list of experiment configs and save comparison report."""
    rows = []
    for path in config_paths:
        config = load_config(path)
        name = path.stem
        rows.append(run_single_experiment(name, config))

    df = pd.DataFrame(rows)
    sort_col = "excess_vs_benchmark" if "excess_vs_benchmark" in df.columns else "excess_vs_bh"
    df = df.sort_values(sort_col, ascending=False, na_position="last")
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "experiment_comparison.csv", index=False)

    best = df.iloc[0].to_dict()
    report = {
        "best_experiment": best["experiment"],
        "ranking_metric": "excess_vs_benchmark",
        "results": rows,
    }
    with (report_dir / "experiment_comparison.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch strategy experiments")
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs/experiments"),
        help="Directory containing experiment YAML files",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/experiments"),
        help="Directory for comparison reports",
    )
    args = parser.parse_args()

    config_paths = sorted(args.configs_dir.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No experiment configs found in {args.configs_dir}")

    print(f"Running {len(config_paths)} experiments...")
    df = run_experiments(config_paths, args.report_dir)

    print("\n=== Comparison (sorted by excess_vs_benchmark) ===")
    cols = [
        "experiment",
        "label_threshold",
        "proba_threshold",
        "strategy_return",
        "buy_hold_return",
        "excess_vs_benchmark",
        "strategy_sharpe",
    ]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nReport saved to {args.report_dir / 'experiment_comparison.csv'}")
    print(f"Best experiment: {df.iloc[0]['experiment']}")


if __name__ == "__main__":
    main()
