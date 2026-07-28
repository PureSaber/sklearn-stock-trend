"""Walk-forward out-of-sample evaluation with rolling retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import TimeSeriesSplit

from stock_trend.backtest import (
    apply_transaction_costs,
    build_signal_frame,
    compute_performance_metrics,
    generate_signals,
    save_equity_plot,
)
from stock_trend.config import load_config
from stock_trend.data import load_or_fetch, load_or_fetch_benchmark
from stock_trend.dataset import build_labeled_frame
from stock_trend.features import get_feature_columns
from stock_trend.models import get_model
from stock_trend.report import write_html_report
from stock_trend.signals import build_signal_frame as build_proba_frame
from stock_trend.signals import default_signals_path, export_proba_signals
from stock_trend.tuning import run_tuning


def run_walkforward(config: dict, output_dir: Path | None = None) -> dict:
    """Train on expanding windows, stitch OOS predictions, and backtest."""
    base = output_dir or Path(config.get("output_dir", "outputs"))
    out = base / "walkforward"
    wf_cfg = config.get("walkforward", {})
    tuning_cfg = config.get("tuning", {})
    backtest_cfg = config.get("backtest", {})
    model_cfg = dict(config["model"])

    n_splits = wf_cfg.get("n_splits", 5)
    min_train_size = wf_cfg.get("min_train_size", 200)
    tune_per_fold = wf_cfg.get("tune_per_fold", False)

    ohlcv = load_or_fetch(
        symbol=config["symbol"],
        start_date=config["start_date"],
        end_date=config["end_date"],
    )
    frame = build_labeled_frame(ohlcv, config)
    feature_columns = get_feature_columns(frame)

    X = frame[feature_columns]
    y = frame["label"].astype(int)
    dates = frame["date"].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < min_train_size:
            continue

        fold_model_cfg = dict(model_cfg)
        fold_info: dict = {
            "fold": fold,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
        }

        if tune_per_fold and tuning_cfg.get("enabled", False):
            best_params, best_score = run_tuning(
                X.iloc[train_idx],
                y.iloc[train_idx],
                tuning_cfg,
                fold_model_cfg,
            )
            fold_model_cfg.update(best_params)
            fold_info["tuning_best_params"] = best_params
            fold_info["tuning_best_score"] = best_score

        model = get_model(fold_model_cfg["type"], fold_model_cfg)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        X_test_fold = X.iloc[test_idx]
        preds = generate_signals(model, X_test_fold, backtest_cfg)
        fold_dates = dates.iloc[test_idx]
        fold_info["test_start"] = str(fold_dates.iloc[0].date())
        fold_info["test_end"] = str(fold_dates.iloc[-1].date())

        oos_parts.append(
            pd.DataFrame({"date": fold_dates, "signal": preds, "fold": fold})
        )
        fold_metrics.append(fold_info)

    if not oos_parts:
        raise ValueError(
            f"No walk-forward folds produced; try lowering min_train_size ({min_train_size}) "
            f"or reducing n_splits ({n_splits})."
        )

    oos = pd.concat(oos_parts, ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    oos = oos.sort_values("date").reset_index(drop=True)

    signals_cfg = config.get("signals", {})
    if signals_cfg.get("export", False) and oos_parts:
        proba_parts: list[pd.DataFrame] = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < min_train_size:
                continue
            fold_model_cfg = dict(model_cfg)
            if tune_per_fold and tuning_cfg.get("enabled", False):
                best_params, _ = run_tuning(
                    X.iloc[train_idx], y.iloc[train_idx], tuning_cfg, fold_model_cfg
                )
                fold_model_cfg.update(best_params)
            model = get_model(fold_model_cfg["type"], fold_model_cfg)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba_parts.append(
                build_proba_frame(
                    dates.iloc[test_idx],
                    config["symbol"],
                    model,
                    X.iloc[test_idx],
                    backtest_cfg,
                    model_type=fold_model_cfg["type"],
                    fold=fold,
                )
            )
        if proba_parts:
            export_proba_signals(
                pd.concat(proba_parts, ignore_index=True),
                default_signals_path(out / "walkforward"),
            )

    benchmark_df = None
    benchmark_symbol = backtest_cfg.get("benchmark")
    if benchmark_symbol:
        benchmark_df = load_or_fetch_benchmark(
            benchmark_symbol,
            oos["date"].min().strftime("%Y-%m-%d"),
            oos["date"].max().strftime("%Y-%m-%d"),
        )

    signal_frame = build_signal_frame(
        oos["date"],
        ohlcv,
        oos["signal"].to_numpy(),
        benchmark=benchmark_df,
    )
    signal_frame = apply_transaction_costs(
        signal_frame,
        commission_rate=backtest_cfg.get("commission_rate", 0.0),
        slippage=backtest_cfg.get("slippage", 0.0),
    )

    strategy_metrics = compute_performance_metrics(signal_frame["strategy_return"])
    buy_hold_metrics = compute_performance_metrics(signal_frame["buy_hold_return"])

    equity = signal_frame[["date", "close", "signal", "position"]].copy()
    equity["strategy_equity"] = (1 + signal_frame["strategy_return"].fillna(0)).cumprod()
    equity["buy_hold_equity"] = (1 + signal_frame["buy_hold_return"].fillna(0)).cumprod()
    if benchmark_df is not None:
        equity["benchmark_equity"] = (1 + signal_frame["benchmark_return"].fillna(0)).cumprod()

    out.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out / "walkforward_equity.csv", index=False)
    oos.to_csv(out / "walkforward_predictions.csv", index=False)

    if backtest_cfg.get("save_plot", True):
        save_equity_plot(equity, out / "walkforward_equity.png")

    summary = {
        "symbol": config["symbol"],
        "model_type": model_cfg["type"],
        "method": "walk_forward_expanding",
        "tune_per_fold": tune_per_fold,
        "n_splits": n_splits,
        "oos_days": int(len(oos)),
        "folds": fold_metrics,
        "strategy": strategy_metrics,
        "buy_and_hold": buy_hold_metrics,
        "excess_total_return": strategy_metrics["total_return"] - buy_hold_metrics["total_return"],
    }
    if benchmark_df is not None:
        benchmark_metrics = compute_performance_metrics(signal_frame["benchmark_return"])
        summary["benchmark"] = benchmark_metrics
        summary["excess_vs_benchmark"] = (
            strategy_metrics["total_return"] - benchmark_metrics["total_return"]
        )

    with (out / "walkforward_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    write_html_report(out, title=f"Walk-forward — {config['symbol']}")

    return {"summary": summary, "equity": equity, "oos": oos}


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))

    print(f"Walk-forward evaluation for {config['symbol']}")
    wf_cfg = config.get("walkforward", {})
    if wf_cfg.get("tune_per_fold"):
        print("Per-fold hyperparameter tuning enabled")
    result = run_walkforward(config, output_dir)
    summary = result["summary"]

    strat = summary["strategy"]
    bh = summary["buy_and_hold"]
    print(f"OOS days: {summary['oos_days']} | Folds: {len(summary['folds'])}")
    print(f"Strategy total return:   {strat['total_return']:.2%}")
    print(f"Buy & hold total return: {bh['total_return']:.2%}")
    if "benchmark" in summary:
        print(f"Benchmark total return:  {summary['benchmark']['total_return']:.2%}")
        print(f"Excess vs benchmark:     {summary['excess_vs_benchmark']:.2%}")
    print(f"Strategy Sharpe:         {strat['sharpe_ratio']:.2f}")
    wf_dir = output_dir / "walkforward"
    print(f"\nResults saved to {wf_dir}")


if __name__ == "__main__":
    main()
