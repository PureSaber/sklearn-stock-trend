"""Simple vectorized backtest from model predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stock_trend.config import load_config
from stock_trend.data import load_or_fetch_benchmark
from stock_trend.dataset import prepare_dataset
from stock_trend.evaluate import load_model_artifacts
from stock_trend.report import write_html_report
from stock_trend.retail_backtest import run_retail_backtest
from stock_trend.signals import build_signal_frame as build_proba_frame
from stock_trend.signals import default_signals_path, export_proba_signals

from stock_trend.metrics import compute_performance_metrics


def generate_signals(model, X_test: pd.DataFrame, backtest_cfg: dict) -> np.ndarray:
    """Build trading signals from hard labels or probability threshold."""
    mode = backtest_cfg.get("signal_mode", "hard")
    if mode == "probability":
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support predict_proba for probability signal mode.")
        proba = model.predict_proba(X_test)[:, 1]
        threshold = backtest_cfg.get("proba_threshold", 0.5)
        return (proba > threshold).astype(int)
    return model.predict(X_test)


def build_signal_frame(
    dates: pd.Series,
    ohlcv: pd.DataFrame,
    predictions: np.ndarray,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align close prices with same-day signals; position applies from next day."""
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "signal": predictions})
    prices = ohlcv[["date", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    merged = frame.merge(prices, on="date", how="left").sort_values("date")
    if merged["close"].isna().any():
        raise ValueError("Missing close prices for some test dates.")

    merged["daily_return"] = merged["close"].pct_change()
    merged["position"] = merged["signal"].shift(1).fillna(0)
    merged["strategy_return"] = merged["position"] * merged["daily_return"]
    merged["buy_hold_return"] = merged["daily_return"]

    if benchmark is not None:
        bench = benchmark.copy()
        bench["date"] = pd.to_datetime(bench["date"])
        bench = bench.sort_values("date")
        bench["benchmark_return"] = bench["close"].pct_change()
        merged = merged.merge(bench[["date", "benchmark_return"]], on="date", how="left")
    else:
        merged["benchmark_return"] = np.nan

    return merged.reset_index(drop=True)


def apply_transaction_costs(
    frame: pd.DataFrame,
    commission_rate: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """Deduct round-trip costs when position changes (enter or exit)."""
    out = frame.copy()
    if commission_rate == 0.0 and slippage == 0.0:
        return out

    position_change = out["position"].diff().abs().fillna(out["position"].abs())
    per_side_cost = commission_rate + slippage
    trade_cost = position_change * per_side_cost
    out["strategy_return"] = out["strategy_return"] - trade_cost
    return out


def save_equity_plot(equity: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity["date"], equity["strategy_equity"], label="Strategy")
    ax.plot(equity["date"], equity["buy_hold_equity"], label="Buy & Hold")
    if "benchmark_equity" in equity.columns and equity["benchmark_equity"].notna().any():
        ax.plot(equity["date"], equity["benchmark_equity"], label="Benchmark")
    ax.set_title("Equity Curves (Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_backtest(config: dict, output_dir: Path | None = None) -> dict:
    """Backtest long-flat strategy on the test split using a saved model."""
    out = output_dir or Path(config.get("output_dir", "outputs"))
    backtest_cfg = config.get("backtest", {})
    model, saved_columns = load_model_artifacts(out)
    dataset = prepare_dataset(config)

    if saved_columns != dataset.feature_columns:
        raise ValueError(
            "Feature columns mismatch between saved model and current config. Retrain first."
        )

    X_test = dataset.X_test[saved_columns]
    predictions = generate_signals(model, X_test, backtest_cfg)

    signals_cfg = config.get("signals", {})
    if signals_cfg.get("export", False):
        proba_frame = build_proba_frame(
            dataset.test_dates,
            config["symbol"],
            model,
            X_test,
            backtest_cfg,
            model_type=config.get("model", {}).get("type", ""),
        )
        export_proba_signals(proba_frame, default_signals_path(out))

    benchmark_df = None
    benchmark_symbol = backtest_cfg.get("benchmark")
    if benchmark_symbol:
        test_start = pd.to_datetime(dataset.test_dates.min())
        test_end = pd.to_datetime(dataset.test_dates.max())
        benchmark_df = load_or_fetch_benchmark(
            benchmark_symbol,
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        )

    signal_frame = build_signal_frame(
        dataset.test_dates,
        dataset.ohlcv,
        predictions,
        benchmark=benchmark_df,
    )

    if backtest_cfg.get("retail_mode", False):
        retail_result = run_retail_backtest(
            signal_frame[["date", "close", "signal"]],
            backtest_cfg,
        )
        retail_equity = retail_result["equity"]
        strategy_metrics = retail_result["metrics"]
        signal_frame = signal_frame.merge(
            retail_equity[["date", "daily_return", "portfolio_value", "shares"]],
            on="date",
            how="left",
        )
        signal_frame["strategy_return"] = signal_frame["daily_return"].fillna(0.0)
    else:
        signal_frame = apply_transaction_costs(
            signal_frame,
            commission_rate=backtest_cfg.get("commission_rate", 0.0),
            slippage=backtest_cfg.get("slippage", 0.0),
        )
        strategy_metrics = compute_performance_metrics(signal_frame["strategy_return"])

    buy_hold_metrics = compute_performance_metrics(signal_frame["buy_hold_return"])
    benchmark_metrics = None
    if benchmark_df is not None:
        benchmark_metrics = compute_performance_metrics(signal_frame["benchmark_return"])

    equity = signal_frame[["date", "close", "signal", "position"]].copy()
    equity["strategy_equity"] = (1 + signal_frame["strategy_return"].fillna(0)).cumprod()
    equity["buy_hold_equity"] = (1 + signal_frame["buy_hold_return"].fillna(0)).cumprod()
    if benchmark_df is not None:
        equity["benchmark_equity"] = (1 + signal_frame["benchmark_return"].fillna(0)).cumprod()

    out.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out / "backtest_equity.csv", index=False)

    if backtest_cfg.get("save_plot", True):
        save_equity_plot(equity, out / "backtest_equity.png")

    summary = {
        "symbol": config["symbol"],
        "signal_mode": backtest_cfg.get("signal_mode", "hard"),
        "proba_threshold": backtest_cfg.get("proba_threshold"),
        "commission_rate": backtest_cfg.get("commission_rate", 0.0),
        "slippage": backtest_cfg.get("slippage", 0.0),
        "benchmark_symbol": benchmark_symbol,
        "test_period": {
            "start": str(signal_frame["date"].iloc[0].date()),
            "end": str(signal_frame["date"].iloc[-1].date()),
            "days": len(signal_frame),
        },
        "strategy": strategy_metrics,
        "buy_and_hold": buy_hold_metrics,
        "excess_total_return": strategy_metrics["total_return"] - buy_hold_metrics["total_return"],
    }
    if benchmark_metrics is not None:
        summary["benchmark"] = benchmark_metrics
        summary["excess_vs_benchmark"] = (
            strategy_metrics["total_return"] - benchmark_metrics["total_return"]
        )

    with (out / "backtest_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

    write_html_report(out, title=f"Backtest — {config['symbol']}")

    return {"summary": summary, "equity": equity}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest stock trend strategy on test split")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))

    print(f"Backtest strategy for {config['symbol']} (test split)")
    result = run_backtest(config, output_dir)
    summary = result["summary"]

    strat = summary["strategy"]
    bh = summary["buy_and_hold"]
    print(f"Signal mode: {summary['signal_mode']}")
    print(f"Test period: {summary['test_period']['start']} -> {summary['test_period']['end']}")
    print(f"Strategy total return:   {strat['total_return']:.2%}")
    print(f"Buy & hold total return: {bh['total_return']:.2%}")
    print(f"Excess return:           {summary['excess_total_return']:.2%}")
    if "benchmark" in summary:
        bench = summary["benchmark"]
        print(f"Benchmark total return:  {bench['total_return']:.2%}")
        print(f"Excess vs benchmark:     {summary['excess_vs_benchmark']:.2%}")
    print(f"Strategy max drawdown:   {strat['max_drawdown']:.2%}")
    print(f"Strategy Sharpe:         {strat['sharpe_ratio']:.2f}")
    print(f"\nEquity curve saved to {output_dir / 'backtest_equity.csv'}")
    if (output_dir / "backtest_equity.png").exists():
        print(f"Plot saved to {output_dir / 'backtest_equity.png'}")


if __name__ == "__main__":
    main()
