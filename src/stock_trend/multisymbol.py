"""Run train/backtest pipeline across multiple symbols."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from stock_trend.backtest import run_backtest
from stock_trend.config import load_config
from stock_trend.train import run_training


def _symbol_row(symbol: str, config: dict, train_m: dict, bt: dict) -> dict:
    summary = bt["summary"]
    strat = summary["strategy"]
    bh = summary["buy_and_hold"]
    row = {
        "symbol": symbol,
        "model_type": config.get("model", {}).get("type", "unknown"),
        "proba_threshold": config.get("backtest", {}).get("proba_threshold"),
        "test_accuracy": train_m["test_accuracy"],
        "strategy_return": strat["total_return"],
        "buy_hold_return": bh["total_return"],
        "excess_vs_bh": summary["excess_total_return"],
        "strategy_sharpe": strat["sharpe_ratio"],
        "strategy_max_drawdown": strat["max_drawdown"],
    }
    if "benchmark" in summary:
        row["benchmark_return"] = summary["benchmark"]["total_return"]
        row["excess_vs_benchmark"] = summary["excess_vs_benchmark"]
    return row


def run_multisymbol(
    base_config: dict,
    symbols: list[str],
    output_root: Path,
    delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Train and backtest the same strategy on multiple symbols."""
    rows = []
    for i, symbol in enumerate(symbols):
        if i > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        config = deepcopy(base_config)
        config["symbol"] = symbol
        safe_name = symbol.replace(".", "_")
        out_dir = output_root / safe_name
        config["output_dir"] = str(out_dir)

        print(f"\n=== Symbol: {symbol} ===")
        try:
            train_m = run_training(config, out_dir)
            bt = run_backtest(config, out_dir)
            row = _symbol_row(symbol, config, train_m, bt)
            rows.append(row)
            print(
                f"  test_acc={row['test_accuracy']:.3f} | "
                f"strategy={row['strategy_return']:.2%} | "
                f"excess_vs_benchmark={row.get('excess_vs_benchmark', float('nan')):.2%}"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"symbol": symbol, "error": str(exc)})

    if not rows:
        raise RuntimeError("All symbols failed")

    df = pd.DataFrame(rows)
    if "excess_vs_benchmark" in df.columns:
        sort_col = "excess_vs_benchmark"
    elif "excess_vs_bh" in df.columns:
        sort_col = "excess_vs_bh"
    else:
        sort_col = None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "multisymbol_comparison.csv", index=False)
    with (output_root / "multisymbol_comparison.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"symbols": symbols, "results": rows}, f, sort_keys=False)

    return df


def load_symbols(path: Path) -> list[str]:
    """Load symbol list from YAML (symbols: [...]) or plain text (one per line)."""
    if path.suffix in {".yaml", ".yml"}:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "symbols" in data:
            return list(data["symbols"])
        if isinstance(data, list):
            return data
        raise ValueError(f"Invalid symbols file format: {path}")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch train/backtest across symbols")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--symbols",
        type=Path,
        default=Path("configs/symbols.yaml"),
        help="YAML or text file listing symbols",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/multisymbol"),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between symbols (reduce API rate limits)",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    symbols = load_symbols(args.symbols)
    if not symbols:
        raise ValueError(f"No symbols found in {args.symbols}")

    print(f"Running multisymbol batch ({len(symbols)} symbols)...")
    df = run_multisymbol(base_config, symbols, args.output_dir, delay_seconds=args.delay)

    print("\n=== Multisymbol Comparison ===")
    cols = ["symbol", "strategy_return", "buy_hold_return", "excess_vs_benchmark", "strategy_sharpe"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nReport saved to {args.output_dir / 'multisymbol_comparison.csv'}")


if __name__ == "__main__":
    main()
