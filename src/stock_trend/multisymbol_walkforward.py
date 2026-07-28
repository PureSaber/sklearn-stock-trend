"""Batch walk-forward evaluation across multiple symbols."""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from stock_trend.config import load_config
from stock_trend.multisymbol import load_symbols
from stock_trend.walkforward import run_walkforward


def _wf_row(symbol: str, config: dict, result: dict) -> dict:
    summary = result["summary"]
    strat = summary["strategy"]
    bh = summary["buy_and_hold"]
    row = {
        "symbol": symbol,
        "model_type": config.get("model", {}).get("type", "unknown"),
        "oos_days": summary.get("oos_days"),
        "tune_per_fold": summary.get("tune_per_fold", False),
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


def run_multisymbol_walkforward(
    base_config: dict,
    symbols: list[str],
    output_root: Path,
    delay_seconds: float = 2.0,
) -> pd.DataFrame:
    rows = []
    for i, symbol in enumerate(symbols):
        if i > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)

        config = deepcopy(base_config)
        config["symbol"] = symbol
        safe_name = symbol.replace(".", "_")
        symbol_out = output_root / safe_name

        print(f"\n=== Walk-forward: {symbol} ===")
        try:
            result = run_walkforward(config, symbol_out)
            row = _wf_row(symbol, config, result)
            rows.append(row)
            print(
                f"  oos_days={row['oos_days']} | strategy={row['strategy_return']:.2%} | "
                f"excess_vs_benchmark={row.get('excess_vs_benchmark', float('nan')):.2%}"
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"symbol": symbol, "error": str(exc)})

    if not rows:
        raise RuntimeError("All walk-forward runs failed")

    df = pd.DataFrame(rows)
    sort_col = "excess_vs_benchmark" if "excess_vs_benchmark" in df.columns else "excess_vs_bh"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last")

    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / "multisymbol_walkforward.csv", index=False)
    with (output_root / "multisymbol_walkforward.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"symbols": symbols, "results": rows}, f, sort_keys=False)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch walk-forward across symbols")
    parser.add_argument("--config", type=Path, default=Path("configs/walkforward_xgb.yaml"))
    parser.add_argument("--symbols", type=Path, default=Path("configs/symbols_top.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multisymbol_walkforward"))
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    base_config = load_config(args.config)
    symbols = load_symbols(args.symbols)
    print(f"Running walk-forward on {len(symbols)} symbols...")
    df = run_multisymbol_walkforward(base_config, symbols, args.output_dir, args.delay)

    cols = ["symbol", "strategy_return", "buy_hold_return", "excess_vs_benchmark", "strategy_sharpe"]
    cols = [c for c in cols if c in df.columns]
    print("\n=== Walk-forward Comparison ===")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nReport saved to {args.output_dir / 'multisymbol_walkforward.csv'}")


if __name__ == "__main__":
    main()
