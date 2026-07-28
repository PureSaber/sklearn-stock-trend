"""Discrete retail long-flat backtest for a single A-share symbol."""

from __future__ import annotations

import pandas as pd

from stock_trend.metrics import compute_performance_metrics
from stock_trend.trading_costs import CostsConfig, buy_trade_cost, round_to_lots, sell_trade_cost


def simulate_retail_long_flat(
    signal_frame: pd.DataFrame,
    costs: CostsConfig,
) -> pd.DataFrame:
    """
    Simulate single-stock long-flat with lot size, T+1 holding, and retail costs.

    signal_frame columns: date, close, signal (1=want long, 0=want flat)
    Position executes on next bar; sells blocked until min_holding_days elapsed.
    """
    frame = signal_frame.sort_values("date").reset_index(drop=True).copy()
    cash = costs.initial_capital
    shares = 0
    buy_date_idx: int | None = None

    equity_rows = []
    for i, row in frame.iterrows():
        price = float(row["close"])
        signal = int(row.get("signal", 0))
        prev_signal = int(frame.loc[i - 1, "signal"]) if i > 0 else 0

        # Execute prior day's signal at today's open/close proxy
        if i > 0:
            target_long = prev_signal == 1

            if shares > 0 and not target_long:
                can_sell = False
                if buy_date_idx is not None:
                    held = i - buy_date_idx
                    can_sell = held >= costs.min_holding_days
                if can_sell:
                    notional = shares * price
                    cost = sell_trade_cost(notional, costs)
                    cash += notional - cost
                    shares = 0
                    buy_date_idx = None

            if shares == 0 and target_long:
                budget_shares = round_to_lots(cash / price, costs.lot_size)
                if budget_shares > 0:
                    notional = budget_shares * price
                    cost = buy_trade_cost(notional, costs)
                    if notional + cost <= cash:
                        cash -= notional + cost
                        shares = budget_shares
                        buy_date_idx = i

        portfolio_value = cash + shares * price
        equity_rows.append(
            {
                "date": row["date"],
                "close": price,
                "signal": signal,
                "shares": shares,
                "cash": cash,
                "portfolio_value": portfolio_value,
            }
        )

    result = pd.DataFrame(equity_rows)
    result["daily_return"] = result["portfolio_value"].pct_change().fillna(0.0)
    result["position"] = (result["shares"] > 0).astype(int)
    return result


def run_retail_backtest(signal_frame: pd.DataFrame, backtest_cfg: dict) -> dict:
    """Run retail simulation and return metrics + equity frame."""
    from stock_trend.trading_costs import costs_from_backtest_cfg

    costs = costs_from_backtest_cfg(backtest_cfg)
    equity = simulate_retail_long_flat(signal_frame, costs)
    metrics = compute_performance_metrics(equity["daily_return"])
    return {"equity": equity, "metrics": metrics, "costs": costs}
