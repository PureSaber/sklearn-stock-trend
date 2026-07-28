"""Performance metrics for backtest returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_performance_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    equity = (1 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    periods = len(clean)
    annualized_return = float((1 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1)

    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())

    std = clean.std()
    sharpe_ratio = float(clean.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }
