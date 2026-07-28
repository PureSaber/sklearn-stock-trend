"""A-share retail trading cost helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CostsConfig:
    """Retail cost parameters for single-stock simulation."""

    commission: float = 0.0003
    slippage: float = 0.0001
    min_commission: float = 5.0
    stamp_tax: float = 0.0005
    lot_size: int = 100
    initial_capital: float = 10_000.0
    min_holding_days: int = 1


def round_to_lots(shares: float, lot_size: int) -> int:
    """Round share count down to the nearest tradable lot."""
    if lot_size <= 0:
        return math.floor(shares)
    return int(shares // lot_size) * lot_size


def buy_trade_cost(notional: float, costs: CostsConfig) -> float:
    """One-way buy cost: commission (with minimum), plus slippage."""
    if notional <= 0:
        return 0.0
    commission = max(notional * costs.commission, costs.min_commission)
    slippage = notional * costs.slippage
    return commission + slippage


def sell_trade_cost(notional: float, costs: CostsConfig) -> float:
    """One-way sell cost: commission (with minimum), stamp tax, plus slippage."""
    if notional <= 0:
        return 0.0
    commission = max(notional * costs.commission, costs.min_commission)
    stamp = notional * costs.stamp_tax
    slippage = notional * costs.slippage
    return commission + stamp + slippage


def costs_from_backtest_cfg(backtest_cfg: dict) -> CostsConfig:
    """Build CostsConfig from YAML backtest block."""
    return CostsConfig(
        commission=float(backtest_cfg.get("commission_rate", 0.0003)),
        slippage=float(backtest_cfg.get("slippage", 0.0001)),
        min_commission=float(backtest_cfg.get("min_commission", 5.0)),
        stamp_tax=float(backtest_cfg.get("stamp_tax", 0.0005)),
        lot_size=int(backtest_cfg.get("lot_size", 100)),
        initial_capital=float(backtest_cfg.get("initial_capital", 10_000.0)),
        min_holding_days=int(backtest_cfg.get("min_holding_days", 1)),
    )
