"""Tests for retail trading costs."""

from stock_trend.trading_costs import (
    CostsConfig,
    buy_trade_cost,
    round_to_lots,
    sell_trade_cost,
)


def test_round_to_lots():
    assert round_to_lots(250, 100) == 200
    assert round_to_lots(99, 100) == 0


def test_buy_trade_cost_min_commission():
    costs = CostsConfig(min_commission=5.0, commission=0.0003)
    assert buy_trade_cost(1000, costs) == 5.0 + 0.1


def test_sell_trade_cost_includes_stamp_tax():
    costs = CostsConfig(min_commission=5.0, stamp_tax=0.0005)
    notional = 10_000
    cost = sell_trade_cost(notional, costs)
    assert cost > 5.0
    assert cost >= notional * 0.0005
