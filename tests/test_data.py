"""Tests for data module helpers."""

import pandas as pd

from stock_trend.data import _normalize_ohlcv, parse_symbol, symbol_to_index_code


def test_parse_symbol():
    assert parse_symbol("000001.SZ") == "000001"
    assert parse_symbol("600519.SH") == "600519"


def test_symbol_to_index_code():
    assert symbol_to_index_code("000300.SH") == "sh000300"
    assert symbol_to_index_code("399006.SZ") == "sz399006"


def test_normalize_ohlcv():
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 10.5],
            "收盘": [10.2, 10.8],
            "最高": [10.3, 11.0],
            "最低": [9.9, 10.4],
            "成交量": [1000, 1200],
        }
    )
    out = _normalize_ohlcv(raw)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert out["close"].iloc[0] == 10.2
