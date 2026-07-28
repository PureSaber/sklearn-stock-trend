"""Tests for technical indicator features."""

import pandas as pd

from stock_trend.features import add_macd, add_rsi, build_features, get_feature_columns


def _sample_ohlcv(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series([100 + i * 0.5 + (i % 7) for i in range(n)], dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_add_rsi_bounded():
    df = add_rsi(_sample_ohlcv(), window=14)
    rsi = df["rsi_14"].dropna()
    assert len(rsi) > 0
    assert rsi.min() >= 0
    assert rsi.max() <= 100


def test_add_macd_columns():
    df = add_macd(_sample_ohlcv())
    assert {"macd", "macd_signal", "macd_hist"}.issubset(df.columns)


def test_build_features_includes_rsi_macd_and_ma_ratios():
    config = {
        "ma_windows": [5, 20],
        "rsi_window": 14,
        "use_macd": True,
    }
    df = build_features(_sample_ohlcv(80), feature_cfg=config)
    cols = get_feature_columns(df)

    assert "rsi_14" in cols
    assert "macd" in cols
    assert "close_ma_5_ratio" in cols
    assert "close_ma_20_ratio" in cols
    assert "close" not in cols
