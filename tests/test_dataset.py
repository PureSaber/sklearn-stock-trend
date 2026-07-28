"""Tests for dataset assembly and time-series split."""

import pandas as pd

from stock_trend.dataset import build_labeled_frame, time_series_split
from stock_trend.features import get_feature_columns


def _sample_ohlcv(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), dtype=float)
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


def test_build_labeled_frame_shape():
    config = {
        "label": {"forward_days": 5, "threshold": 0.0},
        "features": {"ma_windows": [5, 20, 60], "rsi_window": 14, "use_macd": True},
    }
    frame = build_labeled_frame(_sample_ohlcv(), config)
    feature_cols = get_feature_columns(frame)

    assert len(frame) > 0
    assert "label" in frame.columns
    assert set(feature_cols).issubset(set(frame.columns))
    assert frame["label"].isin([0, 1]).all()


def test_time_series_split_is_chronological():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="B"),
            "f1": range(10),
            "f2": range(10, 20),
            "label": [0, 1] * 5,
        }
    )
    feature_columns = ["f1", "f2"]
    X_train, X_test, y_train, y_test, test_dates = time_series_split(
        frame, feature_columns, test_size=0.2
    )

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(test_dates) == 2
    assert X_train["f1"].iloc[-1] == 7
    assert X_test["f1"].iloc[0] == 8
    assert y_train.iloc[-1] == frame["label"].iloc[7]
    assert y_test.iloc[0] == frame["label"].iloc[8]
