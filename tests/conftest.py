"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
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
