"""Technical indicator feature engineering."""

from __future__ import annotations

import pandas as pd

NON_FEATURE_COLUMNS = {"date", "open", "high", "low", "close", "volume", "label"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model input columns (exclude OHLCV, date, and label)."""
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def add_returns(df: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out[col].pct_change()
    out["volatility_20d"] = out["return_1d"].rolling(20).std()
    return out


def add_moving_averages(
    df: pd.DataFrame,
    col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    for w in windows or [5, 20, 60]:
        out[f"ma_{w}"] = out[col].rolling(w).mean()
    return out


def add_ma_ratios(
    df: pd.DataFrame,
    col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Price relative to moving averages (close / ma - 1)."""
    out = df.copy()
    for w in windows or [5, 20, 60]:
        ma_col = f"ma_{w}"
        if ma_col not in out.columns:
            out[ma_col] = out[col].rolling(w).mean()
        out[f"close_ma_{w}_ratio"] = out[col] / out[ma_col] - 1
    return out


def add_rsi(df: pd.DataFrame, col: str = "close", window: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out[col].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    out[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return out


def add_macd(
    df: pd.DataFrame,
    col: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out[col].ewm(span=fast, adjust=False).mean()
    ema_slow = out[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = macd_line - signal_line
    return out


def build_features(df: pd.DataFrame, feature_cfg: dict | None = None) -> pd.DataFrame:
    """Build feature columns from OHLCV DataFrame (NaN rows dropped by caller)."""
    cfg = feature_cfg or {}
    ma_windows = cfg.get("ma_windows", [5, 20, 60])
    rsi_window = cfg.get("rsi_window", 14)
    macd_cfg = cfg.get("macd", {})

    out = add_returns(df)
    out = add_moving_averages(out, windows=ma_windows)
    out = add_ma_ratios(out, windows=ma_windows)
    out = add_rsi(out, window=rsi_window)
    if cfg.get("use_macd", True):
        out = add_macd(
            out,
            fast=macd_cfg.get("fast", 12),
            slow=macd_cfg.get("slow", 26),
            signal=macd_cfg.get("signal", 9),
        )
    return out
