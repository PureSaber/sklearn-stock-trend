"""Assemble feature matrix, labels, and time-ordered train/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_trend.data import load_or_fetch
from stock_trend.features import build_features, get_feature_columns
from stock_trend.labels import make_labels


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    ohlcv: pd.DataFrame
    test_dates: pd.Series


def build_labeled_frame(ohlcv: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Join engineered features with binary labels on aligned dates."""
    label_cfg = config["label"]
    feature_cfg = config.get("features", {})

    featured = build_features(ohlcv, feature_cfg=feature_cfg)
    labels = make_labels(
        ohlcv.set_index("date")["close"],
        forward_days=label_cfg["forward_days"],
        threshold=label_cfg.get("threshold", 0.0),
    ).reset_index()

    merged = featured.merge(labels, on="date", how="inner")
    feature_cols = get_feature_columns(merged)
    return merged.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)


def time_series_split(
    frame: pd.DataFrame,
    feature_columns: list[str],
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split by time: earliest (1 - test_size) for train, remainder for test."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(frame) * (1 - test_size))
    if split_idx < 1 or split_idx >= len(frame):
        raise ValueError(
            f"Not enough samples ({len(frame)}) for test_size={test_size}; "
            "try a shorter test window or more history."
        )

    train = frame.iloc[:split_idx]
    test = frame.iloc[split_idx:]
    return (
        train[feature_columns],
        test[feature_columns],
        train["label"].astype(int),
        test["label"].astype(int),
        test["date"].reset_index(drop=True),
    )


def prepare_dataset(config: dict, cache_dir: Path | None = None) -> DatasetSplit:
    """Load OHLCV, build features/labels, and return a time-ordered split."""
    ohlcv = load_or_fetch(
        symbol=config["symbol"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        cache_dir=cache_dir,
    )
    frame = build_labeled_frame(ohlcv, config)
    feature_columns = get_feature_columns(frame)
    test_size = config["model"]["test_size"]

    X_train, X_test, y_train, y_test, test_dates = time_series_split(
        frame, feature_columns, test_size
    )
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        ohlcv=ohlcv,
        test_dates=test_dates,
    )
