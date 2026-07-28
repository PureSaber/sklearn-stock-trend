"""Binary up/down labels from forward returns."""

import pandas as pd


def make_labels(
    close: pd.Series,
    forward_days: int = 5,
    threshold: float = 0.0,
) -> pd.Series:
    """Label 1 if forward return > threshold, else 0; NA where forward return unknown."""
    forward_return = close.shift(-forward_days) / close - 1
    labels = pd.Series(pd.NA, index=close.index, dtype="Int64")
    known = forward_return.notna()
    labels.loc[known] = (forward_return.loc[known] > threshold).astype(int)
    return labels.rename("label")
