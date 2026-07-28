"""Purged and embargo time-series cross-validation."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time-series CV with purge gap before validation and embargo after.

    Purge removes training samples whose labels overlap the validation window
    (label_horizon trading days). Embargo removes training samples immediately
    after the validation set to reduce serial correlation leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        label_horizon: int = 5,
        embargo_days: int = 5,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo_days = embargo_days

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if n_samples <= self.n_splits:
            raise ValueError(f"Not enough samples ({n_samples}) for {self.n_splits} splits")

        indices = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            val_start = (fold + 1) * fold_size
            val_end = (fold + 2) * fold_size if fold < self.n_splits - 1 else n_samples

            purge_end = max(0, val_start - self.label_horizon)
            embargo_start = min(n_samples, val_end + self.embargo_days)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_start:val_end] = False
            train_mask[purge_end:val_start] = False
            train_mask[val_end:embargo_start] = False

            train_idx = indices[train_mask]
            test_idx = indices[val_start:val_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx
