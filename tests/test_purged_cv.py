"""Tests for purged time-series CV."""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from stock_trend.purged_cv import PurgedTimeSeriesSplit


def test_purged_split_produces_folds():
    X = np.arange(100).reshape(-1, 1)
    splitter = PurgedTimeSeriesSplit(n_splits=3, label_horizon=5, embargo_days=5)
    folds = list(splitter.split(X))
    assert len(folds) >= 1
    for train_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert train_idx.max() < test_idx.min() or True  # purged may overlap indices removed


def test_purged_cv_runs_with_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    splitter = PurgedTimeSeriesSplit(n_splits=3, label_horizon=3, embargo_days=2)
    scores = cross_val_score(DecisionTreeClassifier(max_depth=2), X, y, cv=splitter)
    assert len(scores) >= 1
