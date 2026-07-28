"""Tests for per-symbol probability threshold tuning."""

import numpy as np
import pandas as pd

from stock_trend.symbol_tuning import tune_proba_threshold


class StepProbaModel:
    def predict_proba(self, X):
        # higher index -> higher probability
        p = np.linspace(0.3, 0.9, len(X))
        return np.column_stack([1 - p, p])


def test_tune_proba_threshold_picks_best():
    X = pd.DataFrame({"a": range(20)})
    y = pd.Series([0] * 10 + [1] * 10)
    threshold, score = tune_proba_threshold(
        StepProbaModel(),
        X,
        y,
        thresholds=[0.4, 0.5, 0.6, 0.7],
    )
    assert threshold in [0.4, 0.5, 0.6, 0.7]
    assert score >= 0
