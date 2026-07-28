"""Tests for ensemble classifier."""

import numpy as np
import pandas as pd

from stock_trend.ensemble import EnsembleClassifier
from stock_trend.models import get_model


class ProbaDummy:
    def __init__(self, positive: float):
        self.positive = positive

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.positive)
        return np.column_stack([1 - p, p])


def test_ensemble_averages_probabilities():
    X = pd.DataFrame({"a": [0, 1, 2]})
    y = pd.Series([0, 1, 0])
    model = EnsembleClassifier(
        [ProbaDummy(0.2), ProbaDummy(0.8)],
        weights=[0.5, 0.5],
    )
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert np.allclose(proba, [0.5, 0.5, 0.5])


def test_get_model_ensemble():
    model = get_model(
        "ensemble",
        {
            "random_state": 42,
            "members": ["random_forest", "xgboost"],
            "weights": [0.5, 0.5],
            "random_forest": {"n_estimators": 10, "max_depth": 3},
            "xgboost": {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
        },
    )
    assert isinstance(model, EnsembleClassifier)
    assert len(model.models) == 2
