"""Ensemble classifier combining multiple base models."""

from __future__ import annotations

import numpy as np


class EnsembleClassifier:
    """Average predicted probabilities from multiple fitted classifiers."""

    def __init__(self, models: list, weights: list[float] | None = None):
        if not models:
            raise ValueError("Ensemble requires at least one model")
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("weights length must match number of models")
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        positive = np.zeros(len(X), dtype=float)
        for model, weight in zip(self.models, self.weights):
            positive += weight * model.predict_proba(X)[:, 1]
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average feature importances when all members expose them."""
        importances = [m.feature_importances_ for m in self.models if hasattr(m, "feature_importances_")]
        if not importances:
            raise AttributeError("No base model exposes feature_importances_")
        return np.mean(importances, axis=0)
