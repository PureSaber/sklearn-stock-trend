"""Tests for hyperparameter tuning."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stock_trend.models import build_estimator, get_model
from stock_trend.tuning import run_tuning


def _sample_xy():
    rng = pd.Series(range(200))
    X = pd.DataFrame(
        {
            "f1": rng / 200,
            "f2": (rng % 17) / 17,
            "f3": (rng % 23) / 23,
        }
    )
    y = (rng % 3 == 0).astype(int)
    return X, y


def test_run_tuning_returns_best_params_rf():
    X, y = _sample_xy()
    tuning_cfg = {
        "n_splits": 3,
        "scoring": "f1_macro",
        "param_grid": {
            "max_depth": [3, 5],
            "min_samples_leaf": [5, 10],
            "n_estimators": [10],
        },
    }
    model_cfg = {"type": "random_forest", "random_state": 42}

    best_params, best_score = run_tuning(X, y, tuning_cfg, model_cfg)

    assert isinstance(best_params, dict)
    assert "max_depth" in best_params
    assert isinstance(best_score, float)


def test_run_tuning_xgboost():
    X, y = _sample_xy()
    tuning_cfg = {
        "n_splits": 3,
        "scoring": "f1_macro",
        "param_grid": {
            "max_depth": [3, 5],
            "n_estimators": [10],
            "learning_rate": [0.1],
        },
    }
    model_cfg = {"type": "xgboost", "random_state": 42}

    best_params, best_score = run_tuning(X, y, tuning_cfg, model_cfg)

    assert "max_depth" in best_params
    assert isinstance(best_score, float)


def test_build_estimators():
    assert isinstance(build_estimator("random_forest", 42), RandomForestClassifier)
    xgb = build_estimator("xgboost", 42)
    assert xgb.__class__.__name__ == "XGBClassifier"


def test_get_model_xgboost():
    model = get_model(
        "xgboost",
        {"random_state": 42, "n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
    )
    assert model.__class__.__name__ == "XGBClassifier"
