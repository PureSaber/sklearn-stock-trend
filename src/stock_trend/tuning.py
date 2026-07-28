"""Time-series hyperparameter tuning with GridSearchCV."""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from stock_trend.models import build_estimator
from stock_trend.purged_cv import PurgedTimeSeriesSplit


def run_tuning(
    X_train,
    y_train,
    tuning_cfg: dict,
    model_cfg: dict,
) -> tuple[dict, float]:
    """Search hyperparameters on the training split using TimeSeriesSplit."""
    param_grid = tuning_cfg.get("param_grid")
    if not param_grid:
        raise ValueError("tuning.param_grid must be set when tuning is enabled")

    model_type = model_cfg["type"]
    random_state = model_cfg["random_state"]
    n_splits = tuning_cfg.get("n_splits", 5)
    scoring = tuning_cfg.get("scoring", "f1_macro")
    cv_mode = tuning_cfg.get("cv_mode", "timeseries")
    if cv_mode == "purged":
        tscv = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            label_horizon=int(tuning_cfg.get("label_horizon", 5)),
            embargo_days=int(tuning_cfg.get("embargo_days", 5)),
        )
    else:
        tscv = TimeSeriesSplit(n_splits=n_splits)

    search = GridSearchCV(
        build_estimator(model_type, random_state),
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return dict(search.best_params_), float(search.best_score_)
