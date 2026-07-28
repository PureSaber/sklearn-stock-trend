"""Model factory for classifiers used in training and walk-forward."""

from __future__ import annotations

from stock_trend.ensemble import EnsembleClassifier


def _rf_params(model_cfg: dict, random_state: int) -> dict:
    params = {
        "random_state": random_state,
        "n_estimators": model_cfg.get("n_estimators", 100),
        "min_samples_leaf": model_cfg.get("min_samples_leaf", 1),
    }
    if "max_depth" in model_cfg:
        params["max_depth"] = model_cfg["max_depth"]
    return params


def _xgb_params(model_cfg: dict, random_state: int) -> dict:
    return {
        "random_state": random_state,
        "n_estimators": model_cfg.get("n_estimators", 100),
        "max_depth": model_cfg.get("max_depth", 6),
        "learning_rate": model_cfg.get("learning_rate", 0.1),
        "subsample": model_cfg.get("subsample", 0.8),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "min_child_weight": model_cfg.get("min_child_weight", 1),
        "eval_metric": "logloss",
        "verbosity": 0,
    }


def build_estimator(model_type: str, random_state: int):
    """Unfitted estimator for GridSearchCV."""
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=random_state)
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    raise ValueError(f"Unsupported model type for tuning: {model_type}")


def get_model(model_type: str, model_cfg: dict):
    """Build classifier with hyperparameters from config."""
    random_state = model_cfg["random_state"]

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**_rf_params(model_cfg, random_state))

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**_xgb_params(model_cfg, random_state))

    if model_type == "ensemble":
        members = model_cfg.get("members", ["random_forest", "xgboost"])
        weights = model_cfg.get("weights", [0.5, 0.5])
        sub_models = []
        for member in members:
            sub_cfg = {
                "random_state": random_state,
                **model_cfg.get(member, {}),
            }
            sub_models.append(get_model(member, sub_cfg))
        return EnsembleClassifier(sub_models, weights=weights)

    raise ValueError(f"Unsupported model type: {model_type}")
