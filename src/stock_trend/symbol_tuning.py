"""Per-symbol probability threshold tuning on a validation slice."""

from __future__ import annotations

from sklearn.metrics import f1_score


def tune_proba_threshold(
    model,
    X_val,
    y_val,
    thresholds: list[float] | None = None,
    scoring: str = "f1_macro",
) -> tuple[float, float]:
    """Pick threshold maximizing classification score on validation data."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for threshold tuning")

    candidates = thresholds or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    proba = model.predict_proba(X_val)[:, 1]

    best_threshold = candidates[0]
    best_score = -1.0
    for threshold in candidates:
        preds = (proba > threshold).astype(int)
        if scoring == "f1_macro":
            score = f1_score(y_val, preds, average="macro", zero_division=0)
        else:
            score = f1_score(y_val, preds, average="binary", zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, float(best_score)
