"""Probability signal export for downstream factor fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def predict_proba_up(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(class=1) for each row."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba.")
    return model.predict_proba(X)[:, 1]


def build_signal_frame(
    dates: pd.Series,
    symbol: str,
    model,
    X: pd.DataFrame,
    backtest_cfg: dict,
    model_type: str = "",
    fold: int | None = None,
) -> pd.DataFrame:
    """Build date/symbol/proba/signal rows aligned with X index."""
    mode = backtest_cfg.get("signal_mode", "hard")
    threshold = backtest_cfg.get("proba_threshold", 0.5)

    if mode == "probability":
        proba = predict_proba_up(model, X)
        signal = (proba > threshold).astype(int)
    else:
        signal = model.predict(X)
        proba = np.where(signal == 1, 1.0, 0.0)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).values,
            "symbol": symbol,
            "proba_up": proba,
            "signal": signal,
            "model_type": model_type,
        }
    )
    if fold is not None:
        out["fold"] = fold
    return out


def export_proba_signals(
    df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Write signal frame to Parquet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def default_signals_path(output_dir: Path, name: str = "proba_signals.parquet") -> Path:
    return Path(output_dir) / name
