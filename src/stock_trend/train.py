"""Model training entry point."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report

from stock_trend.config import load_config
from stock_trend.dataset import prepare_dataset
from stock_trend.models import get_model
from stock_trend.signals import build_signal_frame, default_signals_path, export_proba_signals
from stock_trend.symbol_tuning import tune_proba_threshold
from stock_trend.tuning import run_tuning


def save_feature_importance(
    output_dir: Path,
    model,
    feature_columns: list[str],
) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)


def save_training_artifacts(
    output_dir: Path,
    model,
    feature_columns: list[str],
    config: dict,
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.pkl")

    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    save_feature_importance(output_dir, model, feature_columns)

    meta = {
        "symbol": config["symbol"],
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "model_type": config["model"]["type"],
        "train_samples": metrics["train_samples"],
        "test_samples": metrics["test_samples"],
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if metrics.get("optimal_proba_threshold") is not None:
        meta["optimal_proba_threshold"] = metrics["optimal_proba_threshold"]
        meta["proba_threshold_score"] = metrics.get("proba_threshold_score")
    if metrics.get("tuning_best_params"):
        meta["tuning_best_params"] = metrics["tuning_best_params"]
        meta["tuning_best_score"] = metrics["tuning_best_score"]
    with (output_dir / "train_meta.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)


def run_training(config: dict, output_dir: Path | None = None) -> dict:
    """Train model, save artifacts, and return metrics."""
    model_cfg = dict(config["model"])
    tuning_cfg = config.get("tuning", {})
    out = output_dir or Path(config.get("output_dir", "outputs"))

    dataset = prepare_dataset(config)

    tuning_best_params = None
    tuning_best_score = None
    if tuning_cfg.get("enabled", False):
        tuning_best_params, tuning_best_score = run_tuning(
            dataset.X_train,
            dataset.y_train,
            tuning_cfg,
            model_cfg,
        )
        model_cfg.update(tuning_best_params)

    model = get_model(model_cfg["type"], model_cfg)

    symbol_tune_cfg = config.get("symbol_tuning", {})
    optimal_proba_threshold = None
    proba_threshold_score = None
    X_train, y_train = dataset.X_train, dataset.y_train

    if symbol_tune_cfg.get("tune_proba_threshold", False):
        val_ratio = symbol_tune_cfg.get("val_ratio", 0.2)
        split_idx = int(len(X_train) * (1 - val_ratio))
        if split_idx < 1 or split_idx >= len(X_train):
            raise ValueError("symbol_tuning.val_ratio leaves no train or validation samples")

        X_fit, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_fit, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
        model.fit(X_fit, y_fit)

        optimal_proba_threshold, proba_threshold_score = tune_proba_threshold(
            model,
            X_val,
            y_val,
            thresholds=symbol_tune_cfg.get("proba_thresholds"),
            scoring=symbol_tune_cfg.get("scoring", "f1_macro"),
        )
        config.setdefault("backtest", {})["proba_threshold"] = optimal_proba_threshold
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(dataset.X_test)
    train_accuracy = accuracy_score(dataset.y_train, y_train_pred)
    test_accuracy = accuracy_score(dataset.y_test, y_test_pred)

    metrics = {
        "train_samples": len(dataset.X_train),
        "test_samples": len(dataset.X_test),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "tuning_best_params": tuning_best_params,
        "tuning_best_score": tuning_best_score,
        "optimal_proba_threshold": optimal_proba_threshold,
        "proba_threshold_score": proba_threshold_score,
        "feature_count": len(dataset.feature_columns),
        "classification_report_text": classification_report(
            dataset.y_test, y_test_pred, digits=4
        ),
    }
    save_training_artifacts(out, model, dataset.feature_columns, config, metrics)

    signals_cfg = config.get("signals", {})
    if signals_cfg.get("export", False):
        backtest_cfg = config.get("backtest", {})
        proba_frame = build_signal_frame(
            dataset.test_dates,
            config["symbol"],
            model,
            dataset.X_test,
            backtest_cfg,
            model_type=model_cfg["type"],
        )
        export_proba_signals(proba_frame, default_signals_path(out))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stock trend classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    output_dir = Path(config.get("output_dir", "outputs"))
    tuning_cfg = config.get("tuning", {})

    print(f"Symbol: {config['symbol']}, model: {model_cfg['type']}")
    print("Loading data and building dataset...")
    if tuning_cfg.get("enabled", False):
        print("Running TimeSeriesSplit hyperparameter search on training data...")

    metrics = run_training(config, output_dir)

    if metrics.get("tuning_best_params"):
        print(f"CV best params: {metrics['tuning_best_params']}")
        print(f"CV best score ({tuning_cfg.get('scoring', 'f1_macro')}): {metrics['tuning_best_score']:.4f}")

    if metrics.get("optimal_proba_threshold") is not None:
        print(
            f"Optimal proba_threshold: {metrics['optimal_proba_threshold']:.2f} "
            f"(val score: {metrics.get('proba_threshold_score', 0):.4f})"
        )

    print(
        f"Train: {metrics['train_samples']} samples | "
        f"Test: {metrics['test_samples']} samples | "
        f"Features: {metrics['feature_count']}"
    )
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")
    print("\nTest classification report:")
    print(metrics["classification_report_text"])

    print(f"\nModel saved to {output_dir / 'model.pkl'}")
    print(f"Feature importance saved to {output_dir / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
