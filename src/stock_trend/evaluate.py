"""Model evaluation utilities and CLI entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from stock_trend.config import load_config
from stock_trend.dataset import prepare_dataset


def load_model_artifacts(output_dir: Path) -> tuple[object, list[str]]:
    model_path = output_dir / "model.pkl"
    columns_path = output_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train first.")
    if not columns_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {columns_path}.")

    model = joblib.load(model_path)
    with columns_path.open(encoding="utf-8") as f:
        feature_columns = json.load(f)
    return model, feature_columns


def evaluate_model(model, X_test, y_test) -> dict:
    """Run predictions and return structured evaluation metrics."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": report,
        "classification_report_text": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": len(y_test),
    }


def save_eval_report(output_dir: Path, metrics: dict, config: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_report.yaml"

    payload = {
        "symbol": config["symbol"],
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "test_samples": metrics["test_samples"],
        "accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }
    with report_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    text_path = output_dir / "eval_report.txt"
    text_path.write_text(metrics["classification_report_text"], encoding="utf-8")
    return report_path


def run_evaluation(config: dict, output_dir: Path | None = None) -> dict:
    """Load saved model and evaluate on the configured time-based test split."""
    out = output_dir or Path(config.get("output_dir", "outputs"))
    model, saved_columns = load_model_artifacts(out)
    dataset = prepare_dataset(config)

    if saved_columns != dataset.feature_columns:
        raise ValueError(
            "Feature columns mismatch between saved model and current config.\n"
            f"  Saved:   {saved_columns}\n"
            f"  Current: {dataset.feature_columns}\n"
            "Retrain the model after changing features."
        )

    X_test = dataset.X_test[saved_columns]
    metrics = evaluate_model(model, X_test, dataset.y_test)
    save_eval_report(out, metrics, config)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stock trend classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))

    print(f"Evaluate model for {config['symbol']}")
    metrics = run_evaluation(config, output_dir)

    print(f"Test samples: {metrics['test_samples']}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report_text"])
    print(f"\nReport saved to {output_dir / 'eval_report.yaml'}")


if __name__ == "__main__":
    main()
