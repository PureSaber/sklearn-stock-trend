"""HTML report generation for training and backtest outputs."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import yaml


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _metric_rows(prefix: str, metrics: dict) -> str:
    if not metrics:
        return ""
    rows = []
    for key, value in metrics.items():
        if isinstance(value, float):
            display = f"{value:.4f}" if abs(value) < 10 else f"{value:.2%}" if "return" in key or "drawdown" in key else f"{value:.4f}"
            if "return" in key or "drawdown" in key:
                display = f"{value:.2%}"
            else:
                display = f"{value:.4f}"
        else:
            display = html.escape(str(value))
        rows.append(f"<tr><td>{html.escape(prefix + key)}</td><td>{display}</td></tr>")
    return "\n".join(rows)


def write_html_report(output_dir: Path, title: str = "Stock Trend Report") -> Path:
    """Generate outputs/latest/report.html from saved artifacts."""
    output_dir = Path(output_dir)
    latest = output_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    report_path = latest / "report.html"

    train_meta = _read_yaml(output_dir / "train_meta.yaml")
    eval_report = _read_yaml(output_dir / "eval_report.yaml")
    backtest_summary = _read_yaml(output_dir / "backtest_summary.yaml")
    wf_summary = _read_yaml(output_dir / "walkforward" / "walkforward_summary.yaml")

    importance_html = ""
    fi_path = output_dir / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).head(10)
        rows = "".join(
            f"<tr><td>{html.escape(r.feature)}</td><td>{r.importance:.4f}</td></tr>"
            for r in fi.itertuples()
        )
        importance_html = f"<h2>Feature Importance (Top 10)</h2><table><tr><th>Feature</th><th>Importance</th></tr>{rows}</table>"

    plot_html = ""
    for plot_name in ("backtest_equity.png", "walkforward/walkforward_equity.png"):
        plot_path = output_dir / plot_name
        if plot_path.exists():
            dest = latest / plot_path.name.replace("/", "_")
            if not dest.exists():
                dest.write_bytes(plot_path.read_bytes())
            plot_html += f'<h2>{html.escape(plot_path.stem)}</h2><img src="{dest.name}" alt="equity curve" style="max-width:100%"/>'

    strategy = backtest_summary.get("strategy", wf_summary.get("strategy", {}))
    buy_hold = backtest_summary.get("buy_and_hold", wf_summary.get("buy_and_hold", {}))

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 960px; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p><strong>Symbol:</strong> {html.escape(str(train_meta.get('symbol', backtest_summary.get('symbol', 'N/A'))))}</p>

  <h2>Training</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Model</td><td>{html.escape(str(train_meta.get('model_type', 'N/A')))}</td></tr>
    <tr><td>Train accuracy</td><td>{train_meta.get('train_accuracy', 'N/A')}</td></tr>
    <tr><td>Test accuracy</td><td>{train_meta.get('test_accuracy', 'N/A')}</td></tr>
  </table>

  <h2>Evaluation</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {_metric_rows('', eval_report.get('metrics', eval_report))}
  </table>

  <h2>Backtest / Walk-forward</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {_metric_rows('strategy.', strategy)}
    {_metric_rows('buy_hold.', buy_hold)}
  </table>

  {importance_html}
  {plot_html}
</body>
</html>
"""
    report_path.write_text(body, encoding="utf-8")
    return report_path
