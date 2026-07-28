"""Shared configuration loading."""

from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
