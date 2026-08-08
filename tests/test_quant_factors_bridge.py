from pathlib import Path

import yaml


def test_features_from_quant_factors_config() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "features_from_quant_factors.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["features"]["source"] == "quant_factors"
    assert "momentum_20d" in cfg["features"]["columns"]
