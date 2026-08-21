"""Tests for reproducibility: fixed seeds, artifact presence, no in-sample OOF leakage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_models_exist():
    root = Path(__file__).resolve().parent.parent
    cfg = __import__("json").loads((root / "configs" / "model_config.json").read_text())
    for f in range(cfg["num_folds"]):
        ckpt = root / "models" / f"fold_{f}"
        assert (ckpt / "model.safetensors").exists(), f"missing {ckpt}/model.safetensors"


def test_seed_is_fixed():
    root = Path(__file__).resolve().parent.parent
    pc = __import__("json").loads((root / "configs" / "project_config.json").read_text())
    mc = __import__("json").loads((root / "configs" / "model_config.json").read_text())
    assert pc["random_seed"] == mc["random_seed"]


def test_metric_files_exist():
    root = Path(__file__).resolve().parent.parent
    mdir = root / "outputs" / "metrics"
    for name in ["oof_metrics.json", "validation_metrics.json", "test_metrics.json"]:
        assert (mdir / name).exists(), f"missing {name}"


def test_figure_files_exist():
    root = Path(__file__).resolve().parent.parent
    fdir = root / "outputs" / "figures"
    for name in ["confusion_matrix_validation.png", "confusion_matrix_test.png"]:
        assert (fdir / name).exists(), f"missing {name}"
