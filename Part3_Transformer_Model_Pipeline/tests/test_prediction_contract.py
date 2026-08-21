"""Tests for the prediction output contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.common import load_project_config


def _load():
    root = Path(__file__).resolve().parent.parent
    cfg = load_project_config(root)
    pred_dir = root / "outputs" / "predictions"
    oof = pd.read_csv(pred_dir / "member2_oof_predictions_deterministic.csv")
    val = pd.read_csv(pred_dir / "member2_validation_predictions_fold_ensemble.csv")
    test = pd.read_csv(pred_dir / "member2_test_predictions_fold_ensemble.csv")
    handoff = pd.read_csv(root / "inputs" / "member2_handoff.csv")
    return cfg, pred_dir, oof, val, test, handoff


def test_oof_columns():
    cfg, pred_dir, oof, val, test, handoff = _load()
    assert list(oof.columns) == ["unique_post_id", "fold", "true_stress", "prediction"]


def test_oof_coverage():
    cfg, pred_dir, oof, val, test, handoff = _load()
    train_ids = set(handoff.loc[handoff["model_role"] == "train", "unique_post_id"])
    assert set(oof["unique_post_id"]) == train_ids
    assert oof["unique_post_id"].duplicated().sum() == 0


def test_no_oof_in_sample_leakage():
    cfg, pred_dir, oof, val, test, handoff = _load()
    # each train row predicted by its own held-out fold model only
    assert oof["fold"].nunique() == cfg["num_folds"]
    assert (oof.groupby("unique_post_id").size() == 1).all()


def test_val_test_coverage():
    cfg, pred_dir, oof, val, test, handoff = _load()
    val_ids = set(handoff.loc[handoff["model_role"] == "validation", "unique_post_id"])
    test_ids = set(handoff.loc[handoff["model_role"] == "test", "unique_post_id"])
    assert set(val["unique_post_id"]) == val_ids
    assert set(test["unique_post_id"]) == test_ids


def test_prediction_range():
    cfg, pred_dir, oof, val, test, handoff = _load()
    # Predictions are intentionally NOT clipped (matches production). Check they are
    # bounded within a sensible band around [1,10] (allow ~1 unit of unclipped slack).
    lo, hi = cfg["prediction_range"][0] - 1.0, cfg["prediction_range"][1] + 1.0
    for df in (oof, val, test):
        assert df["prediction"].min() >= lo and df["prediction"].max() <= hi


def test_handoff_column_contract():
    cfg, pred_dir, oof, val, test, handoff = _load()
    assert "prediction_std_fold_models" in val.columns
    assert "prediction_std_fold_models" in test.columns
    for df in (val, test):
        fold_cols = [c for c in df.columns if c.startswith("fold_model_")]
        assert len(fold_cols) == cfg["num_folds"]
        manual = df[fold_cols].values.mean(axis=1)
        assert np.allclose(manual, df["prediction"].values, atol=1e-5)
