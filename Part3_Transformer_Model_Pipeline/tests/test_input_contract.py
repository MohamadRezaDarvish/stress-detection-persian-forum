"""Tests for the input data contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.common import load_project_config
from src.validate_inputs import validate_inputs


def _load():
    root = Path(__file__).resolve().parent.parent
    cfg = load_project_config(root)
    handoff = pd.read_csv(root / "inputs" / "member2_handoff.csv")
    man = pd.read_csv(root / "inputs" / "modeling_manifest_v2.csv")
    df = handoff.merge(
        man[["unique_post_id", "group_id", "author", "thread_id", "content_hash"]],
        on="unique_post_id",
        how="left",
    )
    df["content"] = df["content"].fillna("").astype(str)
    return df, cfg


def test_input_schema():
    df, cfg = _load()
    required = ["unique_post_id", "final_stress", "clinical_class", "model_role", "oof_fold", "content"]
    assert all(c in df.columns for c in required)


def test_id_uniqueness():
    df, _ = _load()
    assert df["unique_post_id"].is_unique


def test_target_range():
    df, cfg = _load()
    lab = df[df["model_role"].isin(["train", "validation", "test"])]
    lo, hi = cfg["prediction_range"]
    assert lab["final_stress"].min() >= lo and lab["final_stress"].max() <= hi


def test_role_separation():
    df, _ = _load()
    ids = {r: set(df.loc[df["model_role"] == r, "unique_post_id"]) for r in df["model_role"].unique()}
    for a in ids:
        for b in ids:
            if a < b:
                assert len(ids[a] & ids[b]) == 0, f"overlap {a}/{b}"


def test_fold_completeness():
    df, cfg = _load()
    train = df[df["model_role"] == "train"]
    assert set(train["oof_fold"].dropna().unique()) == set(range(cfg["num_folds"]))
    assert train["oof_fold"].isna().sum() == 0


def test_validate_inputs_passes():
    df, cfg = _load()
    summary = validate_inputs(df, cfg)
    assert summary["train"] == 4226
    assert summary["validation"] == 452
    assert summary["test"] == 453
    assert summary["embargo"] == 484
