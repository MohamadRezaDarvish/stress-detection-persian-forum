from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .feature_engineering import RAW_REQUIRED_COLUMNS, FEATURE_ORDER, FORBIDDEN_FEATURES

REQUIRED_MANIFEST_COLUMNS = {
    "unique_post_id", "group_id", "author", "thread_id", "content_hash",
    "final_stress", "clinical_class", "model_role", "oof_fold",
    "training_sample_weight", "use_for_training", "use_for_model_selection",
    "use_for_final_test", "exclude_from_modeling"
}
REQUIRED_HANDOFF_COLUMNS = {
    "unique_post_id", "final_stress", "clinical_class", "model_role", "oof_fold",
    "group_id", "training_sample_weight", *RAW_REQUIRED_COLUMNS
}


def load_inputs(root: Path, project_config: dict) -> dict:
    paths = {k: root / v for k, v in project_config["input_files"].items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(missing))
    return {
        "paths": paths,
        "handoff": pd.read_csv(paths["handoff"]),
        "manifest": pd.read_csv(paths["manifest"]),
        "feature_contract": json.loads(paths["feature_contract"].read_text(encoding="utf-8")),
        "oof_template": pd.read_csv(paths["oof_template"]),
        "holdout_template": pd.read_csv(paths["holdout_template"]),
    }


def _assert_columns(df: pd.DataFrame, required: set[str], name: str):
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_true_class_ranges(df: pd.DataFrame):
    allowed = {"Low", "Moderate", "High", "Very High"}
    if not set(df["clinical_class"].dropna().unique()).issubset(allowed):
        raise ValueError("Unknown clinical_class value found")
    y = pd.to_numeric(df["final_stress"], errors="coerce")
    checks = {
        "Low": y.between(1, 3, inclusive="both"),
        "Moderate": y.between(3, 5, inclusive="left"),
        "High": y.between(5, 7, inclusive="both"),
        "Very High": y.between(7, 10, inclusive="both"),
    }
    bad = pd.Series(False, index=df.index)
    for label, valid in checks.items():
        bad |= df["clinical_class"].eq(label) & ~valid
    if bad.any():
        raise ValueError(f"clinical_class/target inconsistency in {int(bad.sum())} rows")


def validate_inputs(bundle: dict, project_config: dict) -> dict:
    handoff = bundle["handoff"].copy()
    manifest = bundle["manifest"].copy()
    contract = bundle["feature_contract"]
    oof_template = bundle["oof_template"]
    holdout_template = bundle["holdout_template"]

    _assert_columns(handoff, REQUIRED_HANDOFF_COLUMNS, "member1_handoff")
    _assert_columns(manifest, REQUIRED_MANIFEST_COLUMNS, "modeling_manifest_v2")
    if handoff["unique_post_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate unique_post_id in handoff")
    if manifest["unique_post_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate unique_post_id in manifest")
    if set(handoff["unique_post_id"].astype(str)) != set(manifest["unique_post_id"].astype(str)):
        raise ValueError("Handoff and manifest ID sets differ")

    compare_cols = ["unique_post_id", "model_role", "oof_fold", "final_stress", "clinical_class", "group_id"]
    merged = handoff[compare_cols].merge(manifest[compare_cols], on="unique_post_id", suffixes=("_h", "_m"), validate="one_to_one")
    for col in ["model_role", "clinical_class", "group_id"]:
        if not merged[f"{col}_h"].fillna("__NA__").astype(str).equals(merged[f"{col}_m"].fillna("__NA__").astype(str)):
            raise ValueError(f"{col} differs between handoff and manifest")
    if not np.isclose(merged["final_stress_h"].astype(float), merged["final_stress_m"].astype(float), equal_nan=True).all():
        raise ValueError("final_stress differs between handoff and manifest")
    if not (merged["oof_fold_h"].fillna(-1).astype(int) == merged["oof_fold_m"].fillna(-1).astype(int)).all():
        raise ValueError("oof_fold differs between handoff and manifest")

    valid_roles = {"train", "validation", "test", "embargo"}
    if set(handoff["model_role"].unique()) != valid_roles:
        raise ValueError(f"Unexpected role set: {set(handoff['model_role'].unique())}")
    role_counts = handoff["model_role"].value_counts().to_dict()
    if role_counts != manifest["model_role"].value_counts().to_dict():
        raise ValueError("Role counts do not match manifest")

    labeled = handoff[handoff["model_role"].isin(["train", "validation", "test"])]
    if labeled["final_stress"].isna().any():
        raise ValueError("Missing final_stress in a labeled modeling role")
    if not labeled["final_stress"].between(*project_config["prediction_range"]).all():
        raise ValueError("Target outside configured range")
    _validate_true_class_ranges(labeled)

    train = handoff[handoff["model_role"].eq("train")]
    folds = train["oof_fold"].astype(int)
    if set(folds.unique()) != set(range(project_config["number_of_folds"])):
        raise ValueError("Train folds are not exactly 0..4")
    if folds.isna().any():
        raise ValueError("Missing train fold")
    if handoff.loc[~handoff["model_role"].eq("train"), "oof_fold"].notna().any():
        raise ValueError("Non-train row has an OOF fold")
    if train["training_sample_weight"].isna().any() or (train["training_sample_weight"] <= 0).any():
        raise ValueError("Invalid training_sample_weight")

    for col in RAW_REQUIRED_COLUMNS:
        if labeled[col].isna().any():
            raise ValueError(f"Required upstream feature {col} contains missing values")

    modeled_manifest = manifest[manifest["model_role"].isin(["train", "validation", "test"])]
    group_audit = {}
    for col in project_config["group_columns"]:
        nonnull = modeled_manifest[modeled_manifest[col].notna()]
        cross_roles = int((nonnull.groupby(col)["model_role"].nunique() > 1).sum())
        if cross_roles:
            raise ValueError(f"{col} crosses modeled train/validation/test roles")
        train_nonnull = manifest[(manifest["model_role"].eq("train")) & manifest[col].notna()]
        cross_folds = int((train_nonnull.groupby(col)["oof_fold"].nunique() > 1).sum())
        if cross_folds:
            raise ValueError(f"{col} crosses train folds")
        group_audit[col] = {"cross_modeled_roles": cross_roles, "cross_train_folds": cross_folds}

    if oof_template.columns.tolist() != ["unique_post_id", "fold", "true_stress", "prediction", "prediction_std_optional"]:
        raise ValueError("OOF template schema mismatch")
    if holdout_template.columns.tolist() != ["unique_post_id", "model_role", "true_stress", "prediction", "prediction_std_optional"]:
        raise ValueError("Holdout template schema mismatch")

    contract_forbidden = set(contract.get("direct_text_modeling_forbidden", [])) | set(contract.get("target_or_leakage_columns_forbidden", []))
    forbidden_overlap = sorted(set(FEATURE_ORDER) & (contract_forbidden | FORBIDDEN_FEATURES))
    if forbidden_overlap:
        raise ValueError(f"Forbidden columns in model feature order: {forbidden_overlap}")

    id_sets = {role: set(handoff.loc[handoff["model_role"].eq(role), "unique_post_id"].astype(str)) for role in valid_roles}
    for a in valid_roles:
        for b in valid_roles:
            if a < b and (id_sets[a] & id_sets[b]):
                raise ValueError(f"ID overlap between roles {a} and {b}")

    return {
        "total_rows": int(len(handoff)),
        "role_counts": {k:int(v) for k,v in role_counts.items()},
        "fold_counts": {str(int(k)):int(v) for k,v in folds.value_counts().sort_index().items()},
        "feature_count": len(FEATURE_ORDER),
        "group_separation": group_audit,
        "duplicate_ids": 0,
        "embargo_used": False,
        "train_only_fitting": True,
        "training_weight_column": project_config["weight_column"],
        "input_contract_passed": True,
    }
