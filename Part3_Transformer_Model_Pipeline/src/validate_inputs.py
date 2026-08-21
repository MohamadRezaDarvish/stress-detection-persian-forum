"""Input contract validation. Raises ValueError with clear messages on violation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class InputContractError(ValueError):
    pass


def validate_inputs(df: pd.DataFrame, cfg: dict) -> dict:
    """Validate the merged handoff/manifest dataframe against the project contract.

    Returns a summary dict of sizes/checks. Raises InputContractError on failure.
    """
    id_col = cfg["id_column"]
    role_col = cfg["role_column"]
    fold_col = cfg["fold_column"]
    target_col = cfg["target_column"]
    text_col = cfg["text_column"]
    weight_col = cfg.get("weight_column", "training_sample_weight")

    errors = []

    # required columns
    required = [id_col, role_col, fold_col, target_col, text_col, weight_col, "clinical_class"]
    for c in required:
        if c not in df.columns:
            errors.append(f"missing required column: {c}")
    if errors:
        raise InputContractError("; ".join(errors))

    # IDs
    if df[id_col].isna().any():
        errors.append("null unique_post_id present")
    if df[id_col].duplicated().any():
        errors.append("duplicate unique_post_id present")

    # roles
    allowed_roles = {cfg["train_role"], cfg["validation_role"], cfg["test_role"], cfg["embargo_role"]}
    actual_roles = set(df[role_col].dropna().unique())
    if not actual_roles.issubset(allowed_roles):
        errors.append(f"unexpected model_role values: {actual_roles - allowed_roles}")

    # text
    if df[text_col].isna().any():
        errors.append("null content present")

    # target
    labeled = df[df[role_col].isin([cfg["train_role"], cfg["validation_role"], cfg["test_role"]])]
    if labeled[target_col].isna().any():
        errors.append("null final_stress in labeled roles")
    rng = cfg.get("prediction_range", [1.0, 10.0])
    if labeled[target_col].min() < rng[0] or labeled[target_col].max() > rng[1]:
        errors.append(
            f"final_stress out of expected range {rng}: [{labeled[target_col].min()}, {labeled[target_col].max()}]"
        )

    # folds: only train rows have folds; each train row has exactly one fold 0..4
    train = df[df[role_col] == cfg["train_role"]]
    if train[fold_col].isna().any():
        errors.append("train rows missing oof_fold")
    fold_vals = sorted(train[fold_col].dropna().unique())
    if set(fold_vals) != set(range(cfg["num_folds"])):
        errors.append(f"train oof_fold values != 0..{cfg['num_folds']-1}: {fold_vals}")

    # role separation
    for r in [cfg["validation_role"], cfg["test_role"], cfg["embargo_role"]]:
        sub = df[df[role_col] == r]
        if sub[fold_col].notna().any():
            errors.append(f"{r} rows should not have oof_fold")

    # no overlap between roles
    ids_by_role = {r: set(df.loc[df[role_col] == r, id_col]) for r in allowed_roles}
    for a in allowed_roles:
        for b in allowed_roles:
            if a < b and len(ids_by_role[a] & ids_by_role[b]) > 0:
                errors.append(f"ID overlap between roles {a} and {b}")

    # clinical class consistency with target (REPORT ONLY — clinical_class is authoritative
    # and the frozen data contains some boundary noise, so mismatch is not an error)
    class_match = None
    try:
        from src.common import clinical_class_from_stress

        derived = clinical_class_from_stress(labeled[target_col].values, cfg.get("class_thresholds", (3, 5, 7)))
        class_match = float((derived == labeled["clinical_class"].values).mean())
    except Exception:
        pass

    if errors:
        raise InputContractError("Input validation failed:\n- " + "\n- ".join(errors))

    return {
        "total_rows": int(len(df)),
        "train": int(train_mask_count(df, cfg)),
        "validation": int((df[role_col] == cfg["validation_role"]).sum()),
        "test": int((df[role_col] == cfg["test_role"]).sum()),
        "embargo": int((df[role_col] == cfg["embargo_role"]).sum()),
        "fold_sizes": {int(f): int(n) for f, n in train[fold_col].value_counts().sort_index().items()},
        "unique_ids": int(df[id_col].nunique()),
        "clinical_class_stress_match": class_match,
    }


def train_mask_count(df: pd.DataFrame, cfg: dict) -> int:
    return int((df[cfg["role_column"]] == cfg["train_role"]).sum())


def check_no_overlap(df: pd.DataFrame, cfg: dict) -> None:
    """Assert train/val/test/embargo ID sets are disjoint."""
    id_col = cfg["id_column"]
    role_col = cfg["role_column"]
    sets = {r: set(df.loc[df[role_col] == r, id_col]) for r in df[role_col].unique()}
    keys = list(sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = sets[keys[i]] & sets[keys[j]]
            if inter:
                raise InputContractError(f"role overlap {keys[i]} & {keys[j]}: {len(inter)} ids")
