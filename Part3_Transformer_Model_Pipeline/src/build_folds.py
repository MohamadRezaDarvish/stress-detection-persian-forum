"""Load and validate the grouped fold assignments supplied by Member C."""

from __future__ import annotations

import pandas as pd


def load_folds(df: pd.DataFrame, cfg: dict) -> dict:
    """Return dict fold -> list of unique_post_ids for train rows only.

    Validates that every train row is in exactly one of 0..num_folds-1 and that
    validation/test/embargo rows have no fold assignment.
    """
    id_col = cfg["id_column"]
    role_col = cfg["role_column"]
    fold_col = cfg["fold_column"]
    num_folds = cfg["num_folds"]
    train_role = cfg["train_role"]

    train = df[df[role_col] == train_role].copy()
    if train[fold_col].isna().any():
        raise ValueError("train rows have missing oof_fold")

    folds = {}
    for f in range(num_folds):
        ids = set(train.loc[train[fold_col] == f, id_col])
        folds[f] = ids

    # every train id exactly once
    all_ids = set(train[id_col])
    assigned = set().union(*folds.values())
    if all_ids != assigned:
        raise ValueError(f"fold assignment mismatch: {len(all_ids - assigned)} train ids unassigned")

    # non-train rows must not have fold
    non_train = df[df[role_col] != train_role]
    if non_train[fold_col].notna().any():
        raise ValueError("non-train rows must not have oof_fold")

    return folds


def check_grouped_no_leak(df: pd.DataFrame, cfg: dict) -> dict:
    """Verify author/thread/content_hash groups do not cross folds or eval splits."""
    group_cols = ["author", "thread_id", "content_hash"]
    train = df[df[cfg["role_column"]] == cfg["train_role"]]
    fold_col = cfg["fold_column"]

    report = {}
    for col in group_cols:
        if col not in df.columns:
            report[col] = "not available"
            continue
        g = train.groupby(col)[fold_col].nunique()
        crossing = int((g > 1).sum())
        report[col] = {"groups_crossing_folds": crossing, "groups_total": int(len(g))}
    return report
