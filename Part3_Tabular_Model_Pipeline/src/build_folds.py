from __future__ import annotations
from pathlib import Path
import pandas as pd
from .common import save_json


def validate_and_save_folds(train_raw: pd.DataFrame, manifest: pd.DataFrame, root: Path, project_config: dict) -> dict:
    fold_col = project_config["fold_column"]
    id_col = project_config["id_column"]
    folds = train_raw[[id_col, fold_col, "group_id", "author", "thread_id"]].copy()
    folds[fold_col] = folds[fold_col].astype(int)
    if folds[id_col].duplicated().any():
        raise ValueError("Duplicate train ID in fold table")
    if set(folds[fold_col].unique()) != set(range(project_config["number_of_folds"])):
        raise ValueError("Fold set mismatch")
    counts = folds[fold_col].value_counts().sort_index().to_dict()
    out_dir = root / project_config["output_paths"]["folds"]
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / "member1_train_fold_assignments.csv", index=False)
    payload = {"fold_counts": {str(int(k)):int(v) for k,v in counts.items()}, "source": "modeling_manifest_v2.csv", "random_folds_created": False}
    save_json(payload, out_dir / "fold_manifest.json")
    return payload
