from __future__ import annotations
import numpy as np
import pandas as pd


def generate_oof_predictions(models: list, prepared: dict, project_config: dict) -> tuple[pd.DataFrame, np.ndarray]:
    X = prepared["X"]["train"]
    raw = prepared["raw"]["train"]
    folds = raw[project_config["fold_column"]].astype(int)
    pred = np.full(len(raw), np.nan, dtype=float)
    for fold, model in enumerate(models):
        mask = folds.eq(fold)
        pred[mask.to_numpy()] = model.predict(X.loc[mask])
    lo, hi = project_config["prediction_range"]
    pred = np.clip(pred, lo, hi)
    if np.isnan(pred).any():
        raise RuntimeError("Incomplete OOF prediction coverage")
    output = pd.DataFrame({
        "unique_post_id": raw[project_config["id_column"]].astype(str),
        "fold": folds,
        "true_stress": raw[project_config["target_column"]].astype(float),
        "prediction": pred,
        "prediction_std_optional": np.nan,
    })
    return output, pred
