"""Generate the 5-fold ensemble test predictions (locked test, no tuning)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.generate_oof_predictions import det_predict


def generate_test_ensemble(
    test_df: pd.DataFrame,
    models_dir: Path,
    cfg: dict,
    max_length: int,
    device,
    out_path: Path,
):
    id_col = cfg["id_column"]
    role_col = cfg["role_column"]
    target_col = cfg["target_column"]
    text_col = cfg["text_column"]
    num_folds = cfg["num_folds"]

    col_map = {}
    for fold in range(num_folds):
        ckpt = models_dir / f"fold_{fold}"
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
        model.eval()
        preds = det_predict(model, tokenizer, test_df[text_col].tolist(), max_length, device=device)
        col_map[f"fold_model_{fold}"] = preds
        del model
        torch.cuda.empty_cache()

    mat = np.stack([col_map[f"fold_model_{f}"] for f in range(num_folds)], axis=1)
    final = mat.mean(axis=1)
    std = mat.std(axis=1)

    out = pd.DataFrame({
        id_col: test_df[id_col].values,
        role_col: cfg["test_role"],
        "true_stress": test_df[target_col].values.astype(float),
        "prediction": final,
        "prediction_std_fold_models": std,
        **{f"fold_model_{f}": col_map[f"fold_model_{f}"] for f in range(num_folds)},
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
