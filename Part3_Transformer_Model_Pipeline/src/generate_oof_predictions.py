"""Generate one deterministic OOF prediction per train row (held-out fold model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@torch.no_grad()
def det_predict(model, tokenizer, texts, max_length, batch_size=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        preds.append(logits.cpu().numpy())
    return np.concatenate(preds)


def generate_oof(
    train_df: pd.DataFrame,
    models_dir: Path,
    cfg: dict,
    max_length: int,
    device,
    out_path: Path,
):
    id_col = cfg["id_column"]
    fold_col = cfg["fold_column"]
    target_col = cfg["target_column"]
    num_folds = cfg["num_folds"]
    text_col = cfg["text_column"]

    rows = []
    for fold in range(num_folds):
        ckpt = models_dir / f"fold_{fold}"
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
        model.eval()
        sub = train_df[train_df[fold_col] == fold]
        preds = det_predict(model, tokenizer, sub[text_col].tolist(), max_length, device=device)
        for i in range(len(sub)):
            rows.append({
                id_col: sub.iloc[i][id_col],
                "fold": int(fold),
                "true_stress": float(sub.iloc[i][target_col]),
                "prediction": float(preds[i]),
            })
        del model
        torch.cuda.empty_cache()

    oof = pd.DataFrame(rows).sort_values(id_col).reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_path, index=False)
    return oof
