"""Explainability for the Persian stress model.

IMPORTANT: No SHAP/Integrated-Gradients was used in the production pipeline.
Explainability here is (1) model-level error analysis and (2) a gradient-based
token-attribution approximation for a few example posts, produced AFTER training
for interpretation/reporting only. It was NOT used to select features or tune the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def example_token_attribution(text, model, tokenizer, max_length=384, device=None, n_steps=20):
    """Gradient x input token attribution for one example.

    Returns list of (token_str, attribution) pairs. This is a simple gradient-based
    importance score, not a calibrated SHAP estimate.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    # baseline = zero embeddings; ramp alpha 0->1, accumulate grads w.r.t. input embeddings
    emb = model.base_model.embeddings(enc["input_ids"])  # (1, T, H) leaf? no - detach+clone to make leaf
    emb = emb.detach().clone().requires_grad_(True)
    grad_accum = torch.zeros_like(emb)

    for alpha in np.linspace(0.0, 1.0, n_steps):
        scaled = emb * alpha
        mask = enc["attention_mask"]
        out = model.base_model(inputs_embeds=scaled, attention_mask=mask)
        logit = model.classifier(out.last_hidden_state[:, 0]).squeeze()
        model.zero_grad()
        logit.backward(retain_graph=True)
        grad_accum += emb.grad * alpha  # path-integrated: grad * step-weight

    attrib = (emb.detach() * grad_accum).sum(dim=-1).squeeze().cpu().numpy()
    attrs = [(t, float(a)) for t, a in zip(tokens, attrib) if t not in ("[PAD]", "[CLS]", "[SEP]")]
    attrs.sort(key=lambda x: -abs(x[1]))
    return attrs


def run_error_analysis(df_pred: pd.DataFrame, cfg: dict) -> dict:
    """Model-level error analysis: residuals by stress band and under/over prediction."""
    y = df_pred["true_stress"].values.astype(float)
    p = df_pred["prediction"].values.astype(float)
    err = p - y
    under = err < 0
    over = err > 0

    bands = cfg["class_names"]
    band_err = {}
    for b in bands:
        idx = (df_pred["clinical_class"] == b).values if "clinical_class" in df_pred.columns else None
        if idx is not None and idx.sum() > 0:
            band_err[b] = {
                "n": int(idx.sum()),
                "mean_error": float(err[idx].mean()),
                "mean_abs_error": float(np.abs(err[idx]).mean()),
                "pct_under": float(100 * (err[idx] < 0).mean()),
            }

    return {
        "pct_under_total": float(100 * under.mean()),
        "pct_over_total": float(100 * over.mean()),
        "mean_abs_err_under": float(np.abs(err[under]).mean()) if under.any() else 0.0,
        "mean_abs_err_over": float(np.abs(err[over]).mean()) if over.any() else 0.0,
        "band_error": band_err,
    }


def save_explainability(report: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
