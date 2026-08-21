"""Optional GPU (Modal) prediction generation.

Reproduces the exact OOF + fold-ensemble predictions on a Modal A10G GPU — the
environment the production pipeline ran on.

The fold checkpoints and handoff CSVs are read from the Modal Volume
`persian-stress-vol` (they are placed there by `sync_to_volume()` below, or were
already uploaded during the original pipeline run).

Deploy once with:  modal deploy src/modal_generate.py
"""

from __future__ import annotations

from pathlib import Path

import modal

VOLUME_NAME = "persian-stress-vol"
REMOTE = "/data"
MODELS_REMOTE = "/data/outputs_final/fold_checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "numpy==1.24.3",
        "hazm==0.10.0",
        "safetensors",
        "sentencepiece",
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("member2-predict", image=image)


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={REMOTE: vol},
    memory=16384,
)
def generate_predictions(max_length: int = 384, batch_size: int = 32):
    """Run all 5 fold models on A10G; return OOF + val/test fold-ensembles."""
    import os

    import numpy as np
    import pandas as pd
    import torch
    from hazm import Normalizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda"

    hf = pd.read_csv(f"{REMOTE}/member2_handoff.csv")
    man = pd.read_csv(f"{REMOTE}/modeling_manifest_v2.csv")
    df = hf.merge(man[["unique_post_id"]], on="unique_post_id", how="left")
    df["content"] = df["content"].fillna("").astype(str)
    df["oof_fold"] = df["oof_fold"].astype("Int64")
    n = Normalizer()
    df["content_norm"] = df["content"].map(lambda t: n.normalize(str(t)))

    train_df = df[df["model_role"] == "train"].reset_index(drop=True)
    val_df = df[df["model_role"] == "validation"].reset_index(drop=True)
    test_df = df[df["model_role"] == "test"].reset_index(drop=True)

    def det_predict(model, tokenizer, texts):
        preds = []
        for i in range(0, len(texts), batch_size):
            b = texts[i : i + batch_size]
            enc = tokenizer(b, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1)
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            preds.append(logits.detach().cpu().numpy())
        return np.concatenate(preds)

    oof_rows = []
    fold_preds = {"validation": {}, "test": {}}
    for f in range(5):
        ckpt = f"{MODELS_REMOTE}/fold_{f}"
        tok = AutoTokenizer.from_pretrained(f"{ckpt}/tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
        model.eval()
        sub = train_df[train_df["oof_fold"] == f]
        p = det_predict(model, tok, sub["content_norm"].tolist())
        for i in range(len(sub)):
            oof_rows.append({
                "unique_post_id": sub.iloc[i]["unique_post_id"],
                "fold": int(f),
                "true_stress": float(sub.iloc[i]["final_stress"]),
                "prediction": float(p[i]),
            })
        for role, dfp in [("validation", val_df), ("test", test_df)]:
            fold_preds[role][f] = det_predict(model, tok, dfp["content_norm"].tolist())
        del model
        torch.cuda.empty_cache()

    oof = pd.DataFrame(oof_rows).sort_values("unique_post_id").reset_index(drop=True)

    outs = {}
    for role, dfp in [("validation", val_df), ("test", test_df)]:
        mat = np.stack([fold_preds[role][f] for f in range(5)], axis=1)
        out = pd.DataFrame({
            "unique_post_id": dfp["unique_post_id"].values,
            "model_role": role,
            "true_stress": dfp["final_stress"].values.astype(float),
            "prediction": mat.mean(axis=1),
            "prediction_std_fold_models": mat.std(axis=1),
            **{f"fold_model_{f}": fold_preds[role][f] for f in range(5)},
        })
        outs[role] = out

    return {
        "oof": oof.to_dict("records"),
        "validation": outs["validation"].to_dict("records"),
        "test": outs["test"].to_dict("records"),
    }


def sync_to_volume(project_root: Path) -> None:
    """Ensure handoff CSVs + fold checkpoints exist on the Modal volume.

    Idempotent: if the required files are already present (as in the original
    pipeline run) it does nothing. This is what makes the 'modal' backend
    reproducible from a fresh machine.
    """
    project_root = Path(project_root)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    try:
        root_entries = {e.path for e in vol.iterdir("/")}
    except Exception:
        root_entries = set()
    missing_csv = [f for f in ["member2_handoff.csv", "modeling_manifest_v2.csv"] if f not in root_entries]
    if missing_csv:
        # upload using batch_upload (small files)
        with vol.batch_upload() as batch:
            for fname in missing_csv:
                p = project_root / "data" / "handoff" / fname
                if p.exists():
                    batch.put_file(p, f"/{fname}")
        vol.commit()

    try:
        fold_entries = {
            e.path.removeprefix("outputs_final/fold_checkpoints/").removeprefix("/")
            for e in vol.iterdir("/outputs_final/fold_checkpoints")
        }
    except Exception:
        fold_entries = set()
    missing_folds = []
    for f in range(5):
        if f"fold_{f}/model.safetensors" not in fold_entries:
            src = project_root / "models" / f"fold_{f}"
            if (src / "model.safetensors").exists():
                missing_folds.append(f)

    if missing_folds:
        with vol.batch_upload() as batch:
            for f in missing_folds:
                src = project_root / "models" / f"fold_{f}"
                batch.put_file(src / "model.safetensors", f"/outputs_final/fold_checkpoints/fold_{f}/model.safetensors")
                batch.put_file(src / "config.json", f"/outputs_final/fold_checkpoints/fold_{f}/config.json")
                batch.put_file(src / "tokenizer" / "tokenizer.json", f"/outputs_final/fold_checkpoints/fold_{f}/tokenizer/tokenizer.json")
                batch.put_file(src / "tokenizer" / "tokenizer_config.json", f"/outputs_final/fold_checkpoints/fold_{f}/tokenizer/tokenizer_config.json")
        vol.commit()

    if not missing_csv and not missing_folds:
        print("volume already has all required data (no upload needed)")
    print("volume sync complete")


def generate_predictions_via_modal(max_length: int = 384):
    """Call the deployed Modal GPU function. Returns (oof_df, val_df, test_df)."""
    import pandas as pd

    fn = modal.Function.from_name("member2-predict", "generate_predictions")
    result = fn.remote(max_length=max_length)
    return (
        pd.DataFrame(result["oof"]),
        pd.DataFrame(result["validation"]),
        pd.DataFrame(result["test"]),
    )


def modal_available() -> bool:
    try:
        import subprocess

        r = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True, timeout=30)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False
