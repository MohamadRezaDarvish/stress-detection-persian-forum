"""Reusable inference for the fine-tuned Persian stress model.

Usage:
    python predict.py --text "من خیلی استرس دارم" [--checkpoint outputs_final/model_checkpoint]
    python predict.py --csv new_posts.csv --text_col content --out new_predictions.csv

Produces:
    - continuous stress prediction (1-10 scale)
    - optional MC-dropout std (uncertainty)
    - optional CLS embedding (768-d) for fusion
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hazm import Normalizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_CKPT = Path(__file__).resolve().parent / "model_checkpoint"
DEFAULT_MAX_LENGTH = 384


def load_pipeline(checkpoint_path, max_length=DEFAULT_MAX_LENGTH):
    ckpt = Path(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt / "tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.eval()
    normalizer = Normalizer()
    return tokenizer, model, normalizer, max_length


def clean_text(text, normalizer):
    try:
        return normalizer.normalize(str(text))
    except Exception:
        return str(text)


@torch.no_grad()
def predict_texts(texts, tokenizer, model, normalizer, max_length=DEFAULT_MAX_LENGTH,
                  mc_trials=0, return_embeddings=False, clip_range=None):
    """Predict continuous stress for a list of texts.

    mc_trials > 0: run Monte-Carlo dropout and return (pred, std).
    return_embeddings=True: also return CLS 768-d vectors (extracted in eval mode).
    clip_range=(lo, hi): optionally clip predictions to [lo, hi] (explicit, off by default).

    NOTE: model train/eval state is always restored to its original value, and
    embeddings are always extracted with the model in eval mode (dropout off).
    """
    device = next(model.parameters()).device
    texts = [clean_text(t, normalizer) for t in texts]
    original_mode = model.training
    all_preds, all_stds, all_emb = [], [], []
    bs = 16
    try:
        if mc_trials > 0:
            model.train()  # enable dropout for MC uncertainty
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            if mc_trials > 0:
                trial_preds = []
                for _ in range(mc_trials):
                    logits = model(**enc).logits.squeeze(-1)
                    trial_preds.append(logits.detach().cpu().numpy())
                trial_preds = np.stack(trial_preds, axis=0)
                all_preds.append(trial_preds.mean(axis=0))
                all_stds.append(trial_preds.std(axis=0))
            else:
                logits = model(**enc).logits.squeeze(-1)
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
                all_preds.append(logits.detach().cpu().numpy())
                all_stds.append(np.zeros(len(batch), dtype=np.float32))
        # Always extract embeddings in eval mode (dropout off), regardless of MC
        if return_embeddings:
            model.eval()
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    batch = texts[i : i + bs]
                    enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length,
                                    return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    enc_out = model.base_model(**enc)
                    all_emb.append(enc_out.last_hidden_state[:, 0, :].detach().cpu().numpy())
    finally:
        # restore original train/eval state
        model.train(original_mode)

    preds = np.concatenate(all_preds)
    stds = np.concatenate(all_stds)
    if clip_range is not None:
        lo, hi = clip_range
        preds = np.clip(preds, lo, hi)
    emb = np.concatenate(all_emb, axis=0) if return_embeddings else None
    return preds, stds, emb


def main():
    ap = argparse.ArgumentParser(description="Persian stress regression inference")
    ap.add_argument("--text", type=str, default=None, help="single text to score")
    ap.add_argument("--csv", type=str, default=None, help="input CSV")
    ap.add_argument("--text_col", type=str, default="content", help="column with text")
    ap.add_argument("--id_col", type=str, default="unique_post_id", help="id column (optional)")
    ap.add_argument("--out", type=str, default="new_predictions.csv", help="output CSV path")
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="checkpoint dir")
    ap.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--mc_trials", type=int, default=0, help="MC-dropout trials for uncertainty (0=off)")
    ap.add_argument("--save_embeddings", action="store_true", help="also save CLS embeddings")
    ap.add_argument("--clip_min", type=float, default=None, help="optional explicit lower clip for predictions")
    ap.add_argument("--clip_max", type=float, default=None, help="optional explicit upper clip for predictions")
    args = ap.parse_args()

    clip_range = None
    if args.clip_min is not None or args.clip_max is not None:
        clip_range = (args.clip_min if args.clip_min is not None else -np.inf,
                      args.clip_max if args.clip_max is not None else np.inf)

    tokenizer, model, normalizer, ml = load_pipeline(args.checkpoint, args.max_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if args.text is not None:
        texts = [args.text]
        preds, stds, emb = predict_texts(texts, tokenizer, model, normalizer, ml,
                                         mc_trials=args.mc_trials,
                                         return_embeddings=args.save_embeddings,
                                         clip_range=clip_range)
        print("text:", args.text)
        print("prediction:", float(preds[0]))
        if args.mc_trials > 0:
            print("std:", float(stds[0]))
        if emb is not None:
            print("embedding shape:", emb.shape)
        return

    if args.csv is None:
        ap.error("provide --text or --csv")
    df = pd.read_csv(args.csv)
    texts = df[args.text_col].fillna("").astype(str).tolist()
    preds, stds, emb = predict_texts(texts, tokenizer, model, normalizer, ml,
                                     mc_trials=args.mc_trials,
                                     return_embeddings=args.save_embeddings,
                                     clip_range=clip_range)
    out = pd.DataFrame({"prediction": preds})
    if args.mc_trials > 0:
        out["prediction_std_optional"] = stds
    if args.id_col in df.columns:
        out.insert(0, args.id_col, df[args.id_col].values)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}")

    if args.save_embeddings:
        emb_path = Path(args.out).with_suffix(".embeddings.npy")
        np.save(emb_path, emb)
        print(f"wrote embeddings to {emb_path} shape {emb.shape}")


if __name__ == "__main__":
    main()
