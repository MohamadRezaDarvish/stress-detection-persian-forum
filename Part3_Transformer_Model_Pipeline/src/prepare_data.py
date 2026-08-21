"""Data loading, Persian text cleaning, and tokenization preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hazm import Normalizer


def load_handoff(inputs_dir: Path) -> pd.DataFrame:
    handoff = pd.read_csv(inputs_dir / "member2_handoff.csv")
    manifest = pd.read_csv(inputs_dir / "modeling_manifest_v2.csv")
    cols = [
        "unique_post_id",
        "group_id",
        "author",
        "thread_id",
        "content_hash",
        "official_eval",
        "use_for_training",
        "use_for_model_selection",
        "use_for_final_test",
        "exclude_from_modeling",
    ]
    df = handoff.merge(manifest[cols], on="unique_post_id", how="left")
    df["content"] = df["content"].fillna("").astype(str)
    df["oof_fold"] = df["oof_fold"].astype("Int64")
    return df


def clean_persian_text(text: str, normalizer: Normalizer | None = None) -> str:
    """Persian text normalization using hazm.Normalizer.

    Handles: ي/ي and ك/ك (Arabic-to-Persian), ZWNJ (نیمفاصله) consistency,
    extra whitespace, and misc punctuation normalization.
    """
    if normalizer is None:
        normalizer = Normalizer()
    try:
        return normalizer.normalize(str(text))
    except Exception:
        return str(text)


def make_normalizer() -> Normalizer:
    return Normalizer()


def tokenize_texts(texts, tokenizer, max_length: int, batch_size: int = 256, return_pt: bool = False):
    """Batch tokenize with truncation + padding to max_length."""
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt" if return_pt else None,
    )
    return enc


def token_length_stats(texts, tokenizer) -> dict:
    enc = tokenizer(list(texts), truncation=False, padding=False, add_special_tokens=True)
    lens = np.array([len(x) for x in enc["input_ids"]])
    return {
        p: float(np.percentile(lens, p))
        for p in (50, 90, 95, 99, 100)
    }
