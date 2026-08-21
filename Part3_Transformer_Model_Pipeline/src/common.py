"""Shared utilities: paths, seeds, config, JSON, metrics, class mapping."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_root(start: Path | None = None) -> Path:
    """Find the project root (the folder containing configs/, src/, notebooks/)."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return p  # fall back to cwd


def set_seed(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_project_config(root: Path | None = None):
    root = root or resolve_root()
    return load_json(root / "configs" / "project_config.json")


def load_model_config(root: Path | None = None):
    root = root or resolve_root()
    return load_json(root / "configs" / "model_config.json")


def clinical_class_from_stress(stress, thresholds=(3.0, 5.0, 7.0)):
    """Right-closed bins (authoritative):
    Low: stress <= 3; Moderate: 3 < stress <= 5; High: 5 < stress <= 7; Very High: stress > 7."""
    stress = np.asarray(stress, dtype=float)
    t1, t2, t3 = thresholds
    return np.select(
        [
            stress <= t1,
            (stress > t1) & (stress <= t2),
            (stress > t2) & (stress <= t3),
        ],
        ["Low", "Moderate", "High"],
        default="Very High",
    )


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def pearson_corr(y_true, y_pred):
    from scipy.stats import pearsonr

    return float(pearsonr(np.asarray(y_true), np.asarray(y_pred))[0])


def spearman_corr(y_true, y_pred):
    from scipy.stats import spearmanr

    return float(spearmanr(np.asarray(y_true), np.asarray(y_pred))[0])
