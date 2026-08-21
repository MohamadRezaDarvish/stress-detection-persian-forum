"""Artifact verification: files, IDs, shapes, folds, predictions, coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class VerificationError(ValueError):
    pass


def verify_models(models_dir: Path, num_folds: int) -> list:
    missing = []
    for f in range(num_folds):
        ckpt = models_dir / f"fold_{f}"
        if not (ckpt / "model.safetensors").exists():
            missing.append(f"fold_{f}")
    if missing:
        raise VerificationError(f"missing fold checkpoints: {missing}")
    return [f"fold_{f}" for f in range(num_folds)]


def verify_predictions(oof, val, test, train_ids, val_ids, test_ids, embargo_ids, cfg) -> dict:
    id_col = cfg["id_column"]
    rng = cfg.get("prediction_range", [1.0, 10.0])

    checks = {}

    # OOF coverage
    oof_ids = set(oof[id_col])
    train_set = set(train_ids)
    checks["oof_all_train_ids_present"] = oof_ids == train_set
    checks["oof_no_duplicates"] = oof[id_col].duplicated().sum() == 0
    checks["oof_no_val_ids"] = len(oof_ids & set(val_ids)) == 0
    checks["oof_no_test_ids"] = len(oof_ids & set(test_ids)) == 0
    checks["oof_no_embargo_ids"] = len(oof_ids & set(embargo_ids)) == 0
    checks["oof_rows"] = int(len(oof))

    # val/test coverage
    checks["val_ids_match"] = set(val[id_col]) == set(val_ids)
    checks["test_ids_match"] = set(test[id_col]) == set(test_ids)

    # numeric + range
    # Predictions are intentionally NOT clipped to [1,10] (matches production), so the
    # check allows a small tolerance below 1.0 and above 10.0 for unclipped outputs.
    lo, hi = rng[0] - 1.0, rng[1] + 1.0
    for name, df in [("oof", oof), ("val", val), ("test", test)]:
        p = df["prediction"].values
        checks[f"{name}_numeric"] = pd.api.types.is_numeric_dtype(pd.Series(p))
        checks[f"{name}_within_range"] = bool((p >= lo).all() and (p <= hi).all())

    # ensemble consistency (val/test prediction == mean of fold_model_0..4)
    for name, df in [("val", val), ("test", test)]:
        cols = [c for c in df.columns if c.startswith("fold_model_")]
        if len(cols) == 5:
            manual = df[cols].values.mean(axis=1)
            checks[f"{name}_ensemble_is_mean"] = bool(
                np.allclose(manual, df["prediction"].values, atol=1e-5)
            )

    failed = {k: v for k, v in checks.items() if v is False}
    if failed:
        raise VerificationError(f"prediction verification failed: {failed}")
    return checks


def verify_metrics_files(metrics_dir: Path, required: list) -> None:
    missing = [r for r in required if not (Path(metrics_dir) / r).exists()]
    if missing:
        raise VerificationError(f"missing metrics files: {missing}")


def verify_figures(figures_dir: Path, required: list) -> None:
    missing = [r for r in required if not (Path(figures_dir) / r).exists()]
    if missing:
        raise VerificationError(f"missing figure files: {missing}")


def make_checksum_file(out_path: Path, files: list) -> str:
    import hashlib

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        p = Path(f)
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            rows.append({"file": str(p), "sha256": h.hexdigest(), "size": p.stat().st_size})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return str(out_path)
