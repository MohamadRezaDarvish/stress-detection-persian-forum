"""Unified prediction generation dispatcher.

backends:
  "modal"  -> run all 5 fold models on Modal A10G (fast, exact; requires Modal auth)
  "cpu"    -> run all 5 fold models locally on CPU (slow but correct)
  "load"   -> load already-generated deterministic predictions from data/handoff/
              (used when regenerating is unnecessary or GPU unavailable)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_all_predictions(
    df: pd.DataFrame,
    cfg: dict,
    model_config: dict,
    models_dir: Path,
    backend: str,
    device: str,
    handoff_dir: Path,
    project_root: Path,
) -> dict:
    """Returns dict with keys oof, validation, test (DataFrames)."""
    if backend == "modal":
        from src.modal_generate import (
            generate_predictions_via_modal,
            modal_available,
            sync_to_volume,
        )

        if not modal_available():
            raise RuntimeError(
                "backend='modal' requested but Modal is not authenticated. "
                "Run 'modal token set' or use backend='cpu'/'load'."
            )
        sync_to_volume(project_root)
        oof, val, test = generate_predictions_via_modal(
            max_length=model_config["max_length"],
        )
        return {"oof": oof, "validation": val, "test": test}

    if backend == "load":
        oof = pd.read_csv(handoff_dir / "member2_oof_predictions_deterministic.csv")
        val = pd.read_csv(handoff_dir / "member2_validation_predictions_fold_ensemble.csv")
        test = pd.read_csv(handoff_dir / "member2_test_predictions_fold_ensemble.csv")
        return {"oof": oof, "validation": val, "test": test}

    if backend == "cpu":
        from src.generate_oof_predictions import generate_oof
        from src.generate_validation_predictions import generate_validation_ensemble
        from src.generate_test_predictions import generate_test_ensemble

        train_df = df[df["model_role"] == cfg["train_role"]].reset_index(drop=True)
        val_df = df[df["model_role"] == cfg["validation_role"]].reset_index(drop=True)
        test_df = df[df["model_role"] == cfg["test_role"]].reset_index(drop=True)

        oof = generate_oof(
            train_df, models_dir, cfg, model_config["max_length"], device,
            Path(cfg["output_paths"]["predictions"]) / "member2_oof_predictions_deterministic.csv",
        )
        val = generate_validation_ensemble(
            val_df, models_dir, cfg, model_config["max_length"], device,
            Path(cfg["output_paths"]["predictions"]) / "member2_validation_predictions_fold_ensemble.csv",
        )
        test = generate_test_ensemble(
            test_df, models_dir, cfg, model_config["max_length"], device,
            Path(cfg["output_paths"]["predictions"]) / "member2_test_predictions_fold_ensemble.csv",
        )
        return {"oof": oof, "validation": val, "test": test}

    raise ValueError(f"unknown backend: {backend}")


def pick_backend(modal_ok: bool, regenerated_files_exist: bool, requested: str | None = None) -> str:
    if requested is not None:
        return requested
    if modal_ok:
        return "modal"
    if regenerated_files_exist:
        return "load"
    return "cpu"
