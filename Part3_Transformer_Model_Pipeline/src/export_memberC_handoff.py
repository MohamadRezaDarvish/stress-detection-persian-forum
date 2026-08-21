"""Export the exact Member C fusion handoff files (deterministic predictions)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_handoff(
    oof_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    id_col = cfg["id_column"]
    role_col = cfg["role_column"]

    # OOF: unique_post_id, fold, true_stress, prediction
    oof_out = oof_df[[id_col, "fold", "true_stress", "prediction"]].copy()
    oof_path = out_dir / "member2_oof_predictions_deterministic.csv"
    oof_out.to_csv(oof_path, index=False)

    # Validation: fold ensemble + per-fold columns
    val_out = val_df[[
        id_col, role_col, "true_stress", "prediction", "prediction_std_fold_models",
        *[f"fold_model_{f}" for f in range(cfg["num_folds"])],
    ]].copy()
    val_path = out_dir / "member2_validation_predictions_fold_ensemble.csv"
    val_out.to_csv(val_path, index=False)

    # Test: fold ensemble + per-fold columns
    test_out = test_df[[
        id_col, role_col, "true_stress", "prediction", "prediction_std_fold_models",
        *[f"fold_model_{f}" for f in range(cfg["num_folds"])],
    ]].copy()
    test_path = out_dir / "member2_test_predictions_fold_ensemble.csv"
    test_out.to_csv(test_path, index=False)

    return {
        "oof_path": str(oof_path),
        "validation_path": str(val_path),
        "test_path": str(test_path),
        "oof_rows": int(len(oof_out)),
        "val_rows": int(len(val_out)),
        "test_rows": int(len(test_out)),
    }
