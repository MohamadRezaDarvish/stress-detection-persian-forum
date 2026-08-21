
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from common import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    args = parser.parse_args()
    root = Path(args.project_root)

    required = [
        root / "models/fusion_selected.cbm",
        root / "models/fusion_scalar_fallback.cbm",
        root / "models/fusion_bundle.json",
        root / "outputs/candidate_leaderboard.csv",
        root / "outputs/validation_metrics.json",
        root / "outputs/test_metrics.json",
        root / "outputs/final_metrics.json",
        root / "outputs/validation_predictions.csv",
        root / "outputs/test_predictions.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")

    bundle = json.loads(
        (root / "models/fusion_bundle.json").read_text(encoding="utf-8")
    )
    thresholds = bundle["thresholds"]
    if not (thresholds[0] < thresholds[1] < thresholds[2]):
        raise ValueError("Thresholds are not strictly ordered.")

    for split in ["validation", "test"]:
        frame = pd.read_csv(root / f"outputs/{split}_predictions.csv")
        prediction = pd.to_numeric(
            frame["fusion_prediction"], errors="raise"
        ).to_numpy()
        if not np.isfinite(prediction).all():
            raise ValueError(f"{split} contains non-finite predictions.")
        if frame["unique_post_id"].duplicated().any():
            raise ValueError(f"{split} contains duplicate IDs.")

    model = CatBoostRegressor()
    model.load_model(root / "models/fusion_selected.cbm")

    tracked = [
        *required,
        root / "outputs/global_feature_importance.csv",
        root / "outputs/validation_shap_summary.csv",
        root / "outputs/locked_test_evaluation.json",
    ]
    checksums = {
        str(path.relative_to(root)): sha256_file(path)
        for path in tracked
        if path.exists()
    }
    (root / "outputs/checksums.json").write_text(
        json.dumps(checksums, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "valid", "files_checked": len(checksums)}, indent=2))


if __name__ == "__main__":
    main()
