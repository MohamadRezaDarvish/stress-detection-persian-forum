
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_zscore(values):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if mad <= 1e-12:
        return np.zeros_like(values)
    return (values - median) / (1.4826 * mad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.project_root)
    bundle = json.loads(
        (root / "models/fusion_bundle.json").read_text(encoding="utf-8")
    )
    frame = pd.read_csv(args.predictions)

    required = [
        "fusion_stress_score",
        "member1_prediction",
        "member2_prediction",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing active-learning fields: {missing}")

    thresholds = np.asarray(bundle["thresholds"], dtype=float)
    score = frame["fusion_stress_score"].to_numpy(float)
    disagreement = np.abs(
        frame["member1_prediction"].to_numpy(float)
        - frame["member2_prediction"].to_numpy(float)
    )
    boundary_distance = np.min(
        np.abs(score[:, None] - thresholds[None, :]),
        axis=1,
    )
    boundary_uncertainty = np.exp(-boundary_distance / 0.40)
    tail_priority = 1.0 / (
        1.0 + np.exp(-(score - thresholds[2]) / 0.50)
    )

    optional_uncertainty = (
        frame["prediction_uncertainty"].to_numpy(float)
        if "prediction_uncertainty" in frame.columns
        else np.zeros(len(frame))
    )

    acquisition = (
        0.35 * robust_zscore(disagreement)
        + 0.35 * boundary_uncertainty
        + 0.20 * robust_zscore(optional_uncertainty)
        + 0.10 * tail_priority
    )
    result = frame.copy()
    result["base_model_disagreement"] = disagreement
    result["nearest_threshold_distance"] = boundary_distance
    result["active_learning_score"] = acquisition
    result.sort_values(
        "active_learning_score",
        ascending=False,
        inplace=True,
    )
    result.head(args.top_n).to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
