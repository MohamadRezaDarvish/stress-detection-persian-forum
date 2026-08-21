
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import score_to_predicted_class  # noqa: E402
from member1_feature_engineering import Member1FeatureEngineer  # noqa: E402


def build_matrix(frame, bundle, profile):
    if profile == "scalar":
        required = ["member1_prediction", "member2_prediction"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing scalar inputs: {missing}")
        return frame[required], required, []

    engineer = Member1FeatureEngineer()
    missing_base = [
        column
        for column in ["member1_prediction", "member2_prediction"]
        if column not in frame.columns
    ]
    if missing_base:
        raise ValueError(f"Missing base predictions: {missing_base}")

    matrix = engineer.transform(frame)
    matrix["member1_prediction"] = pd.to_numeric(
        frame["member1_prediction"], errors="raise"
    ).to_numpy()
    matrix["member2_prediction"] = pd.to_numeric(
        frame["member2_prediction"], errors="raise"
    ).to_numpy()
    matrix["base_mean"] = (
        matrix["member1_prediction"] + matrix["member2_prediction"]
    ) / 2.0
    matrix["base_difference"] = (
        matrix["member2_prediction"] - matrix["member1_prediction"]
    )
    matrix["base_absolute_difference"] = np.abs(
        matrix["base_difference"]
    )
    features = bundle["features"]
    return matrix[features], features, bundle["categorical_features"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=str(THIS_DIR.parent))
    parser.add_argument(
        "--profile",
        choices=["primary", "scalar"],
        default="primary",
    )
    parser.add_argument(
        "--shap-output",
        help="Optional CSV path for local SHAP contributions.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    bundle = json.loads(
        (root / "models/fusion_bundle.json").read_text(encoding="utf-8")
    )
    frame = pd.read_csv(args.input)

    if args.profile == "primary":
        model_path = root / "models/fusion_selected.cbm"
        thresholds = bundle["thresholds"]
        profile = bundle["feature_profile"]
    else:
        model_path = root / "models/fusion_scalar_fallback.cbm"
        thresholds = bundle["fallback"]["thresholds"]
        profile = "scalar"

    matrix, features, categorical_features = build_matrix(
        frame, bundle, profile
    )
    model = CatBoostRegressor()
    model.load_model(model_path)
    prediction = np.clip(
        model.predict(matrix),
        *bundle["score_clip"],
    )
    predicted_class = score_to_predicted_class(prediction, thresholds)

    output = pd.DataFrame(
        {
            "member1_prediction": pd.to_numeric(
                frame["member1_prediction"], errors="raise"
            ).to_numpy(),
            "member2_prediction": pd.to_numeric(
                frame["member2_prediction"], errors="raise"
            ).to_numpy(),
            "fusion_stress_score": prediction,
            "fusion_clinical_class": predicted_class,
            "distance_to_low_moderate_threshold": prediction - thresholds[0],
            "distance_to_moderate_high_threshold": prediction - thresholds[1],
            "distance_to_high_very_high_threshold": prediction - thresholds[2],
        }
    )
    if "unique_post_id" in frame.columns:
        output.insert(0, "unique_post_id", frame["unique_post_id"].astype(str))
    output.to_csv(args.output, index=False, encoding="utf-8-sig")

    if args.shap_output:
        categorical_indices = [
            features.index(column)
            for column in categorical_features
        ]
        pool = Pool(matrix, cat_features=categorical_indices)
        shap = model.get_feature_importance(pool, type="ShapValues")
        records = []
        for row_index in range(len(frame)):
            contributions = sorted(
                zip(features, shap[row_index, :-1]),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            identifier = (
                str(frame.iloc[row_index]["unique_post_id"])
                if "unique_post_id" in frame.columns
                else str(row_index)
            )
            for rank, (feature, contribution) in enumerate(
                contributions[:15], start=1
            ):
                records.append(
                    {
                        "unique_post_id": identifier,
                        "rank": rank,
                        "feature": feature,
                        "shap_contribution": float(contribution),
                    }
                )
        pd.DataFrame(records).to_csv(
            args.shap_output,
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
