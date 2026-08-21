
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    SCORE_CLIP,
    evaluate_predictions,
    json_safe_metrics,
    load_project_role,
)
from member1_feature_engineering import (
    CATEGORICAL_FEATURES,
    Member1FeatureEngineer,
)


def prepare(frame):
    engineer = Member1FeatureEngineer()
    matrix = engineer.transform(frame)
    matrix["member1_prediction"] = frame["member1_prediction"].to_numpy(float)
    matrix["member2_prediction"] = frame["member2_prediction"].to_numpy(float)
    matrix["base_mean"] = (
        matrix["member1_prediction"] + matrix["member2_prediction"]
    ) / 2.0
    matrix["base_difference"] = (
        matrix["member2_prediction"] - matrix["member1_prediction"]
    )
    matrix["base_absolute_difference"] = np.abs(matrix["base_difference"])
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(THIS_DIR.parent))
    args = parser.parse_args()

    root = Path(args.project_root)
    bundle = json.loads(
        (root / "models/fusion_bundle.json").read_text(encoding="utf-8")
    )
    train = load_project_role(root, "train")
    matrix = prepare(train)
    features = bundle["features"]
    categorical_indices = [
        features.index(column)
        for column in bundle["categorical_features"]
    ]
    prediction = np.full(len(train), np.nan, dtype=float)

    config = bundle["candidate_config"]
    for fold in range(5):
        fit_mask = train["oof_fold"].ne(fold).to_numpy()
        holdout_mask = train["oof_fold"].eq(fold).to_numpy()
        model = CatBoostRegressor(
            random_seed=20260804 + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
            **config,
        )
        model.fit(
            matrix.loc[fit_mask, features],
            train.loc[fit_mask, "final_stress"],
            sample_weight=train.loc[fit_mask, "training_sample_weight"],
            cat_features=categorical_indices,
        )
        prediction[holdout_mask] = np.clip(
            model.predict(matrix.loc[holdout_mask, features]),
            *SCORE_CLIP,
        )

    if not np.isfinite(prediction).all():
        raise RuntimeError("Fusion OOF generation left invalid rows.")

    metrics = evaluate_predictions(
        train["final_stress"].to_numpy(float),
        train["clinical_class"].astype(str).to_numpy(),
        prediction,
        bundle["thresholds"],
    )
    output = pd.DataFrame(
        {
            "unique_post_id": train["unique_post_id"].astype(str),
            "fold": train["oof_fold"].astype(int),
            "true_stress": train["final_stress"],
            "true_class": train["clinical_class"],
            "fusion_oof_prediction": prediction,
            "predicted_class_using_validation_thresholds": metrics["predicted_class"],
        }
    )
    output.to_csv(
        root / "outputs/fusion_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (root / "outputs/fusion_oof_metrics.json").write_text(
        json.dumps(json_safe_metrics(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete",
        "rows": len(output),
        "mae": metrics["continuous"]["mae"],
        "pearson": metrics["continuous"]["pearson"],
    }, indent=2))


if __name__ == "__main__":
    main()
