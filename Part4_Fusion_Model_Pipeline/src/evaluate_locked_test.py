from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    SCORE_CLIP,
    evaluate_predictions,
    json_safe_metrics,
    load_project_role,
    sha256_file,
    write_metrics_tables,
)
from member1_feature_engineering import Member1FeatureEngineer  # noqa: E402


CLASS_ORDER = ["Low", "Moderate", "High", "Very High"]


def prepare_test_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Recreate the exact feature matrix expected by the saved fusion model."""

    if features == ["member1_prediction", "member2_prediction"]:
        return frame[features]

    engineer = Member1FeatureEngineer()
    tabular = engineer.transform(frame)

    tabular["member1_prediction"] = pd.to_numeric(
        frame["member1_prediction"],
        errors="raise",
    ).to_numpy()

    tabular["member2_prediction"] = pd.to_numeric(
        frame["member2_prediction"],
        errors="raise",
    ).to_numpy()

    tabular["base_mean"] = (
        tabular["member1_prediction"]
        + tabular["member2_prediction"]
    ) / 2.0

    tabular["base_difference"] = (
        tabular["member2_prediction"]
        - tabular["member1_prediction"]
    )

    tabular["base_absolute_difference"] = np.abs(
        tabular["base_difference"]
    )

    return tabular[features]


def summarize_internal_targets(metrics: dict) -> dict:
    """
    Summarize the project-defined recall objectives.

    These are internal project targets, not regulatory, clinical, or general
    real-world deployment requirements.
    """

    recalls = metrics["classification"]["recall_by_class"]
    minimums = {
        "Low": 0.75,
        "Moderate": 0.50,
        "High": 0.50,
        "Very High": 0.75,
    }

    passed = [
        label
        for label in CLASS_ORDER
        if recalls[label] + 1e-12 >= minimums[label]
    ]

    failed = [
        label
        for label in CLASS_ORDER
        if recalls[label] + 1e-12 < minimums[label]
    ]

    if not failed:
        status = "all_internal_recall_targets_met"
    else:
        status = f"{len(passed)}_of_4_internal_recall_targets_met"

    return {
        "status": status,
        "all_targets_met": not failed,
        "passed_targets": passed,
        "failed_targets": failed,
        "passed_target_count": len(passed),
        "total_target_count": len(CLASS_ORDER),
        "minimum_recall": minimums,
        "observed_recall": recalls,
        "interpretation": (
            "These thresholds are project-defined evaluation objectives. "
            "Not meeting every target does not by itself determine whether "
            "the system can be used as a human-in-the-loop monitoring overlay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=str(THIS_DIR.parent),
    )

    parser.add_argument(
        "--confirm-lock",
        action="store_true",
        help="Required acknowledgement that the locked test is being evaluated.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a prior locked evaluation for exact reproduction.",
    )

    args = parser.parse_args()

    if not args.confirm_lock:
        raise SystemExit(
            "Refusing to evaluate test without --confirm-lock."
        )

    root = Path(args.project_root).resolve()
    outputs = root / "outputs"
    models = root / "models"
    marker = outputs / "locked_test_evaluation.json"

    outputs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    if marker.exists() and not args.force:
        raise SystemExit(
            "Locked test evaluation already exists. "
            "Use --force only for exact reproducibility checks."
        )

    bundle_path = models / "fusion_bundle.json"
    model_path = models / "fusion_selected.cbm"

    bundle = json.loads(
        bundle_path.read_text(encoding="utf-8")
    )

    test = load_project_role(root, "test")
    matrix = prepare_test_features(
        test,
        bundle["features"],
    )

    model = CatBoostRegressor()
    model.load_model(model_path)

    prediction = np.clip(
        model.predict(matrix),
        *SCORE_CLIP,
    )

    metrics = evaluate_predictions(
        test["final_stress"].to_numpy(float),
        test["clinical_class"].astype(str).to_numpy(),
        prediction,
        bundle["thresholds"],
    )

    project_target_summary = summarize_internal_targets(metrics)

    predictions = pd.DataFrame(
        {
            "unique_post_id": test["unique_post_id"].astype(str),
            "true_stress": test["final_stress"],
            "true_class": test["clinical_class"],
            "member1_prediction": test["member1_prediction"],
            "member2_prediction": test["member2_prediction"],
            "fusion_prediction": prediction,
            "predicted_class": metrics["predicted_class"],
        }
    )

    predictions.to_csv(
        outputs / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (outputs / "test_metrics.json").write_text(
        json.dumps(
            json_safe_metrics(metrics),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_metrics_tables(
        metrics,
        outputs / "test",
    )

    lock = {
        "status": "locked_test_evaluated",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": bundle["selected_candidate"],
        "thresholds": bundle["thresholds"],
        "model_sha256": sha256_file(model_path),
        "bundle_sha256_before_test_update": sha256_file(bundle_path),
        "test_prediction_sha256": sha256_file(
            outputs / "test_predictions.csv"
        ),
        "internal_project_target_status": project_target_summary["status"],
        "all_internal_recall_targets_met": project_target_summary[
            "all_targets_met"
        ],
        "passed_internal_recall_targets": project_target_summary[
            "passed_targets"
        ],
        "failed_internal_recall_targets": project_target_summary[
            "failed_targets"
        ],
    }

    marker.write_text(
        json.dumps(
            lock,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Preserve the locked evidence in the model bundle, but do not treat the
    # internal recall targets as universal deployment requirements.
    bundle["test_evaluated"] = True
    bundle["locked_test_metrics"] = json_safe_metrics(metrics)
    bundle["internal_project_target_status"] = project_target_summary[
        "status"
    ]
    bundle["internal_project_target_summary"] = project_target_summary

    bundle["deployment_status"] = (
        "candidate_for_human_in_the_loop_risk_monitoring_overlay"
    )

    bundle["deployment_interpretation"] = (
        "The model may be evaluated as a website risk-monitoring and "
        "prioritization overlay. It should support human review and must not "
        "be interpreted as an autonomous clinical diagnosis or an automatic "
        "medical-emergency decision system."
    )

    bundle["primary_safety_objective"] = {
        "class": "Very High",
        "recall": metrics["classification"]["recall_by_class"][
            "Very High"
        ],
        "precision": metrics["classification"]["precision_by_class"][
            "Very High"
        ],
        "internal_recall_target": 0.75,
        "internal_target_met": (
            metrics["classification"]["recall_by_class"]["Very High"]
            >= 0.75
        ),
        "interpretation": (
            "The principal project objective was to surface users who may "
            "be experiencing severe or dangerous stress for human review."
        ),
    }

    bundle_path.write_text(
        json.dumps(
            bundle,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    validation_metrics = json.loads(
        (outputs / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    validation_target_summary = summarize_internal_targets(
        validation_metrics
    )

    final_metrics = {
        "selected_candidate": bundle["selected_candidate"],
        "validation": validation_metrics,
        "test": json_safe_metrics(metrics),

        # Retained for compatibility with existing reports and notebooks.
        # These booleans refer only to the internal project recall objectives.
        "validation_gate": validation_target_summary["all_targets_met"],
        "test_gate": project_target_summary["all_targets_met"],

        "validation_internal_target_summary": validation_target_summary,
        "test_internal_target_summary": project_target_summary,

        "project_acceptance_status": project_target_summary["status"],

        "deployment_status": (
            "candidate_for_human_in_the_loop_risk_monitoring_overlay"
        ),

        "deployment_interpretation": (
            "The system can be evaluated as an overlay that continuously "
            "scores posts, aggregates user risk over time, and prioritizes "
            "high-risk users for moderator or professional review. "
            "It is not an autonomous diagnostic system."
        ),

        "primary_safety_objective": bundle[
            "primary_safety_objective"
        ],

        "further_work": [
            (
                "Improve Moderate-class separation, especially near the "
                "Low/Moderate and Moderate/High boundaries."
            ),
            (
                "Use longitudinal per-user aggregation such as time-decayed "
                "mean stress, recent maximum stress, repeated Very-High posts, "
                "and increasing-risk trends."
            ),
            (
                "Evaluate alert volume and false-positive burden in a "
                "moderator-facing pilot."
            ),
            (
                "Create a new untouched dual-annotated confirmation set "
                "before claiming improved generalization."
            ),
            (
                "Define consent, privacy, access-control and human-escalation "
                "protocols before real-world operation."
            ),
        ],
    }

    (outputs / "final_metrics.json").write_text(
        json.dumps(
            final_metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            lock,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()