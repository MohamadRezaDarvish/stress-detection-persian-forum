
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

CLASS_ORDER = ["Low", "Moderate", "High", "Very High"]
MINIMUM_RECALL = np.array([0.75, 0.50, 0.50, 0.75], dtype=float)
SCORE_CLIP = (1.0, 10.0)


def normalize_id(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(
        str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})
    )
    return re.sub(r"\s+", " ", text).strip()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_to_predicted_class(
    prediction: np.ndarray,
    thresholds: list[float] | tuple[float, float, float],
) -> np.ndarray:
    labels = np.asarray(CLASS_ORDER, dtype=object)
    return labels[np.digitize(prediction, thresholds, right=False)]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def evaluate_predictions(
    true_stress: np.ndarray,
    true_class: np.ndarray,
    prediction: np.ndarray,
    thresholds: list[float],
) -> dict:
    true_stress = np.asarray(true_stress, dtype=float)
    true_class = np.asarray(true_class, dtype=object)
    prediction = np.asarray(prediction, dtype=float)
    predicted_class = score_to_predicted_class(prediction, thresholds)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_class,
        predicted_class,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    matrix = confusion_matrix(true_class, predicted_class, labels=CLASS_ORDER)

    very_high_truth = (true_class == "Very High").astype(int)
    high_or_very_high = np.isin(true_class, ["High", "Very High"])
    underprediction = prediction < true_stress

    recall_ci = {
        CLASS_ORDER[index]: wilson_interval(int(matrix[index, index]), int(support[index]))
        for index in range(4)
    }

    return {
        "continuous": {
            "mae": float(mean_absolute_error(true_stress, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(true_stress, prediction))),
            "pearson": float(pearsonr(true_stress, prediction).statistic),
            "spearman": float(spearmanr(true_stress, prediction).statistic),
            "prediction_min": float(np.min(prediction)),
            "prediction_max": float(np.max(prediction)),
        },
        "classification": {
            "thresholds": [float(value) for value in thresholds],
            "accuracy": float(accuracy_score(true_class, predicted_class)),
            "macro_f1": float(
                f1_score(
                    true_class,
                    predicted_class,
                    labels=CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                )
            ),
            "precision_by_class": dict(zip(CLASS_ORDER, map(float, precision))),
            "recall_by_class": dict(zip(CLASS_ORDER, map(float, recall))),
            "f1_by_class": dict(zip(CLASS_ORDER, map(float, f1))),
            "support_by_class": dict(zip(CLASS_ORDER, map(int, support))),
            "recall_wilson_95": recall_ci,
            "confusion_matrix": matrix.tolist(),
            "constraints_pass": bool(np.all(recall + 1e-12 >= MINIMUM_RECALL)),
            "failed_recall_constraints": [
                CLASS_ORDER[index]
                for index in range(4)
                if recall[index] + 1e-12 < MINIMUM_RECALL[index]
            ],
        },
        "safety": {
            "very_high_pr_auc": float(
                average_precision_score(very_high_truth, prediction)
            ),
            "high_or_very_high_underprediction_rate": float(
                underprediction[high_or_very_high].mean()
            ),
            "very_high_underprediction_rate": float(
                underprediction[true_class == "Very High"].mean()
            ),
        },
        "predicted_class": predicted_class,
    }


def write_metrics_tables(metrics: dict, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    class_rows = []
    for label in CLASS_ORDER:
        class_rows.append(
            {
                "clinical_class": label,
                "precision": metrics["classification"]["precision_by_class"][label],
                "recall": metrics["classification"]["recall_by_class"][label],
                "f1": metrics["classification"]["f1_by_class"][label],
                "support": metrics["classification"]["support_by_class"][label],
                "recall_ci_low": metrics["classification"]["recall_wilson_95"][label][0],
                "recall_ci_high": metrics["classification"]["recall_wilson_95"][label][1],
                "minimum_required_recall": float(
                    MINIMUM_RECALL[CLASS_ORDER.index(label)]
                ),
            }
        )
    pd.DataFrame(class_rows).to_csv(
        output_prefix.with_name(output_prefix.name + "_per_class.csv"),
        index=False,
    )
    pd.DataFrame(
        metrics["classification"]["confusion_matrix"],
        index=CLASS_ORDER,
        columns=CLASS_ORDER,
    ).to_csv(
        output_prefix.with_name(output_prefix.name + "_confusion_matrix.csv")
    )


def json_safe_metrics(metrics: dict) -> dict:
    result = dict(metrics)
    result.pop("predicted_class", None)
    return result


def load_project_role(project_root: str | Path, role: str) -> pd.DataFrame:
    root = Path(project_root)
    data_dir = root / "data"

    manifest = pd.read_csv(
        data_dir / "modeling_manifest_v2.csv",
        dtype={"unique_post_id": "string"},
    )
    handoff = pd.read_csv(
        data_dir / "member1_handoff.csv.gz",
        compression="gzip",
        dtype={"unique_post_id": "string"},
    )

    manifest["join_id"] = manifest["unique_post_id"].map(normalize_id)
    handoff["join_id"] = handoff["unique_post_id"].map(normalize_id)

    if role == "train":
        member1_path = data_dir / "member1_oof_predictions.csv"
        member2_path = data_dir / "member2_oof_predictions_deterministic.csv"
        expected_role = "train"
    elif role == "validation":
        member1_path = data_dir / "member1_validation_predictions_for_fusion.csv"
        member2_path = data_dir / "member2_validation_predictions_fold_ensemble.csv"
        expected_role = "validation"
    elif role == "test":
        member1_path = data_dir / "member1_test_predictions_for_fusion.csv"
        member2_path = data_dir / "member2_test_predictions_fold_ensemble.csv"
        expected_role = "test"
    else:
        raise ValueError(f"Unsupported role: {role}")

    member1 = pd.read_csv(member1_path, dtype={"unique_post_id": "string"})
    member2 = pd.read_csv(member2_path, dtype={"unique_post_id": "string"})
    member1["join_id"] = member1["unique_post_id"].map(normalize_id)
    member2["join_id"] = member2["unique_post_id"].map(normalize_id)

    expected = manifest.loc[
        manifest["model_role"].eq(expected_role)
    ].copy()
    expected_ids = set(expected["join_id"])

    for name, frame in [("Member 1", member1), ("Member 2", member2)]:
        if frame["join_id"].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate IDs.")
        actual_ids = set(frame["join_id"])
        missing = expected_ids - actual_ids
        unexpected = actual_ids - expected_ids
        if missing or unexpected:
            raise ValueError(
                f"{name} ID mismatch for {role}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    member1 = member1[["join_id", "prediction"]].rename(
        columns={"prediction": "member1_prediction"}
    )
    member2_keep = ["join_id", "prediction"]
    if "prediction_std_fold_models" in member2.columns:
        member2_keep.append("prediction_std_fold_models")
    member2 = member2[member2_keep].rename(
        columns={
            "prediction": "member2_prediction",
            "prediction_std_fold_models": "member2_fold_std",
        }
    )

    role_handoff = handoff.loc[
        handoff["model_role"].eq(expected_role)
    ].copy()

    merged = (
        expected.merge(role_handoff, on="join_id", suffixes=("_manifest", "_data"))
        .merge(member1, on="join_id", validate="one_to_one")
        .merge(member2, on="join_id", validate="one_to_one")
    )

    merged["unique_post_id"] = merged["unique_post_id_manifest"].astype(str)
    merged["final_stress"] = pd.to_numeric(
        merged["final_stress_manifest"], errors="raise"
    )
    merged["clinical_class"] = merged["clinical_class_manifest"].astype(str)

    if role == "train":
        merged["oof_fold"] = pd.to_numeric(
            merged["oof_fold_manifest"], errors="raise"
        ).astype(int)
        merged["training_sample_weight"] = pd.to_numeric(
            merged["training_sample_weight_manifest"], errors="raise"
        )
        merged["label_source"] = merged["label_source_manifest"].astype(str)

    return merged


def file_hash_manifest(project_root: str | Path, paths: list[Path]) -> dict:
    root = Path(project_root)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in paths
    }
