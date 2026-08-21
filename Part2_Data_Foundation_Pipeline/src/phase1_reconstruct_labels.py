#!/usr/bin/env python3
"""Phase 1: reconstruct a canonical label table with provenance.

The script is deliberately CSV-only so it can run without Excel libraries.
The two original annotation workbooks have already been exported to CSV.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold

P1_WEIGHT = 1.7
P2_WEIGHT = 1.0
CLASS_THRESHOLDS = (3.0, 5.0, 7.0)
DEFAULT_WEIGHTS = {
    "original_dual": 1.00,
    "original_p1_high_selected": 0.90,
    "active_dual": 1.00,
    "active_p2_calibrated": 0.65,
}


def normalize_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_hash(value: Any) -> str:
    normalized = normalize_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def stress_class(score: float) -> str:
    if score < CLASS_THRESHOLDS[0]:
        return "Low"
    if score < CLASS_THRESHOLDS[1]:
        return "Moderate"
    if score < CLASS_THRESHOLDS[2]:
        return "High"
    return "Very High"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_gz(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_plain(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assert_unique(rows: list[dict[str, str]], name: str) -> None:
    counts = Counter(normalize_id(row["unique_post_id"]) for row in rows)
    duplicates = {key: count for key, count in counts.items() if key and count > 1}
    if duplicates:
        preview = list(duplicates.items())[:10]
        raise ValueError(f"{name} contains duplicate unique_post_id values: {preview}")


def evaluate_calibrators(p1: np.ndarray, p2: np.ndarray, consensus: np.ndarray) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = {
        "identity": np.zeros_like(consensus, dtype=float),
        "mean_shift": np.zeros_like(consensus, dtype=float),
        "linear_to_consensus": np.zeros_like(consensus, dtype=float),
        "linear_p2_to_p1_then_weighted": np.zeros_like(consensus, dtype=float),
        "isotonic_to_consensus": np.zeros_like(consensus, dtype=float),
        "ordinal_lookup": np.zeros_like(consensus, dtype=float),
    }

    for train_idx, test_idx in splitter.split(p2.reshape(-1, 1), p2.astype(int)):
        x_train = p2[train_idx]
        x_test = p2[test_idx]
        y_train = consensus[train_idx]
        p1_train = p1[train_idx]

        predictions["identity"][test_idx] = x_test
        predictions["mean_shift"][test_idx] = x_test + (y_train.mean() - x_train.mean())

        linear_consensus = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
        predictions["linear_to_consensus"][test_idx] = linear_consensus.predict(x_test.reshape(-1, 1))

        linear_p1 = LinearRegression().fit(x_train.reshape(-1, 1), p1_train)
        estimated_p1 = linear_p1.predict(x_test.reshape(-1, 1))
        predictions["linear_p2_to_p1_then_weighted"][test_idx] = (
            P1_WEIGHT * estimated_p1 + P2_WEIGHT * x_test
        ) / (P1_WEIGHT + P2_WEIGHT)

        isotonic = IsotonicRegression(out_of_bounds="clip").fit(x_train, y_train)
        predictions["isotonic_to_consensus"][test_idx] = isotonic.predict(x_test)

        lookup = {value: float(y_train[x_train == value].mean()) for value in np.unique(x_train)}
        predictions["ordinal_lookup"][test_idx] = np.array([lookup[float(value)] for value in x_test])

    metrics: dict[str, Any] = {}
    for name, pred in predictions.items():
        metrics[name] = {
            "mae_to_consensus": float(mean_absolute_error(consensus, pred)),
            "rmse_to_consensus": float(mean_squared_error(consensus, pred) ** 0.5),
            "mae_to_person1": float(mean_absolute_error(p1, pred)),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("inputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_p1 = read_csv(input_dir / "original_person1_2000.csv")
    original_p2 = read_csv(input_dir / "original_person2_2000.csv")
    active_p1 = read_csv(input_dir / "active_person1_1000.csv")
    active_p2 = read_csv(input_dir / "active_person2_3000.csv")
    active_historical = read_csv(input_dir / "active_historical_final_3000.csv")
    fusion_index = read_csv(input_dir / "historical_fusion_index.csv")

    for name, rows in [
        ("original_person1_2000", original_p1),
        ("original_person2_2000", original_p2),
        ("active_person1_1000", active_p1),
        ("active_person2_3000", active_p2),
        ("active_historical_final_3000", active_historical),
        ("historical_fusion_index", fusion_index),
    ]:
        assert_unique(rows, name)

    original_p1_by_id = {normalize_id(row["unique_post_id"]): row for row in original_p1}
    original_p2_by_id = {normalize_id(row["unique_post_id"]): row for row in original_p2}
    active_p1_by_id = {normalize_id(row["unique_post_id"]): row for row in active_p1}
    active_p2_by_id = {normalize_id(row["unique_post_id"]): row for row in active_p2}
    active_final_by_id = {normalize_id(row["unique_post_id"]): row for row in active_historical}
    fusion_by_id = {normalize_id(row["unique_post_id"]): row for row in fusion_index}

    if set(original_p1_by_id) != set(original_p2_by_id):
        raise ValueError("Original Person 1 and Person 2 ID sets differ")
    for post_id in original_p1_by_id:
        if normalize_text(original_p1_by_id[post_id]["content"]) != normalize_text(original_p2_by_id[post_id]["content"]):
            raise ValueError(f"Original annotator content mismatch for {post_id}")

    if list(active_p1_by_id) != [normalize_id(row["unique_post_id"]) for row in active_p2[: len(active_p1)]]:
        raise ValueError("Active Person 1 rows are not aligned to the first active Person 2 rows")
    if set(active_p2_by_id) != set(active_final_by_id):
        raise ValueError("Active Person 2 and historical final label ID sets differ")

    # Historical active-learning calibration: predict Person 1 from Person 2 on the 1,000 overlap,
    # then apply the trusted 1.7:1 weighted consensus.
    overlap_ids = [normalize_id(row["unique_post_id"]) for row in active_p1]
    p1_values = np.array([float(active_p1_by_id[post_id]["stress"]) for post_id in overlap_ids])
    p2_values = np.array([float(active_p2_by_id[post_id]["stress"]) for post_id in overlap_ids])
    overlap_consensus = (P1_WEIGHT * p1_values + P2_WEIGHT * p2_values) / (P1_WEIGHT + P2_WEIGHT)

    p2_to_p1 = LinearRegression().fit(p2_values.reshape(-1, 1), p1_values)
    p2_to_p1_slope = float(p2_to_p1.coef_[0])
    p2_to_p1_intercept = float(p2_to_p1.intercept_)
    consensus_slope = float((P1_WEIGHT * p2_to_p1_slope + P2_WEIGHT) / (P1_WEIGHT + P2_WEIGHT))
    consensus_intercept = float(P1_WEIGHT * p2_to_p1_intercept / (P1_WEIGHT + P2_WEIGHT))

    calibrator_cv = evaluate_calibrators(p1_values, p2_values, overlap_consensus)

    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    # Original 2,000 dual-labeled rows.
    for post_id, row_p1 in original_p1_by_id.items():
        row_p2 = original_p2_by_id[post_id]
        p1_raw = float(row_p1["A2_stress"])
        p2_raw = float(row_p2["A3_stress"])
        final = (P1_WEIGHT * p1_raw + P2_WEIGHT * p2_raw) / (P1_WEIGHT + P2_WEIGHT)
        fusion_row = fusion_by_id.get(post_id)
        candidates.append({
            "unique_post_id": post_id,
            "content": row_p1["content"],
            "content_hash": content_hash(row_p1["content"]),
            "final_stress": final,
            "clinical_class": stress_class(final),
            "label_source": "original_dual",
            "label_round": "original_round",
            "person1_raw": p1_raw,
            "person2_raw": p2_raw,
            "person1_estimated": "",
            "historical_active_final": "",
            "historical_split": fusion_row["historical_split"] if fusion_row else "",
            "historical_included": bool(fusion_row),
            "selection_method": "original_selection_then_moderate_undersampling",
            "confidence_tier": "high",
            "default_label_weight": DEFAULT_WEIGHTS["original_dual"],
            "canonical_preference_rank": 1,
            "notes": "",
        })
        history.extend([
            {"unique_post_id": post_id, "label_round": "original_round", "annotator": "person1", "stress": p1_raw, "observed": True, "content_hash": content_hash(row_p1["content"])},
            {"unique_post_id": post_id, "label_round": "original_round", "annotator": "person2", "stress": p2_raw, "observed": True, "content_hash": content_hash(row_p1["content"])},
        ])

    # Recover the selected trusted-Person-1-only rows from the historical fusion index.
    original_ids = set(original_p1_by_id)
    for post_id, fusion_row in fusion_by_id.items():
        if post_id in original_ids:
            continue
        p1_raw = float(fusion_row["stress"])
        if not math.isclose(p1_raw, round(p1_raw), abs_tol=1e-9):
            raise ValueError(f"Recovered Person 1-only label is unexpectedly non-integer: {post_id}={p1_raw}")
        candidates.append({
            "unique_post_id": post_id,
            "content": "",
            "content_hash": "",
            "final_stress": p1_raw,
            "clinical_class": stress_class(p1_raw),
            "label_source": "original_p1_high_selected",
            "label_round": "original_extra_person1_round",
            "person1_raw": p1_raw,
            "person2_raw": "",
            "person1_estimated": "",
            "historical_active_final": "",
            "historical_split": fusion_row["historical_split"],
            "historical_included": True,
            "selection_method": "selected_after_person1_high_score; exact cutoff unavailable",
            "confidence_tier": "medium_high",
            "default_label_weight": DEFAULT_WEIGHTS["original_p1_high_selected"],
            "canonical_preference_rank": 2,
            "notes": "Raw extra-800 annotation file is unavailable; retained label recovered from fusion target.",
        })
        history.append({"unique_post_id": post_id, "label_round": "original_extra_person1_round", "annotator": "person1", "stress": p1_raw, "observed": True, "content_hash": ""})

    # Active-learning 3,000 rows.
    active_overlap_set = set(active_p1_by_id)
    for row_p2 in active_p2:
        post_id = normalize_id(row_p2["unique_post_id"])
        p2_raw = float(row_p2["stress"])
        content = row_p2["content"]
        if post_id in active_overlap_set:
            p1_raw = float(active_p1_by_id[post_id]["stress"])
            p1_estimated: float | str = ""
            final = (P1_WEIGHT * p1_raw + P2_WEIGHT * p2_raw) / (P1_WEIGHT + P2_WEIGHT)
            source = "active_dual"
            tier = "high"
        else:
            p1_raw = ""
            p1_estimated = p2_to_p1_slope * p2_raw + p2_to_p1_intercept
            final = (P1_WEIGHT * float(p1_estimated) + P2_WEIGHT * p2_raw) / (P1_WEIGHT + P2_WEIGHT)
            source = "active_p2_calibrated"
            tier = "medium"
        historical_final = float(active_final_by_id[post_id]["stress"])
        if not math.isclose(final, historical_final, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Could not reproduce active historical target for {post_id}: {final} vs {historical_final}")
        candidates.append({
            "unique_post_id": post_id,
            "content": content,
            "content_hash": content_hash(content),
            "final_stress": final,
            "clinical_class": stress_class(final),
            "label_source": source,
            "label_round": "active_learning_round",
            "person1_raw": p1_raw,
            "person2_raw": p2_raw,
            "person1_estimated": p1_estimated,
            "historical_active_final": historical_final,
            "historical_split": "",
            "historical_included": False,
            "selection_method": "active_learning_threshold_near_boundary; 750 prescreened per predicted class",
            "confidence_tier": tier,
            "default_label_weight": DEFAULT_WEIGHTS[source],
            "canonical_preference_rank": 3 if source == "active_dual" else 4,
            "notes": "",
        })
        history.append({"unique_post_id": post_id, "label_round": "active_learning_round", "annotator": "person2", "stress": p2_raw, "observed": True, "content_hash": content_hash(content)})
        if source == "active_dual":
            history.append({"unique_post_id": post_id, "label_round": "active_learning_round", "annotator": "person1", "stress": p1_raw, "observed": True, "content_hash": content_hash(content)})
        else:
            history.append({"unique_post_id": post_id, "label_round": "active_learning_round", "annotator": "person1_estimated", "stress": p1_estimated, "observed": False, "content_hash": content_hash(content)})

    # Resolve duplicate IDs across rounds. Prefer real dual annotations, then trusted P1-only,
    # then active dual, then calibrated single-rater labels. Keep all events in history.
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_id[row["unique_post_id"]].append(row)

    canonical: list[dict[str, Any]] = []
    duplicate_id_audit: list[dict[str, Any]] = []
    for post_id, rows in by_id.items():
        rows.sort(key=lambda row: int(row["canonical_preference_rank"]))
        selected = dict(rows[0])
        selected["duplicate_id_across_rounds"] = len(rows) > 1
        selected["annotation_versions"] = len(rows)
        selected["discarded_label_sources"] = "|".join(row["label_source"] for row in rows[1:])
        if len(rows) > 1:
            selected["notes"] = (selected["notes"] + " Canonical row chosen by provenance rank; repeat annotations retained in history.").strip()
            for row in rows:
                duplicate_id_audit.append({
                    "unique_post_id": post_id,
                    "selected_canonical": row is rows[0],
                    "label_source": row["label_source"],
                    "label_round": row["label_round"],
                    "final_stress": row["final_stress"],
                    "content_hash": row["content_hash"],
                })
        canonical.append(selected)

    # Exact duplicate text groups must remain together during future splitting.
    content_counts = Counter(row["content_hash"] for row in canonical if row["content_hash"])
    for row in canonical:
        row["duplicate_content_group_size"] = content_counts.get(row["content_hash"], 0) if row["content_hash"] else 0
        row["exact_duplicate_content"] = row["duplicate_content_group_size"] > 1

    canonical.sort(key=lambda row: (row["label_round"], row["unique_post_id"]))
    history.sort(key=lambda row: (row["unique_post_id"], row["label_round"], row["annotator"]))

    canonical_fields = [
        "unique_post_id", "content", "content_hash", "final_stress", "clinical_class",
        "label_source", "label_round", "person1_raw", "person2_raw", "person1_estimated",
        "historical_active_final", "historical_split", "historical_included", "selection_method",
        "confidence_tier", "default_label_weight", "duplicate_id_across_rounds", "annotation_versions",
        "discarded_label_sources", "duplicate_content_group_size", "exact_duplicate_content", "notes",
    ]
    history_fields = ["unique_post_id", "label_round", "annotator", "stress", "observed", "content_hash"]
    duplicate_fields = ["unique_post_id", "selected_canonical", "label_source", "label_round", "final_stress", "content_hash"]

    write_csv_gz(output_dir / "canonical_labels.csv.gz", canonical_fields, canonical)
    write_csv_gz(output_dir / "annotation_history.csv.gz", history_fields, history)
    write_csv_plain(output_dir / "duplicate_id_audit.csv", duplicate_fields, duplicate_id_audit)
    write_csv_gz(
        output_dir / "phase2_lookup_keys.csv.gz",
        ["unique_post_id", "content", "content_hash", "final_stress", "clinical_class", "label_source", "confidence_tier", "default_label_weight"],
        canonical,
    )

    source_counts = Counter(row["label_source"] for row in canonical)
    class_counts = Counter(row["clinical_class"] for row in canonical)
    source_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in canonical:
        source_class_counts[row["label_source"]][row["clinical_class"]] += 1

    original_overlap_active = sorted(set(original_p1_by_id) & set(active_p2_by_id))
    report = {
        "status": "phase1_complete",
        "class_thresholds": list(CLASS_THRESHOLDS),
        "person1_weight": P1_WEIGHT,
        "person2_weight": P2_WEIGHT,
        "counts": {
            "original_dual_rows": len(original_p1),
            "historical_fusion_rows": len(fusion_index),
            "original_dual_in_historical_fusion": len(set(original_p1_by_id) & set(fusion_by_id)),
            "original_dual_removed_by_historical_undersampling": len(set(original_p1_by_id) - set(fusion_by_id)),
            "recovered_original_p1_high_selected": len(set(fusion_by_id) - set(original_p1_by_id)),
            "active_rows": len(active_p2),
            "active_dual_rows": len(active_p1),
            "active_p2_calibrated_rows": len(active_p2) - len(active_p1),
            "duplicate_ids_across_labeling_rounds": len(original_overlap_active),
            "canonical_unique_posts": len(canonical),
            "annotation_history_rows": len(history),
        },
        "canonical_source_counts": dict(source_counts),
        "canonical_class_counts": dict(class_counts),
        "source_by_class_counts": {key: dict(value) for key, value in source_class_counts.items()},
        "active_calibration": {
            "method_reproducing_historical_labels": "OLS Person2->Person1 on 1000 overlaps, then weighted mean 1.7:1",
            "person2_to_person1_slope": p2_to_p1_slope,
            "person2_to_person1_intercept": p2_to_p1_intercept,
            "equivalent_person2_to_consensus_slope": consensus_slope,
            "equivalent_person2_to_consensus_intercept": consensus_intercept,
            "cv_metrics": calibrator_cv,
        },
        "duplicate_id_values": original_overlap_active,
        "exact_duplicate_content_groups": sum(1 for count in content_counts.values() if count > 1),
        "exact_duplicate_content_rows": sum(count for count in content_counts.values() if count > 1),
        "default_label_weights_are_provisional": True,
        "phase2_required": "Run phase2_extract_enriched.py locally against cleaned_full_ninisite_stress_proxy3.csv.",
        "phase3_blocker": "Final author/thread connected-component split requires Phase 2 enriched output.",
    }
    with (output_dir / "phase1_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print("Source counts:", dict(source_counts))
    print("Class counts:", dict(class_counts))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
