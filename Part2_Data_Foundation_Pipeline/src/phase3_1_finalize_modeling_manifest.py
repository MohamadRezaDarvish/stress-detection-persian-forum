
#!/usr/bin/env python3
"""Finalize the Phase 3 split for model development.

This script:
- converts the initial component split into train/validation/test/embargo model roles;
- moves val/test components with no official evaluation rows into training;
- embargoes non-official rows sharing a component with official evaluation rows;
- creates five leakage-safe grouped OOF folds for training;
- combines annotation-confidence and moderate class-balancing weights;
- creates Member 1 and Member 2 handoff datasets.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def normalize_id(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(
        str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})
    )
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--phase3-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    report = json.loads(args.phase3_report.read_text(encoding="utf-8"))
    manifest = pd.read_csv(
        args.manifest,
        dtype={"unique_post_id": "string"},
    )
    enriched = pd.read_csv(
        args.enriched,
        compression="infer",
        dtype={"unique_post_id": "string"},
    )

    manifest["join_id"] = manifest["unique_post_id"].map(normalize_id)
    enriched["join_id"] = enriched["unique_post_id"].map(normalize_id)

    if not manifest["join_id"].is_unique:
        raise ValueError("Split manifest IDs are not unique after normalization.")
    if not enriched["join_id"].is_unique:
        raise ValueError("Enriched IDs are not unique after normalization.")
    if set(manifest["join_id"]) != set(enriched["join_id"]):
        missing_from_data = set(manifest["join_id"]) - set(enriched["join_id"])
        missing_from_manifest = set(enriched["join_id"]) - set(manifest["join_id"])
        raise ValueError(
            "Enriched/manifest ID sets differ: "
            f"missing_from_data={len(missing_from_data)}, "
            f"missing_from_manifest={len(missing_from_manifest)}"
        )

    # The initial Phase 3 optimizer balanced whole components. Components in val/test
    # with no official evaluation rows can safely move to training.
    group_eval_count = manifest.groupby("group_id")["evaluation_eligible"].sum()
    group_original_split = manifest.groupby("group_id")["split"].first()
    move_to_train_groups = set(
        group_eval_count[
            (group_eval_count == 0)
            & group_original_split.isin(["val", "test"])
        ].index
    )

    def assign_role(row: pd.Series) -> str:
        if row["split"] == "train" or row["group_id"] in move_to_train_groups:
            return "train"
        if bool(row["evaluation_eligible"]) and row["split"] == "val":
            return "validation"
        if bool(row["evaluation_eligible"]) and row["split"] == "test":
            return "test"
        return "embargo"

    manifest["model_role"] = manifest.apply(assign_role, axis=1)
    manifest["official_eval"] = manifest["model_role"].isin(["validation", "test"])
    manifest["moved_from_original_split"] = (
        manifest["group_id"].isin(move_to_train_groups)
        & manifest["split"].isin(["val", "test"])
    )

    # Five grouped OOF folds for fusion/base-model training.
    manifest["oof_fold"] = pd.Series(pd.NA, index=manifest.index, dtype="Int64")
    train_part = manifest.loc[manifest["model_role"].eq("train")].copy()
    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=int(report["seed"]),
    )
    for fold, (_, heldout_positions) in enumerate(
        splitter.split(
            np.zeros((len(train_part), 1)),
            train_part["clinical_class"],
            groups=train_part["group_id"],
        )
    ):
        heldout_index = train_part.iloc[heldout_positions].index
        manifest.loc[heldout_index, "oof_fold"] = fold

    if manifest.loc[manifest["model_role"].eq("train"), "oof_fold"].isna().any():
        raise RuntimeError("Some training rows did not receive an OOF fold.")
    if (
        manifest.loc[manifest["model_role"].eq("train")]
        .groupby("group_id")["oof_fold"]
        .nunique()
        .max()
        != 1
    ):
        raise RuntimeError("A connected component crosses OOF folds.")

    merged = manifest.merge(
        enriched,
        on="join_id",
        how="inner",
        suffixes=("_manifest", "_data"),
        validate="one_to_one",
    )

    merged["unique_post_id_clean"] = merged["unique_post_id_data"].astype(str)
    merged["final_stress_clean"] = pd.to_numeric(merged["final_stress"], errors="raise")
    merged["clinical_class_clean"] = merged["clinical_class"].astype(str)
    merged["annotation_weight"] = pd.to_numeric(
        merged["label__default_label_weight"],
        errors="coerce",
    ).fillna(1.0)

    # Moderate square-root inverse-frequency class weighting.
    training_counts = (
        merged.loc[merged["model_role"].eq("train"), "clinical_class_clean"]
        .value_counts()
        .to_dict()
    )
    n_train = sum(training_counts.values())
    class_weight_map = {
        label: float(np.sqrt(n_train / (len(training_counts) * count)))
        for label, count in training_counts.items()
    }
    merged["class_weight"] = merged["clinical_class_clean"].map(class_weight_map)
    raw_weight = merged["annotation_weight"] * merged["class_weight"]
    normalization = raw_weight[merged["model_role"].eq("train")].mean()
    merged["training_sample_weight"] = np.where(
        merged["model_role"].eq("train"),
        raw_weight / normalization,
        np.nan,
    )

    modeling_manifest = pd.DataFrame(
        {
            "unique_post_id": merged["unique_post_id_clean"],
            "group_id": merged["group_id"],
            "author": merged["author_manifest"],
            "thread_id": merged["thread_id_manifest"],
            "content_hash": merged["content_hash"],
            "final_stress": merged["final_stress_clean"],
            "clinical_class": merged["clinical_class_clean"],
            "label_source": merged["label_source"],
            "label_round": merged["label__label_round"],
            "label_confidence_tier": merged["label__confidence_tier"],
            "annotation_weight": merged["annotation_weight"],
            "class_weight": merged["class_weight"],
            "training_sample_weight": merged["training_sample_weight"],
            "original_component_split": merged["split"],
            "model_role": merged["model_role"],
            "official_eval": merged["official_eval"],
            "moved_from_original_split": merged["moved_from_original_split"],
            "oof_fold": merged["oof_fold"],
            "split_seed": merged["split_seed"],
            "dataset_hash": merged["dataset_hash"],
            "manifest_version": 2,
        }
    )
    modeling_manifest["use_for_training"] = modeling_manifest["model_role"].eq("train")
    modeling_manifest["use_for_model_selection"] = modeling_manifest["model_role"].eq("validation")
    modeling_manifest["use_for_final_test"] = modeling_manifest["model_role"].eq("test")
    modeling_manifest["exclude_from_modeling"] = modeling_manifest["model_role"].eq("embargo")

    control = pd.DataFrame(
        {
            "unique_post_id": merged["unique_post_id_clean"],
            "final_stress": merged["final_stress_clean"],
            "clinical_class": merged["clinical_class_clean"],
            "model_role": merged["model_role"],
            "official_eval": merged["official_eval"],
            "oof_fold": merged["oof_fold"],
            "group_id": merged["group_id"],
            "label_source": merged["label_source"],
            "label_confidence_tier": merged["label__confidence_tier"],
            "annotation_weight": merged["annotation_weight"],
            "class_weight": merged["class_weight"],
            "training_sample_weight": merged["training_sample_weight"],
        }
    )

    member1_source_columns = [
        "thread_id_data", "thread_title", "author_data", "posted_at", "content",
        "likes", "reply_to", "is_starter", "user_post_count", "gender", "gender_code",
        "join_date", "children_count", "category", "sub_category", "age_num",
        "education_clean", "sig_char_count", "sig_punct_count", "sig_question_count",
        "sig_excl_count", "sig_emoji_count", "sig_word_count", "sig_neg_count",
        "sig_pos_count", "sig_pos_emoji", "sig_neg_emoji", "post_char_count",
        "post_punct_count", "post_question_count", "post_excl_count", "post_emoji_count",
        "post_pos_emoji", "post_neg_emoji", "post_word_count", "post_neg_count",
        "post_pos_count", "stress_proxy", "post_neg_count_temp", "post_pos_count_temp",
    ]
    member1_raw = merged[member1_source_columns].rename(
        columns={"thread_id_data": "thread_id", "author_data": "author"}
    )
    member1_handoff = pd.concat(
        [control.reset_index(drop=True), member1_raw.reset_index(drop=True)],
        axis=1,
    )
    member2_handoff = pd.concat(
        [control.reset_index(drop=True), merged[["content"]].reset_index(drop=True)],
        axis=1,
    )

    role_order = {"train": 0, "validation": 1, "test": 2, "embargo": 3}
    for frame in [modeling_manifest, member1_handoff, member2_handoff]:
        frame["_role_order"] = frame["model_role"].map(role_order)
        frame.sort_values(
            ["_role_order", "oof_fold", "unique_post_id"],
            na_position="last",
            inplace=True,
        )
        frame.drop(columns="_role_order", inplace=True)

    modeling_manifest.to_csv(
        output / "modeling_manifest_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    member1_handoff.to_csv(
        output / "member1_handoff.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    member2_handoff.to_csv(
        output / "member2_handoff.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    for role in ["train", "validation", "test", "embargo"]:
        modeling_manifest.loc[modeling_manifest["model_role"].eq(role)].to_csv(
            output / f"{role}_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )

    fold_distribution = (
        modeling_manifest.loc[modeling_manifest["model_role"].eq("train")]
        .groupby(["oof_fold", "clinical_class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    fold_distribution.to_csv(
        output / "oof_fold_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    role_distribution = (
        modeling_manifest.groupby(["model_role", "clinical_class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    role_distribution.to_csv(
        output / "model_role_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    minimum_recall = {"Low": 0.75, "Moderate": 0.50, "High": 0.50, "Very High": 0.75}
    acceptance_rows = []
    for role in ["validation", "test"]:
        role_counts = (
            modeling_manifest.loc[modeling_manifest["model_role"].eq(role), "clinical_class"]
            .value_counts()
        )
        for label in ["Low", "Moderate", "High", "Very High"]:
            count = int(role_counts.get(label, 0))
            minimum_correct = int(np.ceil(count * minimum_recall[label]))
            acceptance_rows.append(
                {
                    "evaluation_split": role,
                    "clinical_class": label,
                    "available_rows": count,
                    "minimum_recall": minimum_recall[label],
                    "minimum_correct_predictions": minimum_correct,
                    "maximum_allowed_misses": count - minimum_correct,
                }
            )
    pd.DataFrame(acceptance_rows).to_csv(
        output / "acceptance_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cleanup_report = {
        "status": "complete",
        "source_dataset_hash": report["dataset_hash"],
        "seed": report["seed"],
        "rows_total": int(len(modeling_manifest)),
        "roles": {
            key: int(value)
            for key, value in modeling_manifest["model_role"].value_counts().items()
        },
        "rows_moved_from_non_eval_val_test_components_to_train": int(
            modeling_manifest["moved_from_original_split"].sum()
        ),
        "embargo_rows": int(modeling_manifest["model_role"].eq("embargo").sum()),
        "oof_folds": args.folds,
        "training_class_counts": {key: int(value) for key, value in training_counts.items()},
        "class_weight_method": "square-root inverse frequency",
        "class_weight_map": class_weight_map,
        "training_sample_weight_mean": float(
            modeling_manifest.loc[
                modeling_manifest["model_role"].eq("train"),
                "training_sample_weight",
            ].mean()
        ),
    }
    (output / "cleanup_report.json").write_text(
        json.dumps(cleanup_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(cleanup_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
