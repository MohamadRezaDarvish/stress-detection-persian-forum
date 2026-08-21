#!/usr/bin/env python3
"""Phase 3: build an immutable author/thread/content-grouped split manifest.

This script must be run only after Phase 2 has added author and thread_id.
It builds connected components across authors, threads, and exact duplicate text,
then assigns whole components to train/validation/test.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLASSES = ("Low", "Moderate", "High", "Very High")
DEFAULT_EVAL_SOURCES = {
    "original_dual",
    "active_dual",
}


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_id(value: Any) -> str:
    text = normalize_text(value)
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def meaningful(value: Any) -> bool:
    text = normalize_text(value).lower()
    return bool(text) and text not in {"-1", "nan", "none", "null", "unknown", "نامشخص"}


def open_input(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


@dataclass
class GroupStats:
    group_id: str
    row_indices: list[int]
    total_count: int
    eval_class_counts: Counter
    source_counts: Counter
    train_only_count: int


def squared_relative_error(actual: float, target: float) -> float:
    denominator = max(target, 1.0)
    return ((actual - target) / denominator) ** 2


def assignment_score(
    split_totals: dict[str, int],
    split_classes: dict[str, Counter],
    split_sources: dict[str, Counter],
    split_train_only: dict[str, int],
    target_totals: dict[str, float],
    target_classes: dict[str, dict[str, float]],
    target_sources: dict[str, dict[str, float]],
) -> float:
    score = 0.0
    for split in target_totals:
        score += 0.35 * squared_relative_error(split_totals[split], target_totals[split])
        for cls in CLASSES:
            score += 2.5 * squared_relative_error(split_classes[split][cls], target_classes[split][cls])
        for source, target in target_sources[split].items():
            score += 0.25 * squared_relative_error(split_sources[split][source], target)
        if split != "train":
            score += 0.50 * split_train_only[split] / max(target_totals[split], 1.0)
    return score


def assign_groups(groups: list[GroupStats], ratios: dict[str, float], seed: int, restarts: int) -> tuple[dict[str, str], float]:
    splits = list(ratios)
    total_rows = sum(group.total_count for group in groups)
    total_classes = Counter()
    total_sources = Counter()
    for group in groups:
        total_classes.update(group.eval_class_counts)
        total_sources.update(group.source_counts)

    target_totals = {split: ratios[split] * total_rows for split in splits}
    target_classes = {split: {cls: ratios[split] * total_classes[cls] for cls in CLASSES} for split in splits}
    target_sources = {split: {source: ratios[split] * count for source, count in total_sources.items()} for split in splits}

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    base_rng = random.Random(seed)

    for restart in range(restarts):
        rng = random.Random(base_rng.randint(0, 2**31 - 1))
        # Rarity first, then size. Small random jitter changes tie order across restarts.
        ordered = sorted(
            groups,
            key=lambda group: (
                -sum(group.eval_class_counts[cls] / max(total_classes[cls], 1) for cls in CLASSES),
                -group.total_count,
                rng.random(),
            ),
        )
        split_totals = {split: 0 for split in splits}
        split_classes = {split: Counter() for split in splits}
        split_sources = {split: Counter() for split in splits}
        split_train_only = {split: 0 for split in splits}
        assignment: dict[str, str] = {}

        for group in ordered:
            candidate_scores = []
            for split in splits:
                split_totals[split] += group.total_count
                split_classes[split].update(group.eval_class_counts)
                split_sources[split].update(group.source_counts)
                split_train_only[split] += group.train_only_count
                score = assignment_score(
                    split_totals, split_classes, split_sources, split_train_only,
                    target_totals, target_classes, target_sources,
                )
                # Undo trial.
                split_totals[split] -= group.total_count
                split_classes[split].subtract(group.eval_class_counts)
                split_sources[split].subtract(group.source_counts)
                split_train_only[split] -= group.train_only_count
                candidate_scores.append((score, rng.random(), split))
            _, _, chosen = min(candidate_scores)
            assignment[group.group_id] = chosen
            split_totals[chosen] += group.total_count
            split_classes[chosen].update(group.eval_class_counts)
            split_sources[chosen].update(group.source_counts)
            split_train_only[chosen] += group.train_only_count

        score = assignment_score(
            split_totals, split_classes, split_sources, split_train_only,
            target_totals, target_classes, target_sources,
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment

    assert best_assignment is not None
    return best_assignment, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("split_manifest.csv"))
    parser.add_argument("--report", type=Path, default=Path("phase3_split_report.json"))
    parser.add_argument("--author-column", default="author")
    parser.add_argument("--thread-column", default="thread_id")
    parser.add_argument("--content-hash-column", default="label__content_hash")
    parser.add_argument("--id-column", default="unique_post_id")
    parser.add_argument("--stress-column", default="label__final_stress")
    parser.add_argument("--class-column", default="label__clinical_class")
    parser.add_argument("--source-column", default="label__label_source")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--restarts", type=int, default=300)
    parser.add_argument("--min-eval-per-class", type=int, default=25)
    parser.add_argument("--include-p1-only-in-evaluation", action="store_true")
    parser.add_argument("--include-calibrated-single-rater-in-evaluation", action="store_true")
    args = parser.parse_args()

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")

    with open_input(args.enriched) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    required = {args.id_column, args.author_column, args.thread_column, args.stress_column, args.class_column, args.source_column}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Enriched file is missing required columns: {sorted(missing)}")

    ids = [normalize_id(row[args.id_column]) for row in rows]
    duplicate_ids = [key for key, count in Counter(ids).items() if key and count > 1]
    if duplicate_ids:
        raise ValueError(f"Enriched canonical data must be unique by ID; duplicates: {duplicate_ids[:10]}")

    eval_sources = set(DEFAULT_EVAL_SOURCES)
    if args.include_p1_only_in_evaluation:
        eval_sources.add("original_p1_high_selected")
    if args.include_calibrated_single_rater_in_evaluation:
        eval_sources.add("active_p2_calibrated")

    # Count exact duplicate content hashes first; only duplicated hashes need to become graph nodes.
    content_hash_counts = Counter(row.get(args.content_hash_column, "") for row in rows if row.get(args.content_hash_column, ""))

    uf = UnionFind()
    for idx, row in enumerate(rows):
        post_node = f"P:{idx}"
        uf.add(post_node)
        author = normalize_text(row.get(args.author_column, ""))
        thread = normalize_id(row.get(args.thread_column, ""))
        content_hash = row.get(args.content_hash_column, "")
        linked = False
        if meaningful(author):
            uf.union(post_node, f"A:{author}")
            linked = True
        if meaningful(thread):
            uf.union(post_node, f"T:{thread}")
            linked = True
        if content_hash and content_hash_counts[content_hash] > 1:
            uf.union(post_node, f"C:{content_hash}")
            linked = True
        if not linked:
            uf.union(post_node, f"I:{ids[idx]}")

    component_rows: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(rows)):
        component_rows[uf.find(f"P:{idx}")].append(idx)

    groups: list[GroupStats] = []
    for ordinal, (_, indices) in enumerate(sorted(component_rows.items(), key=lambda item: min(item[1]))):
        group_id = f"G{ordinal:05d}"
        class_counts = Counter()
        source_counts = Counter()
        train_only_count = 0
        for idx in indices:
            source = rows[idx][args.source_column]
            source_counts[source] += 1
            if source in eval_sources:
                class_counts[rows[idx][args.class_column]] += 1
            else:
                train_only_count += 1
        groups.append(GroupStats(group_id, indices, len(indices), class_counts, source_counts, train_only_count))

    assignment, objective = assign_groups(groups, ratios, args.seed, args.restarts)
    row_group = {}
    for group in groups:
        for idx in group.row_indices:
            row_group[idx] = group.group_id

    dataset_digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: normalize_id(value[args.id_column])):
        dataset_digest.update(normalize_id(row[args.id_column]).encode("utf-8"))
        dataset_digest.update(b"\x1f")
        dataset_digest.update(str(row[args.stress_column]).encode("utf-8"))
        dataset_digest.update(b"\n")
    dataset_hash = dataset_digest.hexdigest()

    manifest_rows = []
    split_counts = Counter()
    eval_class_counts: dict[str, Counter] = {split: Counter() for split in ratios}
    source_counts: dict[str, Counter] = {split: Counter() for split in ratios}
    for idx, row in enumerate(rows):
        group_id = row_group[idx]
        split = assignment[group_id]
        source = row[args.source_column]
        evaluation_eligible = source in eval_sources
        training_eligible = split == "train"
        manifest = {
            "unique_post_id": normalize_id(row[args.id_column]),
            "split": split,
            "group_id": group_id,
            "author": row.get(args.author_column, ""),
            "thread_id": row.get(args.thread_column, ""),
            "content_hash": row.get(args.content_hash_column, ""),
            "final_stress": row[args.stress_column],
            "clinical_class": row[args.class_column],
            "label_source": source,
            "evaluation_eligible": evaluation_eligible,
            "training_eligible": training_eligible,
            "split_seed": args.seed,
            "dataset_hash": dataset_hash,
            "manifest_version": "1.0",
        }
        manifest_rows.append(manifest)
        split_counts[split] += 1
        source_counts[split][source] += 1
        if evaluation_eligible:
            eval_class_counts[split][row[args.class_column]] += 1

    manifest_fields = list(manifest_rows[0]) if manifest_rows else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    minimum_failures = []
    for split in ("val", "test"):
        for cls in CLASSES:
            count = eval_class_counts[split][cls]
            if count < args.min_eval_per_class:
                minimum_failures.append({"split": split, "class": cls, "count": count, "minimum": args.min_eval_per_class})

    group_sizes = sorted((group.total_count for group in groups), reverse=True)
    report = {
        "status": "complete" if not minimum_failures else "complete_with_class_count_warning",
        "dataset_hash": dataset_hash,
        "seed": args.seed,
        "ratios": ratios,
        "rows": len(rows),
        "connected_components": len(groups),
        "largest_component_sizes": group_sizes[:20],
        "assignment_objective": objective,
        "split_counts": dict(split_counts),
        "evaluation_sources": sorted(eval_sources),
        "evaluation_class_counts": {split: dict(counts) for split, counts in eval_class_counts.items()},
        "label_source_counts": {split: dict(counts) for split, counts in source_counts.items()},
        "minimum_eval_per_class": args.min_eval_per_class,
        "minimum_failures": minimum_failures,
        "leakage_guards": ["author connected components", "thread connected components", "exact duplicate content connected components"],
        "evaluation_scope": "safety-enriched labeled data; not a natural-prevalence website estimate",
    }
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
