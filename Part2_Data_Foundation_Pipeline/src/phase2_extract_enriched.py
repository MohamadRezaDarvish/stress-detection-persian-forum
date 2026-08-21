#!/usr/bin/env python3
"""Phase 2: safely enrich canonical labels from the 360k raw CSV.

Key properties:
- streams the 176 MB file once; does not load it fully into memory;
- never performs a many-to-many dataframe merge;
- handles duplicate unique_post_id values using normalized content;
- writes a detailed match audit and can fail in strict mode.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def text_hash(value: Any) -> str:
    text = normalize_text(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8-sig", newline="")
    return path.open(mode, encoding="utf-8-sig", newline="")


def read_labels(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open_text(path, "r") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    ids = [normalize_id(row.get("unique_post_id")) for row in rows]
    duplicates = [key for key, count in Counter(ids).items() if key and count > 1]
    if duplicates:
        raise ValueError(f"Canonical labels are not unique by unique_post_id: {duplicates[:10]}")
    for row, post_id in zip(rows, ids):
        row["unique_post_id"] = post_id
    return rows, fields


def row_fingerprint(row: dict[str, str], ignored: set[str]) -> str:
    payload = "\x1f".join(f"{key}={row.get(key, '')}" for key in sorted(row) if key not in ignored)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_candidate(label: dict[str, str], candidates: list[dict[str, str]], content_column: str) -> tuple[dict[str, str] | None, dict[str, Any]]:
    label_content = normalize_text(label.get("content", ""))
    audit: dict[str, Any] = {
        "unique_post_id": label["unique_post_id"],
        "source_duplicate_count": len(candidates),
        "match_status": "",
        "matched_by": "",
        "content_similarity": "",
        "chosen_source_row_number": "",
        "candidate_source_row_numbers": "|".join(str(row["__source_row_number"]) for row in candidates),
    }
    if not candidates:
        audit["match_status"] = "missing_id"
        return None, audit
    if len(candidates) == 1:
        chosen = candidates[0]
        similarity = 1.0 if label_content and label_content == normalize_text(chosen.get(content_column, "")) else ""
        audit.update({
            "match_status": "unique_id_match",
            "matched_by": "unique_post_id",
            "content_similarity": similarity,
            "chosen_source_row_number": chosen["__source_row_number"],
        })
        return chosen, audit

    # Multiple source rows share the ID. Prefer exact normalized content.
    exact = [row for row in candidates if label_content and normalize_text(row.get(content_column, "")) == label_content]
    if len(exact) == 1:
        chosen = exact[0]
        audit.update({
            "match_status": "duplicate_id_resolved_exact_content",
            "matched_by": "unique_post_id+exact_normalized_content",
            "content_similarity": 1.0,
            "chosen_source_row_number": chosen["__source_row_number"],
        })
        return chosen, audit
    if len(exact) > 1:
        fingerprints = {row_fingerprint(row, {"__source_row_number"}) for row in exact}
        if len(fingerprints) == 1:
            chosen = min(exact, key=lambda row: int(row["__source_row_number"]))
            audit.update({
                "match_status": "duplicate_id_multiple_identical_exact_rows",
                "matched_by": "unique_post_id+exact_content+identical_metadata",
                "content_similarity": 1.0,
                "chosen_source_row_number": chosen["__source_row_number"],
            })
            return chosen, audit
        audit["match_status"] = "ambiguous_duplicate_id_multiple_exact_content_rows"
        return None, audit

    # If canonical content is unavailable (the recovered trusted-P1-only rows),
    # resolve only when all duplicate source rows contain the same normalized text.
    normalized_candidate_texts = {normalize_text(row.get(content_column, "")) for row in candidates}
    if not label_content:
        if len(normalized_candidate_texts) == 1:
            chosen = min(candidates, key=lambda row: int(row["__source_row_number"]))
            audit.update({
                "match_status": "duplicate_id_resolved_identical_source_content",
                "matched_by": "unique_post_id+identical_source_content",
                "chosen_source_row_number": chosen["__source_row_number"],
            })
            return chosen, audit
        audit["match_status"] = "ambiguous_duplicate_id_no_label_content"
        return None, audit

    # Last resort: a uniquely strong fuzzy content match. This never happens silently.
    scored = []
    for row in candidates:
        source_content = normalize_text(row.get(content_column, ""))
        ratio = difflib.SequenceMatcher(None, label_content, source_content, autojunk=False).ratio()
        scored.append((ratio, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_ratio, best_row = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    if best_ratio >= 0.97 and best_ratio - second_ratio >= 0.02:
        audit.update({
            "match_status": "duplicate_id_resolved_unique_fuzzy_content",
            "matched_by": "unique_post_id+fuzzy_content",
            "content_similarity": best_ratio,
            "chosen_source_row_number": best_row["__source_row_number"],
        })
        return best_row, audit

    audit.update({
        "match_status": "ambiguous_duplicate_id_content_mismatch",
        "content_similarity": best_ratio,
    })
    return None, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="cleaned_full_ninisite_stress_proxy3.csv")
    parser.add_argument("--labels", type=Path, required=True, help="canonical_labels.csv.gz")
    parser.add_argument("--output", type=Path, default=Path("active_and_original_labels_enriched.csv.gz"))
    parser.add_argument("--audit", type=Path, default=Path("phase2_match_audit.csv"))
    parser.add_argument("--report", type=Path, default=Path("phase2_report.json"))
    parser.add_argument("--id-column", default="unique_post_id")
    parser.add_argument("--content-column", default="content")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any label is missing or ambiguous")
    args = parser.parse_args()

    labels, label_fields = read_labels(args.labels)
    label_by_id = {row["unique_post_id"]: row for row in labels}
    target_ids = set(label_by_id)
    candidates_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)

    with args.raw.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_fields = list(reader.fieldnames or [])
        if args.id_column not in raw_fields:
            raise ValueError(f"Raw file lacks ID column {args.id_column!r}")
        if args.content_column not in raw_fields:
            raise ValueError(f"Raw file lacks content column {args.content_column!r}")
        raw_rows_scanned = 0
        for source_row_number, row in enumerate(reader, start=2):
            raw_rows_scanned += 1
            post_id = normalize_id(row.get(args.id_column))
            if post_id in target_ids:
                row[args.id_column] = post_id
                row["__source_row_number"] = str(source_row_number)
                candidates_by_id[post_id].append(row)

    enriched_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    unresolved = []
    for label in labels:
        post_id = label["unique_post_id"]
        chosen, audit = resolve_candidate(label, candidates_by_id.get(post_id, []), args.content_column)
        audit_rows.append(audit)
        if chosen is None:
            unresolved.append(post_id)
            continue
        output_row = {key: value for key, value in chosen.items() if key != "__source_row_number"}
        for key, value in label.items():
            output_row[f"label__{key}"] = value
        output_row["match_status"] = audit["match_status"]
        output_row["source_duplicate_count"] = audit["source_duplicate_count"]
        output_row["matched_source_row_number"] = audit["chosen_source_row_number"]
        output_row["raw_content_hash"] = text_hash(chosen.get(args.content_column, ""))
        enriched_rows.append(output_row)

    output_fields = raw_fields + [
        f"label__{field}" for field in label_fields
    ] + ["match_status", "source_duplicate_count", "matched_source_row_number", "raw_content_hash"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(enriched_rows)

    audit_fields = [
        "unique_post_id", "source_duplicate_count", "match_status", "matched_by",
        "content_similarity", "chosen_source_row_number", "candidate_source_row_numbers",
    ]
    with args.audit.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit_rows)

    status_counts = Counter(row["match_status"] for row in audit_rows)
    duplicate_id_targets = sum(1 for rows in candidates_by_id.values() if len(rows) > 1)
    report = {
        "raw_rows_scanned": raw_rows_scanned,
        "canonical_labels_requested": len(labels),
        "enriched_rows_written": len(enriched_rows),
        "unresolved_rows": len(unresolved),
        "unresolved_ids": unresolved,
        "target_ids_with_duplicate_source_rows": duplicate_id_targets,
        "match_status_counts": dict(status_counts),
        "strict_mode": bool(args.strict),
        "output": str(args.output.resolve()),
        "audit": str(args.audit.resolve()),
    }
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict and unresolved:
        print("Strict mode failed: unresolved labels are listed in the audit/report.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
