import csv
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_phase2_duplicate_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            labels = tmp / "labels.csv.gz"
            with gzip.open(labels, "wt", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["unique_post_id", "content", "final_stress", "clinical_class", "label_source"])
                w.writeheader()
                w.writerows([
                    {"unique_post_id": "1", "content": "متن اول", "final_stress": 2, "clinical_class": "Low", "label_source": "original_dual"},
                    {"unique_post_id": "2", "content": "متن درست", "final_stress": 6, "clinical_class": "High", "label_source": "active_dual"},
                    {"unique_post_id": "3", "content": "یکسان", "final_stress": 7, "clinical_class": "Very High", "label_source": "original_p1_high_selected"},
                ])
            raw = tmp / "raw.csv"
            with raw.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["unique_post_id", "content", "author", "thread_id"])
                w.writeheader()
                w.writerows([
                    {"unique_post_id": "1", "content": "متن اول", "author": "a", "thread_id": "t1"},
                    {"unique_post_id": "2", "content": "متن غلط", "author": "b", "thread_id": "t2"},
                    {"unique_post_id": "2", "content": "متن درست", "author": "b", "thread_id": "t2"},
                    {"unique_post_id": "3", "content": "یکسان", "author": "c", "thread_id": "t3"},
                    {"unique_post_id": "3", "content": "یکسان", "author": "c", "thread_id": "t3"},
                ])
            output = tmp / "enriched.csv.gz"
            audit = tmp / "audit.csv"
            report = tmp / "report.json"
            subprocess.run([
                sys.executable, str(ROOT / "src" / "phase2_extract_enriched.py"),
                "--raw", str(raw), "--labels", str(labels), "--output", str(output),
                "--audit", str(audit), "--report", str(report), "--strict",
            ], check=True)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["enriched_rows_written"], 3)
            self.assertEqual(result["unresolved_rows"], 0)
            self.assertEqual(result["target_ids_with_duplicate_source_rows"], 2)

    def test_phase3_connected_components_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            enriched = tmp / "enriched.csv.gz"
            fields = [
                "unique_post_id", "author", "thread_id", "label__content_hash",
                "label__final_stress", "label__clinical_class", "label__label_source",
            ]
            rows = []
            classes = ["Low", "Moderate", "High", "Very High"]
            # 80 independent components plus linked examples.
            for i in range(80):
                rows.append({
                    "unique_post_id": str(i), "author": f"a{i}", "thread_id": f"t{i}",
                    "label__content_hash": f"h{i}", "label__final_stress": i % 10 + 1,
                    "label__clinical_class": classes[i % 4], "label__label_source": "original_dual",
                })
            rows += [
                {"unique_post_id": "100", "author": "linked_author", "thread_id": "thread_x", "label__content_hash": "hx", "label__final_stress": 5, "label__clinical_class": "High", "label__label_source": "active_dual"},
                {"unique_post_id": "101", "author": "other_author", "thread_id": "thread_x", "label__content_hash": "hy", "label__final_stress": 7, "label__clinical_class": "Very High", "label__label_source": "active_dual"},
                {"unique_post_id": "102", "author": "linked_author", "thread_id": "thread_y", "label__content_hash": "hz", "label__final_stress": 4, "label__clinical_class": "Moderate", "label__label_source": "active_dual"},
            ]
            with gzip.open(enriched, "wt", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(rows)
            manifest = tmp / "manifest.csv"
            report = tmp / "report.json"
            subprocess.run([
                sys.executable, str(ROOT / "src" / "phase3_build_split_manifest.py"),
                "--enriched", str(enriched), "--output", str(manifest), "--report", str(report),
                "--restarts", "20", "--min-eval-per-class", "1",
            ], check=True)
            with manifest.open("r", encoding="utf-8-sig", newline="") as f:
                out = list(csv.DictReader(f))
            by_id = {row["unique_post_id"]: row for row in out}
            self.assertEqual(by_id["100"]["group_id"], by_id["101"]["group_id"])
            self.assertEqual(by_id["100"]["group_id"], by_id["102"]["group_id"])
            self.assertEqual(by_id["100"]["split"], by_id["101"]["split"])
            self.assertEqual(by_id["100"]["split"], by_id["102"]["split"])


if __name__ == "__main__":
    unittest.main()
