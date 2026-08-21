
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_bundle_and_roles_exist():
    bundle = json.loads(
        (ROOT / "models/fusion_bundle.json").read_text(encoding="utf-8")
    )
    assert bundle["thresholds"][0] < bundle["thresholds"][1] < bundle["thresholds"][2]
    manifest = pd.read_csv(ROOT / "data/modeling_manifest_v2.csv")
    assert (manifest["model_role"] == "train").sum() == 4226
    assert (manifest["model_role"] == "validation").sum() == 452
    assert (manifest["model_role"] == "test").sum() == 453
    assert (manifest["model_role"] == "embargo").sum() == 484
