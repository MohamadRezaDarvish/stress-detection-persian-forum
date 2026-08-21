from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from .feature_engineering import Member1FeatureEngineer, FEATURE_ORDER, CATEGORICAL_FEATURES
from .common import save_json


def prepare_data(handoff: pd.DataFrame, root: Path, project_config: dict) -> dict:
    roles = {}
    for role in ["train", "validation", "test"]:
        roles[role] = handoff.loc[handoff["model_role"].eq(role)].copy().reset_index(drop=True)
    embargo = handoff.loc[handoff["model_role"].eq("embargo")].copy().reset_index(drop=True)

    engineer = Member1FeatureEngineer()
    X_train = engineer.fit_transform(roles["train"])
    X_validation = engineer.transform(roles["validation"])
    X_test = engineer.transform(roles["test"])

    processed_dir = root / project_config["output_paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(engineer, processed_dir / "member1_preprocessor.joblib")
    save_json(engineer.to_config_dict(), processed_dir / "member1_preprocessor_config.json")
    pd.DataFrame({"position": range(len(FEATURE_ORDER)), "feature": FEATURE_ORDER, "is_categorical": [f in CATEGORICAL_FEATURES for f in FEATURE_ORDER]}).to_csv(processed_dir / "member1_exact_feature_order.csv", index=False)

    manifest = {
        "feature_count": len(FEATURE_ORDER),
        "feature_order": FEATURE_ORDER,
        "categorical_features": CATEGORICAL_FEATURES,
        "row_counts": {role: int(len(df)) for role, df in roles.items()} | {"embargo": int(len(embargo))},
        "transformer_stateless": True,
        "fit_scope": "train rows only; transformer is stateless",
        "embargo_transformed": False,
    }
    save_json(manifest, processed_dir / "processed_data_manifest.json")
    return {
        "raw": roles,
        "embargo": embargo,
        "X": {"train": X_train, "validation": X_validation, "test": X_test},
        "engineer": engineer,
        "manifest": manifest,
    }
