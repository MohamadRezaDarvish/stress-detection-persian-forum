from __future__ import annotations
from pathlib import Path
import pandas as pd
from .common import save_json


def verify_artifacts(root: Path, project_config: dict, prepared: dict) -> dict:
    checks={}
    for fold in range(project_config["number_of_folds"]):
        p=root/"models"/f"fold_{fold}"/"model.cbm"
        checks[f"fold_{fold}_model_exists"]=p.exists() and p.stat().st_size>0
    required=[
        root/"data/handoff/member1_oof_predictions.csv",
        root/"data/handoff/member1_validation_predictions.csv",
        root/"data/handoff/member1_test_predictions.csv",
        root/"outputs/metrics/member1_metrics.json",
        root/"outputs/figures/global_feature_importance.png",
        root/"outputs/explainability/catboost_global_feature_importance.csv",
        root/"models/model_manifest.json",
    ]
    for p in required: checks[f"exists::{p.relative_to(root)}"]=p.exists() and p.stat().st_size>0
    oof=pd.read_csv(root/"data/handoff/member1_oof_predictions.csv")
    val=pd.read_csv(root/"data/handoff/member1_validation_predictions.csv")
    test=pd.read_csv(root/"data/handoff/member1_test_predictions.csv")
    train_ids=set(prepared["raw"]["train"][project_config["id_column"]].astype(str)); val_ids=set(prepared["raw"]["validation"][project_config["id_column"]].astype(str)); test_ids=set(prepared["raw"]["test"][project_config["id_column"]].astype(str))
    checks.update({
        "oof_row_count":len(oof)==len(train_ids),"validation_row_count":len(val)==len(val_ids),"test_row_count":len(test)==len(test_ids),
        "oof_id_set":set(oof.unique_post_id.astype(str))==train_ids,"validation_id_set":set(val.unique_post_id.astype(str))==val_ids,"test_id_set":set(test.unique_post_id.astype(str))==test_ids,
        "oof_unique_ids":not oof.unique_post_id.astype(str).duplicated().any(),"validation_unique_ids":not val.unique_post_id.astype(str).duplicated().any(),"test_unique_ids":not test.unique_post_id.astype(str).duplicated().any(),
        "oof_no_missing_prediction":not oof.prediction.isna().any(),"validation_no_missing_prediction":not val.prediction.isna().any(),"test_no_missing_prediction":not test.prediction.isna().any(),
        "prediction_range_ok":bool(oof.prediction.between(*project_config["prediction_range"]).all() and val.prediction.between(*project_config["prediction_range"]).all() and test.prediction.between(*project_config["prediction_range"]).all()),
        "no_role_id_overlap":not bool((train_ids&val_ids)|(train_ids&test_ids)|(val_ids&test_ids)),
        "validation_has_ensemble_std":bool(val.prediction_std_optional.notna().all()),"test_has_ensemble_std":bool(test.prediction_std_optional.notna().all()),
    })
    passed=all(bool(v) for v in checks.values())
    payload={"passed":passed,"checks":checks}
    save_json(payload,root/"outputs/metrics/artifact_verification.json")
    if not passed:
        failed=[k for k,v in checks.items() if not bool(v)]; raise RuntimeError("Artifact verification failed: "+", ".join(failed))
    return payload
