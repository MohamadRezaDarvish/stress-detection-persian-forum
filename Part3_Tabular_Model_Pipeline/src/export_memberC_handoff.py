from __future__ import annotations
from pathlib import Path
import pandas as pd
from .common import save_json


def export_handoff(prepared: dict, oof_output: pd.DataFrame, validation_output: pd.DataFrame, test_output: pd.DataFrame, validation_fold_output: pd.DataFrame, test_fold_output: pd.DataFrame, root: Path, project_config: dict) -> dict:
    pred_dir = root / project_config["output_paths"]["predictions"]
    handoff_dir = root / project_config["output_paths"]["handoff"]
    pred_dir.mkdir(parents=True,exist_ok=True); handoff_dir.mkdir(parents=True,exist_ok=True)
    files={}
    for name,df in [("member1_oof_predictions.csv",oof_output),("member1_validation_predictions.csv",validation_output),("member1_test_predictions.csv",test_output)]:
        df.to_csv(pred_dir/name,index=False); df.to_csv(handoff_dir/name,index=False); files[name]=len(df)
    validation_fold_output.to_csv(pred_dir/"member1_validation_fold_predictions.csv",index=False)
    test_fold_output.to_csv(pred_dir/"member1_test_fold_predictions.csv",index=False)

    for role in ["train","validation","test"]:
        raw=prepared["raw"][role].reset_index(drop=True)
        X=prepared["X"][role].reset_index(drop=True)
        meta=pd.DataFrame({
            "unique_post_id":raw[project_config["id_column"]].astype(str),
            "model_role":role,
            "oof_fold":raw[project_config["fold_column"]] if role=="train" else pd.Series([pd.NA]*len(raw)),
            "final_stress":raw[project_config["target_column"]].astype(float),
            "clinical_class":raw[project_config["class_column"]].astype(str),
        })
        full=pd.concat([meta,X],axis=1)
        fn=f"member1_{role}_engineered_features.csv.gz"
        full.to_csv(handoff_dir/fn,index=False,compression="gzip"); files[fn]=len(full)

    schema={
        "prediction_files":{
            "train_oof":"member1_oof_predictions.csv",
            "validation_ensemble":"member1_validation_predictions.csv",
            "test_ensemble":"member1_test_predictions.csv"
        },
        "prediction_range":project_config["prediction_range"],
        "feature_order":prepared["engineer"].feature_order_,
        "categorical_features":prepared["engineer"].categorical_features_,
        "row_counts":files,
        "alignment_key":project_config["id_column"],
        "missing_value_policy":"numeric NaN is handled natively by CatBoost; categorical missing values use __MISSING__; signature -1 means no signature and is converted to 0 with has_signature indicator",
    }
    save_json(schema,handoff_dir/"member1_handoff_manifest.json")
    return schema
