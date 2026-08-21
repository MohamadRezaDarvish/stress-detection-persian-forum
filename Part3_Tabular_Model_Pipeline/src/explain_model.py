from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import Pool
from .common import save_json


def explain_models(models: list, prepared: dict, root: Path, project_config: dict, run_shap: bool=True) -> dict:
    out_dir = root / project_config["output_paths"]["explainability"]
    fig_dir = root / project_config["output_paths"]["figures"]
    features = prepared["X"]["train"].columns.tolist()
    importances = np.vstack([m.get_feature_importance(type="FeatureImportance") for m in models])
    global_imp = pd.DataFrame({"feature":features,"mean_importance":importances.mean(0),"std_importance":importances.std(0)})
    global_imp = global_imp.sort_values("mean_importance",ascending=False).reset_index(drop=True)
    global_imp.to_csv(out_dir/"catboost_global_feature_importance.csv",index=False)
    top = global_imp.head(25).sort_values("mean_importance")
    plt.figure(figsize=(9,8)); plt.barh(top["feature"],top["mean_importance"]); plt.xlabel('Mean CatBoost importance'); plt.title('Global feature importance across five folds'); plt.tight_layout(); plt.savefig(fig_dir/"global_feature_importance.png",dpi=170); plt.close()

    payload = {"method":"CatBoost feature importance", "feature_count":len(features), "used_for_formal_feature_selection":False, "shap_run":False}
    if run_shap:
        Xv = prepared["X"]["validation"]
        raw = prepared["raw"]["validation"]
        n = min(project_config.get("shap_sample_size",200),len(Xv))
        sample = Xv.iloc[:n].copy()
        pool = Pool(sample, cat_features=prepared["engineer"].categorical_features_)
        shap_arrays = []
        for model in models:
            values = model.get_feature_importance(pool, type="ShapValues")
            shap_arrays.append(values[:,:-1])
        shap_mean = np.mean(np.stack(shap_arrays,axis=0),axis=0)
        mean_abs = np.mean(np.abs(shap_mean),axis=0)
        shap_global = pd.DataFrame({"feature":features,"mean_abs_shap":mean_abs}).sort_values("mean_abs_shap",ascending=False)
        shap_global.to_csv(out_dir/"native_shap_global_ranking.csv",index=False)
        top_shap = shap_global.head(25).sort_values("mean_abs_shap")
        plt.figure(figsize=(9,8)); plt.barh(top_shap["feature"],top_shap["mean_abs_shap"]); plt.xlabel('Mean absolute SHAP'); plt.title('Native CatBoost SHAP ranking'); plt.tight_layout(); plt.savefig(fig_dir/"native_shap_global_ranking.png",dpi=170); plt.close()

        local_rows=[]
        for row_idx in range(min(5,n)):
            order=np.argsort(np.abs(shap_mean[row_idx]))[::-1][:10]
            for rank,feat_idx in enumerate(order,1):
                local_rows.append({"unique_post_id":str(raw.iloc[row_idx]["unique_post_id"]),"rank":rank,"feature":features[feat_idx],"feature_value":sample.iloc[row_idx,feat_idx],"shap_value":float(shap_mean[row_idx,feat_idx])})
        pd.DataFrame(local_rows).to_csv(out_dir/"native_shap_local_examples.csv",index=False)
        payload.update({"shap_run":True,"shap_method":"native CatBoost ShapValues averaged across five fold models","shap_rows":n})
    save_json(payload,out_dir/"explainability_manifest.json")
    return payload
