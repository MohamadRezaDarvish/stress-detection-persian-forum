from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from .common import save_json


def _params(model_config: dict, seed: int) -> dict:
    keys = ["loss_function","eval_metric","iterations","depth","learning_rate","l2_leaf_reg","random_strength","bagging_temperature","allow_writing_files","verbose","thread_count","task_type"]
    params = {k:model_config[k] for k in keys}
    params["random_seed"] = int(seed)
    return params


def train_or_load_fold_models(prepared: dict, root: Path, project_config: dict, model_config: dict, full_retrain: bool=True) -> list:
    X = prepared["X"]["train"]
    raw = prepared["raw"]["train"]
    y = raw[project_config["target_column"]].astype(float)
    weights = raw[project_config["weight_column"]].astype(float)
    folds = raw[project_config["fold_column"]].astype(int)
    cat_features = prepared["engineer"].categorical_features_
    models = []
    model_manifest = {"models":[], "full_retrain": bool(full_retrain), "model_config": model_config}

    for fold in range(project_config["number_of_folds"]):
        fold_dir = root / "models" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / "model.cbm"
        config_path = fold_dir / "fold_config.json"
        history_path = fold_dir / "training_history.json"
        fit_mask = folds.ne(fold)
        held_mask = folds.eq(fold)
        seed = project_config["random_seed"] + fold
        if full_retrain:
            model = CatBoostRegressor(**_params(model_config, seed))
            model.fit(
                X.loc[fit_mask], y.loc[fit_mask],
                cat_features=cat_features,
                sample_weight=weights.loc[fit_mask],
                eval_set=(X.loc[held_mask], y.loc[held_mask]),
                use_best_model=False,
                verbose=False,
            )
            model.save_model(str(model_path), format="cbm")
            history = model.get_evals_result()
            save_json(history, history_path)
            save_json({
                "fold": fold, "seed": seed, "train_rows": int(fit_mask.sum()), "heldout_rows": int(held_mask.sum()),
                "heldout_ids": raw.loc[held_mask, project_config["id_column"]].astype(str).tolist(),
                "hyperparameters": _params(model_config, seed), "sample_weight_used": True,
                "heldout_used_for_early_stopping": False,
            }, config_path)
        else:
            if not model_path.exists():
                raise FileNotFoundError(f"Missing fold model: {model_path}")
            model = CatBoostRegressor()
            model.load_model(str(model_path))
        models.append(model)
        model_manifest["models"].append({"fold":fold,"path":str(model_path.relative_to(root)).replace('\\','/'),"seed":seed})

    # Optional full-train model, not used for OOF/validation/test ensemble.
    final_dir = root / "models" / "final_full_train"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "model.cbm"
    if full_retrain:
        final_model = CatBoostRegressor(**_params(model_config, project_config["random_seed"]))
        final_model.fit(X, y, cat_features=cat_features, sample_weight=weights, verbose=False)
        final_model.save_model(str(final_path), format="cbm")
    elif not final_path.exists():
        raise FileNotFoundError(f"Missing full-train model: {final_path}")
    model_manifest["final_full_train_model"] = str(final_path.relative_to(root)).replace('\\','/')
    model_manifest["note"] = "Validation/test and future-post default inference use the five fold-model ensemble."
    save_json(model_manifest, root / "models" / "model_manifest.json")
    return models
