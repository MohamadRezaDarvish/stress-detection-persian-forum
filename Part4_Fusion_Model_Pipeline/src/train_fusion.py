
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    CLASS_ORDER,
    SCORE_CLIP,
    evaluate_predictions,
    file_hash_manifest,
    json_safe_metrics,
    load_project_role,
    sha256_file,
    write_metrics_tables,
)
from member1_feature_engineering import (  # noqa: E402
    CATEGORICAL_FEATURES,
    Member1FeatureEngineer,
)
from thresholds import search_constrained_thresholds  # noqa: E402


def prepare_features(frame: pd.DataFrame):
    engineer = Member1FeatureEngineer()
    tabular = engineer.transform(frame)
    tabular["member1_prediction"] = pd.to_numeric(
        frame["member1_prediction"], errors="raise"
    ).to_numpy()
    tabular["member2_prediction"] = pd.to_numeric(
        frame["member2_prediction"], errors="raise"
    ).to_numpy()
    tabular["base_mean"] = (
        tabular["member1_prediction"] + tabular["member2_prediction"]
    ) / 2.0
    tabular["base_difference"] = (
        tabular["member2_prediction"] - tabular["member1_prediction"]
    )
    tabular["base_absolute_difference"] = np.abs(
        tabular["base_difference"]
    )
    hybrid_features = list(engineer.feature_order_) + [
        "member1_prediction",
        "member2_prediction",
        "base_mean",
        "base_difference",
        "base_absolute_difference",
    ]
    categorical = [
        hybrid_features.index(column)
        for column in CATEGORICAL_FEATURES
    ]
    return engineer, tabular, hybrid_features, categorical


def candidate_metrics(true_stress, prediction):
    prediction = np.clip(prediction, *SCORE_CLIP)
    return {
        "mae": float(mean_absolute_error(true_stress, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(true_stress, prediction))),
        "pearson": float(pearsonr(true_stress, prediction).statistic),
    }


def fit_catboost(
    name,
    train_features,
    train_target,
    validation_features,
    sample_weight,
    categorical_features,
    config,
):
    model = CatBoostRegressor(
        random_seed=20260804,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
        **config,
    )
    model.fit(
        train_features,
        train_target,
        sample_weight=sample_weight,
        cat_features=categorical_features,
    )
    validation_prediction = np.clip(
        model.predict(validation_features), *SCORE_CLIP
    )
    return model, validation_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(THIS_DIR.parent))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    outputs = root / "outputs"
    models_dir = root / "models"
    outputs.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train = load_project_role(root, "train")
    validation = load_project_role(root, "validation")

    engineer, train_hybrid, hybrid_features, categorical_indices = prepare_features(train)
    _, validation_hybrid, _, _ = prepare_features(validation)

    scalar_features = ["member1_prediction", "member2_prediction"]
    train_target = train["final_stress"].to_numpy(float)
    validation_target = validation["final_stress"].to_numpy(float)
    validation_class = validation["clinical_class"].astype(str).to_numpy()
    sample_weight = train["training_sample_weight"].to_numpy(float)

    candidates = {}
    trained_models = {}
    candidate_metadata = {}

    def register(name, prediction, model=None, feature_profile="scalar", config=None):
        prediction = np.clip(np.asarray(prediction, dtype=float), *SCORE_CLIP)
        continuous = candidate_metrics(validation_target, prediction)
        threshold_result = search_constrained_thresholds(
            validation_class,
            prediction,
            validation_mae=continuous["mae"],
            top_n=100,
        )
        candidates[name] = {
            "prediction": prediction,
            "continuous": continuous,
            "threshold_result": threshold_result,
        }
        if model is not None:
            trained_models[name] = model
        candidate_metadata[name] = {
            "feature_profile": feature_profile,
            "config": config or {},
        }

    register(
        "member1_only",
        validation["member1_prediction"].to_numpy(float),
    )
    register(
        "member2_only",
        validation["member2_prediction"].to_numpy(float),
    )
    for member2_weight in np.arange(0.50, 1.001, 0.05):
        prediction = (
            (1.0 - member2_weight)
            * validation["member1_prediction"].to_numpy(float)
            + member2_weight
            * validation["member2_prediction"].to_numpy(float)
        )
        register(
            f"weighted_blend_member2_{member2_weight:.2f}",
            prediction,
            config={"member2_weight": float(member2_weight)},
        )

    scalar_train = train[scalar_features]
    scalar_validation = validation[scalar_features]

    for alpha in [0.0, 0.1, 1.0, 10.0]:
        if alpha == 0.0:
            model = LinearRegression()
            name = "linear_scalar"
        else:
            model = Ridge(alpha=alpha)
            name = f"ridge_scalar_alpha_{alpha:g}"
        model.fit(
            scalar_train,
            train_target,
            sample_weight=sample_weight,
        )
        register(
            name,
            model.predict(scalar_validation),
            model=model,
            feature_profile="scalar",
            config={"alpha": alpha},
        )

    scalar_catboost_configs = {
        "scalar_catboost_depth2": {
            "iterations": 300,
            "depth": 2,
            "learning_rate": 0.04,
            "loss_function": "RMSE",
            "l2_leaf_reg": 7,
            "monotone_constraints": [1, 1],
        },
        "scalar_catboost_depth3": {
            "iterations": 350,
            "depth": 3,
            "learning_rate": 0.035,
            "loss_function": "RMSE",
            "l2_leaf_reg": 8,
            "monotone_constraints": [1, 1],
        },
    }
    for name, config in scalar_catboost_configs.items():
        model, prediction = fit_catboost(
            name,
            scalar_train,
            train_target,
            scalar_validation,
            sample_weight,
            [],
            config,
        )
        register(
            name,
            prediction,
            model=model,
            feature_profile="scalar",
            config=config,
        )

    hybrid_configs = {
        "hybrid_catboost_rmse_depth3": {
            "iterations": 450,
            "depth": 3,
            "learning_rate": 0.035,
            "loss_function": "RMSE",
            "l2_leaf_reg": 10,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
        },
        "hybrid_catboost_rmse_depth4": {
            "iterations": 450,
            "depth": 4,
            "learning_rate": 0.035,
            "loss_function": "RMSE",
            "l2_leaf_reg": 10,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
        },
        "hybrid_catboost_rmse_depth5": {
            "iterations": 400,
            "depth": 5,
            "learning_rate": 0.03,
            "loss_function": "RMSE",
            "l2_leaf_reg": 12,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
        },
        "hybrid_catboost_mae_depth4": {
            "iterations": 450,
            "depth": 4,
            "learning_rate": 0.035,
            "loss_function": "MAE",
            "l2_leaf_reg": 10,
            "random_strength": 0.5,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
        },
    }

    for name, config in hybrid_configs.items():
        model, prediction = fit_catboost(
            name,
            train_hybrid[hybrid_features],
            train_target,
            validation_hybrid[hybrid_features],
            sample_weight,
            categorical_indices,
            config,
        )
        register(
            name,
            prediction,
            model=model,
            feature_profile="hybrid",
            config=config,
        )

    rows = []
    feasible_names = []
    for name, result in candidates.items():
        threshold_result = result["threshold_result"]
        best = threshold_result["best"]
        row = {
            "candidate": name,
            "feature_profile": candidate_metadata[name]["feature_profile"],
            "validation_mae": result["continuous"]["mae"],
            "validation_rmse": result["continuous"]["rmse"],
            "validation_pearson": result["continuous"]["pearson"],
            "threshold_status": threshold_result["status"],
            "feasible_threshold_count": threshold_result["feasible_count"],
        }
        if best is not None:
            feasible_names.append(name)
            row.update(
                {
                    "threshold_low_moderate": best["thresholds"][0],
                    "threshold_moderate_high": best["thresholds"][1],
                    "threshold_high_very_high": best["thresholds"][2],
                    "validation_accuracy": best["accuracy"],
                    "validation_macro_f1": best["macro_f1"],
                    "validation_low_recall": best["recall_by_class"]["Low"],
                    "validation_moderate_recall": best["recall_by_class"]["Moderate"],
                    "validation_high_recall": best["recall_by_class"]["High"],
                    "validation_very_high_recall": best["recall_by_class"]["Very High"],
                    "validation_low_precision": best["precision_by_class"]["Low"],
                    "validation_moderate_precision": best["precision_by_class"]["Moderate"],
                    "validation_high_precision": best["precision_by_class"]["High"],
                    "validation_very_high_precision": best["precision_by_class"]["Very High"],
                }
            )
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    if not feasible_names:
        raise RuntimeError(
            "No candidate satisfies all validation recall constraints."
        )

    def candidate_rank(name):
        result = candidates[name]
        best = result["threshold_result"]["best"]
        return (
            best["precision_by_class"]["Very High"],
            best["macro_f1"],
            best["precision_by_class"]["High"],
            -result["continuous"]["mae"],
        )

    selected_name = max(feasible_names, key=candidate_rank)
    selected = candidates[selected_name]
    selected_model = trained_models.get(selected_name)
    selected_profile = candidate_metadata[selected_name]["feature_profile"]
    selected_thresholds = selected["threshold_result"]["best"]["thresholds"]

    # The project requires a deployable learned model. If a non-model candidate
    # won, fall back to the best feasible trained model under the same ranking.
    if selected_model is None:
        trained_feasible = [
            name for name in feasible_names if name in trained_models
        ]
        if not trained_feasible:
            raise RuntimeError("No feasible trained fusion model.")
        selected_name = max(trained_feasible, key=candidate_rank)
        selected = candidates[selected_name]
        selected_model = trained_models[selected_name]
        selected_profile = candidate_metadata[selected_name]["feature_profile"]
        selected_thresholds = selected["threshold_result"]["best"]["thresholds"]

    leaderboard["selected"] = leaderboard["candidate"].eq(selected_name)
    leaderboard.sort_values(
        [
            "selected",
            "validation_very_high_precision",
            "validation_macro_f1",
            "validation_high_precision",
            "validation_mae",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
        inplace=True,
    )
    leaderboard.to_csv(
        outputs / "candidate_leaderboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if selected_profile == "hybrid":
        selected_features = hybrid_features
        selected_categorical = CATEGORICAL_FEATURES
        selected_validation_matrix = validation_hybrid[selected_features]
        selected_train_matrix = train_hybrid[selected_features]
    else:
        selected_features = scalar_features
        selected_categorical = []
        selected_validation_matrix = scalar_validation
        selected_train_matrix = scalar_train

    selected_prediction = np.clip(
        selected_model.predict(selected_validation_matrix),
        *SCORE_CLIP,
    )
    validation_metrics = evaluate_predictions(
        validation_target,
        validation_class,
        selected_prediction,
        selected_thresholds,
    )

    validation_output = pd.DataFrame(
        {
            "unique_post_id": validation["unique_post_id"].astype(str),
            "true_stress": validation_target,
            "true_class": validation_class,
            "member1_prediction": validation["member1_prediction"],
            "member2_prediction": validation["member2_prediction"],
            "fusion_prediction": selected_prediction,
            "predicted_class": validation_metrics["predicted_class"],
        }
    )
    validation_output.to_csv(
        outputs / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (outputs / "validation_metrics.json").write_text(
        json.dumps(
            json_safe_metrics(validation_metrics),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_metrics_tables(
        validation_metrics,
        outputs / "validation",
    )

    selected_model.save_model(
        models_dir / "fusion_selected.cbm"
    )

    # Save scalar fallback independently.
    fallback_model = trained_models["scalar_catboost_depth2"]
    fallback_model.save_model(
        models_dir / "fusion_scalar_fallback.cbm"
    )
    fallback_best = candidates["scalar_catboost_depth2"]["threshold_result"]["best"]

    selected_threshold_rows = []
    for rank, record in enumerate(
        selected["threshold_result"]["top"], start=1
    ):
        selected_threshold_rows.append(
            {
                "rank": rank,
                "threshold_low_moderate": record["thresholds"][0],
                "threshold_moderate_high": record["thresholds"][1],
                "threshold_high_very_high": record["thresholds"][2],
                "accuracy": record["accuracy"],
                "macro_f1": record["macro_f1"],
                "low_precision": record["precision_by_class"]["Low"],
                "moderate_precision": record["precision_by_class"]["Moderate"],
                "high_precision": record["precision_by_class"]["High"],
                "very_high_precision": record["precision_by_class"]["Very High"],
                "low_recall": record["recall_by_class"]["Low"],
                "moderate_recall": record["recall_by_class"]["Moderate"],
                "high_recall": record["recall_by_class"]["High"],
                "very_high_recall": record["recall_by_class"]["Very High"],
            }
        )
    pd.DataFrame(selected_threshold_rows).to_csv(
        outputs / "selected_threshold_search_top100.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_importance = selected_model.get_feature_importance()
    pd.DataFrame(
        {
            "feature": selected_features,
            "importance": feature_importance,
        }
    ).sort_values("importance", ascending=False).to_csv(
        outputs / "global_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shap_pool = Pool(
        selected_validation_matrix,
        cat_features=[
            selected_features.index(column)
            for column in selected_categorical
        ],
    )
    shap_values = selected_model.get_feature_importance(
        shap_pool,
        type="ShapValues",
    )
    shap_feature_values = shap_values[:, :-1]
    pd.DataFrame(
        {
            "feature": selected_features,
            "mean_absolute_shap": np.mean(
                np.abs(shap_feature_values), axis=0
            ),
            "mean_signed_shap": np.mean(
                shap_feature_values, axis=0
            ),
        }
    ).sort_values("mean_absolute_shap", ascending=False).to_csv(
        outputs / "validation_shap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    input_hashes = file_hash_manifest(
        root,
        [
            root / "data/modeling_manifest_v2.csv",
            root / "data/member1_handoff.csv.gz",
            root / "data/member1_oof_predictions.csv",
            root / "data/member1_validation_predictions_for_fusion.csv",
            root / "data/member2_oof_predictions_deterministic.csv",
            root / "data/member2_validation_predictions_fold_ensemble.csv",
        ],
    )

    bundle = {
        "project_version": "1.0.0",
        "selected_candidate": selected_name,
        "feature_profile": selected_profile,
        "features": selected_features,
        "categorical_features": selected_categorical,
        "thresholds": selected_thresholds,
        "score_clip": list(SCORE_CLIP),
        "candidate_config": candidate_metadata[selected_name]["config"],
        "validation_metrics": json_safe_metrics(validation_metrics),
        "fallback": {
            "model": "fusion_scalar_fallback.cbm",
            "features": scalar_features,
            "thresholds": fallback_best["thresholds"],
        },
        "input_hashes": input_hashes,
        "test_evaluated": False,
    }
    (models_dir / "fusion_bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    training_status = {
        "status": "complete",
        "selected_candidate": selected_name,
        "feature_profile": selected_profile,
        "validation_constraints_pass": validation_metrics["classification"][
            "constraints_pass"
        ],
        "validation_failed_constraints": validation_metrics[
            "classification"
        ]["failed_recall_constraints"],
        "test_loaded_or_evaluated": False,
        "model_sha256": sha256_file(models_dir / "fusion_selected.cbm"),
        "bundle_sha256": sha256_file(models_dir / "fusion_bundle.json"),
    }
    (outputs / "training_status.json").write_text(
        json.dumps(training_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(training_status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
