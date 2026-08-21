from __future__ import annotations
import pandas as pd
from .prediction_utils import ensemble_predict


def generate_validation_predictions(models: list, prepared: dict, project_config: dict):
    raw = prepared["raw"]["validation"]
    mean, std, matrix = ensemble_predict(models, prepared["X"]["validation"], project_config["prediction_range"])
    output = pd.DataFrame({
        "unique_post_id": raw[project_config["id_column"]].astype(str),
        "model_role": "validation",
        "true_stress": raw[project_config["target_column"]].astype(float),
        "prediction": mean,
        "prediction_std_optional": std,
    })
    fold_output = pd.DataFrame({"unique_post_id": output["unique_post_id"]})
    for fold in range(matrix.shape[0]): fold_output[f"fold_{fold}_prediction"] = matrix[fold]
    fold_output["ensemble_prediction"] = mean
    fold_output["prediction_std"] = std
    return output, fold_output
