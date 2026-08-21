from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, confusion_matrix, accuracy_score, f1_score, average_precision_score
from .common import save_json


def class_from_prediction(prediction, thresholds, class_order):
    return np.asarray(class_order, dtype=object)[np.digitize(np.asarray(prediction, dtype=float), thresholds, right=False)]


def continuous_metrics(y_true, prediction):
    y = np.asarray(y_true, dtype=float); p = np.asarray(prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(y,p)),
        "rmse": float(np.sqrt(mean_squared_error(y,p))),
        "pearson": float(pearsonr(y,p).statistic),
        "spearman": float(spearmanr(y,p).statistic),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
        "prediction_mean": float(p.mean()),
    }


def classification_metrics(y_class, prediction, thresholds, class_order):
    true = np.asarray(y_class, dtype=object)
    pred_class = class_from_prediction(prediction, thresholds, class_order)
    precision, recall, f1, support = precision_recall_fscore_support(true, pred_class, labels=class_order, zero_division=0)
    per_class = pd.DataFrame({"class":class_order,"precision":precision,"recall":recall,"f1":f1,"support":support})
    cm = pd.DataFrame(confusion_matrix(true,pred_class,labels=class_order), index=[f"true_{c}" for c in class_order], columns=[f"pred_{c}" for c in class_order])
    very_high_true = (true == "Very High").astype(int)
    return {
        "accuracy": float(accuracy_score(true,pred_class)),
        "macro_f1": float(f1_score(true,pred_class,labels=class_order,average="macro",zero_division=0)),
        "very_high_pr_auc": float(average_precision_score(very_high_true, np.asarray(prediction,dtype=float))),
        "per_class": per_class.set_index("class").to_dict(orient="index"),
    }, per_class, cm, pred_class


def evaluate_all(prepared: dict, oof_output: pd.DataFrame, validation_output: pd.DataFrame, test_output: pd.DataFrame, root: Path, project_config: dict) -> dict:
    thresholds = project_config["class_prediction_thresholds"]
    classes = project_config["class_order"]
    roles = prepared["raw"]
    inputs = {
        "oof": (roles["train"], oof_output),
        "validation": (roles["validation"], validation_output),
        "test": (roles["test"], test_output),
    }
    metrics_dir = root / project_config["output_paths"]["metrics"]
    fig_dir = root / project_config["output_paths"]["figures"]
    metrics = {"thresholds":thresholds,"threshold_source":"frozen validation-optimized thresholds; test labels not used","continuous":{},"classification":{}}
    for split,(raw,preds) in inputs.items():
        y = raw[project_config["target_column"]].astype(float).to_numpy()
        p = preds["prediction"].astype(float).to_numpy()
        metrics["continuous"][split] = continuous_metrics(y,p)
        c_payload, per_class, cm, pred_class = classification_metrics(raw[project_config["class_column"]], p, thresholds, classes)
        metrics["classification"][split] = c_payload
        per_class.to_csv(metrics_dir/f"{split}_per_class_metrics.csv",index=False)
        cm.to_csv(metrics_dir/f"{split}_confusion_matrix.csv")
        scored = preds.copy(); scored["predicted_class"] = pred_class
        scored.to_csv(root/project_config["output_paths"]["predictions"]/f"{split}_predictions_with_class.csv",index=False)

        plt.figure(figsize=(6,5)); plt.scatter(y,p,alpha=.45); plt.plot([1,10],[1,10],'--'); plt.xlabel('True stress'); plt.ylabel('Prediction'); plt.title(f'{split}: true vs predicted'); plt.tight_layout(); plt.savefig(fig_dir/f"{split}_actual_vs_predicted.png",dpi=160); plt.close()
        plt.figure(figsize=(6,5)); plt.imshow(cm.values, cmap='Blues'); plt.xticks(range(len(classes)),classes,rotation=30,ha='right'); plt.yticks(range(len(classes)),classes); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'{split} confusion matrix');
        for i in range(len(classes)):
            for j in range(len(classes)): plt.text(j,i,int(cm.values[i,j]),ha='center',va='center')
        plt.tight_layout(); plt.savefig(fig_dir/f"{split}_confusion_matrix.png",dpi=160); plt.close()
    metrics["acceptance_floor_final_fusion"] = project_config["acceptance_floor_final_fusion"]
    save_json(metrics, metrics_dir/"member1_metrics.json")
    return metrics
