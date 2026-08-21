"""Compute and save the evaluation metrics + confusion matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from src.common import clinical_class_from_stress


def evaluate_predictions(df_pred: pd.DataFrame, cfg: dict, name: str) -> dict:
    y = df_pred["true_stress"].values.astype(float)
    p = df_pred["prediction"].values.astype(float)

    # continuous metrics
    mae = float(mean_absolute_error(y, p))
    mse = float(mean_squared_error(y, p))
    rmse = float(np.sqrt(mse))
    pear = float(pearsonr(y, p)[0])
    spear = float(spearmanr(y, p)[0])

    # Class metrics.
    # True class: use the SUPPLIED clinical_class column when available (authoritative),
    # else derive from stress with the documented right-closed bins.
    # Predicted class: always derived from predicted stress with the same bins.
    classes = cfg["class_names"]
    thresholds = cfg.get("class_thresholds", (3, 5, 7))
    if "clinical_class" in df_pred.columns and df_pred["clinical_class"].notna().all():
        yc = df_pred["clinical_class"].astype(str).values
    else:
        yc = clinical_class_from_stress(y, thresholds)
    pc = clinical_class_from_stress(p, thresholds)

    prec, rec, f1, sup = precision_recall_fscore_support(
        yc, pc, labels=classes, average=None, zero_division=0
    )
    acc = float(accuracy_score(yc, pc))
    macro_f1 = float(f1_score(yc, pc, labels=classes, average="macro", zero_division=0))
    cm = confusion_matrix(yc, pc, labels=classes)

    return {
        "split": name,
        "n": int(len(y)),
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "Pearson": pear,
        "Spearman": spear,
        "pred_min": float(p.min()),
        "pred_max": float(p.max()),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_precision": dict(zip(classes, [float(x) for x in prec])),
        "per_class_recall": dict(zip(classes, [float(x) for x in rec])),
        "per_class_f1": dict(zip(classes, [float(x) for x in f1])),
        "per_class_support": dict(zip(classes, [int(x) for x in sup])),
        "confusion_matrix": cm.tolist(),
        "class_names": classes,
    }


def save_metrics(report: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # JSON-safe copy
    safe = json_safe(report)
    import json

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def plot_confusion_matrix(report: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.array(report["confusion_matrix"])
    classes = report["class_names"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{report['split']} — confusion matrix")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
