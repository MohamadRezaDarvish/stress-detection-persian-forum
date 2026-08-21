
from __future__ import annotations

import heapq

import numpy as np

from common import CLASS_ORDER, MINIMUM_RECALL


def _candidate_grid(prediction: np.ndarray) -> np.ndarray:
    unique_values = np.unique(np.round(np.asarray(prediction, dtype=float), 12))
    if len(unique_values) == 1:
        value = unique_values[0]
        return np.array([value - 1e-8, value + 1e-8])
    return np.r_[
        unique_values[0] - 1e-8,
        (unique_values[:-1] + unique_values[1:]) / 2.0,
        unique_values[-1] + 1e-8,
    ]


def search_constrained_thresholds(
    true_class: np.ndarray,
    prediction: np.ndarray,
    validation_mae: float,
    top_n: int = 100,
) -> dict:
    true_class = np.asarray(true_class, dtype=object)
    prediction = np.asarray(prediction, dtype=float)
    label_to_index = {label: index for index, label in enumerate(CLASS_ORDER)}
    true_index = np.array([label_to_index[value] for value in true_class])
    support = np.bincount(true_index, minlength=4)
    grid = _candidate_grid(prediction)

    cumulative = np.zeros((4, len(grid)), dtype=int)
    for class_index in range(4):
        values = np.sort(prediction[true_index == class_index])
        cumulative[class_index] = np.searchsorted(values, grid, side="left")

    low_recall = cumulative[0] / support[0]
    very_high_recall = (support[3] - cumulative[3]) / support[3]

    feasible_count = 0
    best = None
    top_heap: list[tuple] = []
    eligible_t3 = np.where(
        very_high_recall + 1e-12 >= MINIMUM_RECALL[3]
    )[0]

    for t1_index in np.where(
        low_recall + 1e-12 >= MINIMUM_RECALL[0]
    )[0]:
        moderate_recall = (
            cumulative[1] - cumulative[1, t1_index]
        ) / support[1]
        eligible_t2 = np.where(
            (np.arange(len(grid)) > t1_index)
            & (moderate_recall + 1e-12 >= MINIMUM_RECALL[1])
        )[0]

        for t2_index in eligible_t2:
            high_recall_for_t3 = (
                cumulative[2, eligible_t3] - cumulative[2, t2_index]
            ) / support[2]
            eligible_final = eligible_t3[
                (eligible_t3 > t2_index)
                & (
                    high_recall_for_t3 + 1e-12
                    >= MINIMUM_RECALL[2]
                )
            ]

            for t3_index in eligible_final:
                feasible_count += 1
                confusion = np.empty((4, 4), dtype=int)
                confusion[:, 0] = cumulative[:, t1_index]
                confusion[:, 1] = (
                    cumulative[:, t2_index] - cumulative[:, t1_index]
                )
                confusion[:, 2] = (
                    cumulative[:, t3_index] - cumulative[:, t2_index]
                )
                confusion[:, 3] = support - cumulative[:, t3_index]

                diagonal = np.diag(confusion)
                recall = diagonal / support
                predicted_support = confusion.sum(axis=0)
                precision = np.divide(
                    diagonal,
                    predicted_support,
                    out=np.zeros(4, dtype=float),
                    where=predicted_support > 0,
                )
                f1 = np.divide(
                    2.0 * precision * recall,
                    precision + recall,
                    out=np.zeros(4, dtype=float),
                    where=(precision + recall) > 0,
                )
                macro_f1 = float(f1.mean())
                thresholds = [
                    float(grid[t1_index]),
                    float(grid[t2_index]),
                    float(grid[t3_index]),
                ]

                rank_key = (
                    round(float(precision[3]), 12),
                    round(macro_f1, 12),
                    round(float(precision[2]), 12),
                    -float(validation_mae),
                )
                record = {
                    "thresholds": thresholds,
                    "accuracy": float(diagonal.sum() / len(prediction)),
                    "macro_f1": macro_f1,
                    "precision_by_class": dict(
                        zip(CLASS_ORDER, map(float, precision))
                    ),
                    "recall_by_class": dict(
                        zip(CLASS_ORDER, map(float, recall))
                    ),
                    "f1_by_class": dict(zip(CLASS_ORDER, map(float, f1))),
                    "support_by_class": dict(
                        zip(CLASS_ORDER, map(int, support))
                    ),
                    "confusion_matrix": confusion.tolist(),
                    "rank_key": list(rank_key),
                }

                if best is None or rank_key > best["rank_key_tuple"]:
                    best = {
                        "rank_key_tuple": rank_key,
                        "record": record,
                    }

                heap_item = (rank_key, thresholds, record)
                if len(top_heap) < top_n:
                    heapq.heappush(top_heap, heap_item)
                elif rank_key > top_heap[0][0]:
                    heapq.heapreplace(top_heap, heap_item)

    top_records = [
        item[2]
        for item in sorted(top_heap, key=lambda item: item[0], reverse=True)
    ]
    return {
        "status": "feasible" if best is not None else "not_feasible",
        "feasible_count": int(feasible_count),
        "best": None if best is None else best["record"],
        "top": top_records,
    }
