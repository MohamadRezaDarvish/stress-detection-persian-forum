from __future__ import annotations
import numpy as np
import pandas as pd


def ensemble_predict(models: list, X: pd.DataFrame, prediction_range: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.vstack([model.predict(X) for model in models])
    lo, hi = prediction_range
    matrix = np.clip(matrix, lo, hi)
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=0), matrix
