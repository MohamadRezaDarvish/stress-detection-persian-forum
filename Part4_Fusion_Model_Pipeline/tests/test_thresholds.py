
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import score_to_predicted_class


def test_prediction_threshold_boundaries():
    thresholds = [3.1, 4.9, 6.5]
    scores = np.array([1.0, 3.099, 3.1, 4.899, 4.9, 6.499, 6.5, 9.0])
    result = score_to_predicted_class(scores, thresholds).tolist()
    assert result == [
        "Low", "Low", "Moderate", "Moderate",
        "High", "High", "Very High", "Very High"
    ]
