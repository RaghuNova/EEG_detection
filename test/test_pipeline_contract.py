from __future__ import annotations

import numpy as np

from backend.mlout import select_decision_threshold


def test_threshold_selection_uses_validation_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90])
    assert select_decision_threshold(y_true, y_prob) == 0.6
