from __future__ import annotations

import numpy as np

from fullnote_ehr.direct_fusion import binary_metrics, select_f1_threshold


def test_threshold_selection_and_binary_metrics():
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    selected = select_f1_threshold(labels, probabilities)
    assert np.isclose(selected["threshold"], 0.6)
    assert np.isclose(selected["validation_f1"], 1.0)
    metrics = binary_metrics(labels, probabilities, selected["threshold"])
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
