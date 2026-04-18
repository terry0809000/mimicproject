import numpy as np

from src.evaluation.metrics import classification_report_dict


def test_metrics_keys():
    out = classification_report_dict(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert "macro_f1" in out
