"""
utils/metrics.py
----------------
Lightweight metric utilities used during training and evaluation.
These functions wrap sklearn so that the rest of the pipeline
doesn't need to import sklearn directly in training-critical paths.
"""

from typing import List, Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute percentage accuracy.

    Args:
        y_true : Ground-truth integer labels.
        y_pred : Predicted integer labels.

    Returns:
        Accuracy as a percentage float (e.g. 92.5 for 92.5 %).
    """
    return 100.0 * accuracy_score(y_true, y_pred)


def compute_precision_recall_f1(
    y_true:  List[int],
    y_pred:  List[int],
    average: str = "macro",
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1-score.

    Args:
        y_true  : Ground-truth integer labels.
        y_pred  : Predicted integer labels.
        average : Averaging strategy — "macro" | "micro" | "weighted".

    Returns:
        (precision, recall, f1)  as Python floats.
    """
    precision = precision_score(y_true, y_pred, average=average,
                                zero_division=0)
    recall    = recall_score(y_true, y_pred, average=average,
                             zero_division=0)
    f1        = f1_score(y_true, y_pred, average=average,
                         zero_division=0)
    return float(precision), float(recall), float(f1)


def compute_all_metrics(
    y_true:  List[int],
    y_pred:  List[int],
    average: str = "macro",
) -> dict:
    """
    Convenience wrapper that returns all metrics as a dict.

    Returns:
        {
          "accuracy":  float (%),
          "precision": float,
          "recall":    float,
          "f1":        float,
        }
    """
    acc               = compute_accuracy(y_true, y_pred)
    prec, rec, f1     = compute_precision_recall_f1(y_true, y_pred, average)
    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
    }
