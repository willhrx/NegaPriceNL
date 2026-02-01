"""
Threshold Optimization for Classification

Finds the optimal classification threshold to maximize F1 score
or other metrics for imbalanced classification problems.
"""

from typing import Tuple, Callable, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
    step: float = 0.01
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the specified metric.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    metric : str
        Metric to optimize: 'f1', 'precision', 'recall', 'f2' (F-beta with beta=2)
    min_threshold : float
        Minimum threshold to consider
    max_threshold : float
        Maximum threshold to consider
    step : float
        Step size for threshold search

    Returns
    -------
    Tuple[float, float]
        (best_threshold, best_score)
    """
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            # F-beta with beta=2 (weights recall higher than precision)
            score = fbeta_score(y_true, y_pred, beta=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 2) -> float:
    """
    Calculate F-beta score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    beta : float
        Beta parameter (beta > 1 weights recall higher)

    Returns
    -------
    float
        F-beta score
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    if precision + recall == 0:
        return 0.0

    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.5,
    min_recall: float = 0.5
) -> Tuple[float, dict]:
    """
    Find threshold that maximizes F1 while meeting minimum precision/recall constraints.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities
    min_precision : float
        Minimum acceptable precision
    min_recall : float
        Minimum acceptable recall

    Returns
    -------
    Tuple[float, dict]
        (best_threshold, metrics_dict)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Check constraints
        if precision >= min_precision and recall >= min_recall:
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

    # If no threshold meets constraints, find best F1 anyway
    if not best_metrics:
        best_threshold, best_f1 = optimize_threshold(y_true, y_proba, metric='f1')
        y_pred = (y_proba >= best_threshold).astype(int)
        best_metrics = {
            'threshold': best_threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': best_f1,
            'constraints_met': False
        }
    else:
        best_metrics['constraints_met'] = True

    return best_threshold, best_metrics


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> dict:
    """
    Analyze metrics across different thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    thresholds : np.ndarray, optional
        Thresholds to evaluate. Default: 0.1 to 0.9 in steps of 0.1

    Returns
    -------
    dict
        Dictionary with threshold as key, metrics dict as value
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    results = {}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        results[round(t, 2)] = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'predicted_positive': y_pred.sum(),
            'actual_positive': y_true.sum()
        }

    return results
