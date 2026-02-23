"""
Regression evaluation metrics for quantile price forecasting.

Provides pinball loss, CRPS approximation, calibration diagnostics,
and prediction interval statistics for evaluating quantile forecasts.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Pinball loss (quantile loss) for a single quantile.

    L_alpha(y, q) = (alpha - 1[y < q]) * (y - q)

    Lower is better. A well-calibrated quantile forecast minimises this.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    errors = y_true - y_pred
    return np.mean(np.where(errors >= 0, alpha * errors, (alpha - 1) * errors))


def mean_quantile_loss(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    alphas: List[float],
) -> float:
    """
    Average pinball loss across all quantile levels.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
    quantile_preds : array of shape (n_samples, n_quantiles)
    alphas : list of quantile levels

    Returns
    -------
    float
        Mean pinball loss across all quantiles.
    """
    losses = []
    for i, alpha in enumerate(alphas):
        losses.append(quantile_loss(y_true, quantile_preds[:, i], alpha))
    return np.mean(losses)


def quantile_loss_per_alpha(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    alphas: List[float],
) -> Dict[float, float]:
    """Pinball loss broken down by quantile level."""
    return {
        alpha: quantile_loss(y_true, quantile_preds[:, i], alpha)
        for i, alpha in enumerate(alphas)
    }


def coverage_probability(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
) -> float:
    """
    Fraction of actual values falling within the prediction interval.

    For a [q10, q90] interval, expected coverage is 80%.
    """
    y_true = np.asarray(y_true).flatten()
    q_lower = np.asarray(q_lower).flatten()
    q_upper = np.asarray(q_upper).flatten()
    covered = (y_true >= q_lower) & (y_true <= q_upper)
    return float(np.mean(covered))


def interval_width(q_lower: np.ndarray, q_upper: np.ndarray) -> Dict[str, float]:
    """Statistics on prediction interval width."""
    widths = np.asarray(q_upper).flatten() - np.asarray(q_lower).flatten()
    return {
        'mean_width': float(np.mean(widths)),
        'median_width': float(np.median(widths)),
        'std_width': float(np.std(widths)),
    }


def crps_quantile(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    alphas: List[float],
) -> float:
    """
    Approximate CRPS (Continuous Ranked Probability Score) from quantile forecasts.

    Uses the quantile decomposition:
        CRPS ≈ (2 / K) * Σ_k pinball_loss(alpha_k)

    where K is the number of quantiles. This is an approximation that improves
    with more quantile levels. Lower is better.
    """
    y_true = np.asarray(y_true).flatten()
    total = 0.0
    for i, alpha in enumerate(alphas):
        total += quantile_loss(y_true, quantile_preds[:, i], alpha)
    return 2.0 * total / len(alphas)


def calibration_table(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    alphas: List[float],
) -> pd.DataFrame:
    """
    Compare expected vs observed coverage for each quantile level.

    A well-calibrated model should have observed coverage close to
    the nominal quantile level (e.g., ~10% of actuals below q10).

    Returns
    -------
    pd.DataFrame with columns: alpha, expected_below, observed_below, deviation
    """
    y_true = np.asarray(y_true).flatten()
    rows = []
    for i, alpha in enumerate(alphas):
        observed_below = float(np.mean(y_true < quantile_preds[:, i]))
        rows.append({
            'alpha': alpha,
            'expected_below': alpha,
            'observed_below': observed_below,
            'deviation': observed_below - alpha,
        })
    return pd.DataFrame(rows)


def evaluate_quantile_forecast(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    alphas: List[float],
) -> Dict[str, Any]:
    """
    Full evaluation of a quantile forecast.

    Returns a dict with all key metrics for reporting.
    """
    y_true = np.asarray(y_true).flatten()

    # Per-quantile pinball loss
    per_alpha = quantile_loss_per_alpha(y_true, quantile_preds, alphas)

    # Find q10 and q90 indices
    q10_idx = alphas.index(0.10) if 0.10 in alphas else 0
    q90_idx = alphas.index(0.90) if 0.90 in alphas else len(alphas) - 1

    q_lower = quantile_preds[:, q10_idx]
    q_upper = quantile_preds[:, q90_idx]

    # Coverage and interval
    cov = coverage_probability(y_true, q_lower, q_upper)
    iw = interval_width(q_lower, q_upper)

    # CRPS
    crps = crps_quantile(y_true, quantile_preds, alphas)

    # Calibration
    cal = calibration_table(y_true, quantile_preds, alphas)

    # Quantile crossing rate (should be 0 for well-behaved forecasts)
    crossings = 0
    for i in range(quantile_preds.shape[0]):
        if not np.all(np.diff(quantile_preds[i]) >= 0):
            crossings += 1
    crossing_rate = crossings / quantile_preds.shape[0]

    return {
        'mean_quantile_loss': mean_quantile_loss(y_true, quantile_preds, alphas),
        'quantile_loss_per_alpha': per_alpha,
        'crps': crps,
        'coverage_80pct': cov,
        'expected_coverage_80pct': alphas[q90_idx] - alphas[q10_idx],
        **iw,
        'crossing_rate': crossing_rate,
        'calibration': cal,
    }


def compare_calibration(
    y_true: np.ndarray,
    raw_preds: np.ndarray,
    calibrated_preds: np.ndarray,
    quantiles: List[float]
) -> pd.DataFrame:
    """
    Compare calibration of raw vs calibrated quantile predictions.

    This function is used to evaluate the improvement from conformal calibration.
    The goal is for calibrated predictions to have observed coverage that matches
    the nominal (expected) coverage for each quantile.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Actual observed values
    raw_preds : np.ndarray, shape (n_samples, n_quantiles)
        Raw quantile predictions before conformal calibration
    calibrated_preds : np.ndarray, shape (n_samples, n_quantiles)
        Calibrated quantile predictions after conformal calibration
    quantiles : List[float]
        Quantile levels (e.g., [0.05, 0.10, ..., 0.95])

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - quantile: Quantile level
        - expected: Nominal coverage (same as quantile level)
        - raw_observed: Fraction of actuals below raw prediction
        - calibrated_observed: Fraction of actuals below calibrated prediction
        - improvement_pp: Improvement in percentage points (lower is better for deviations)

    Example
    -------
    >>> comp = compare_calibration(y_test, raw_preds, cal_preds, [0.1, 0.5, 0.9])
    >>> print(comp)
       quantile  expected  raw_observed  calibrated_observed  improvement_pp
    0      0.10      0.10         0.189                0.103          -8.600
    1      0.50      0.50         0.708                0.512          -19.600
    2      0.90      0.90         0.901                0.898          -0.300

    The improvement_pp column shows how much closer calibrated predictions are
    to the nominal coverage. Negative values indicate reduction in bias.
    """
    y_true = np.asarray(y_true).flatten()

    if raw_preds.shape != calibrated_preds.shape:
        raise ValueError(
            f"raw_preds shape ({raw_preds.shape}) must match "
            f"calibrated_preds shape ({calibrated_preds.shape})"
        )

    if raw_preds.shape[1] != len(quantiles):
        raise ValueError(
            f"Number of quantile columns ({raw_preds.shape[1]}) must match "
            f"length of quantiles list ({len(quantiles)})"
        )

    results = []
    for i, alpha in enumerate(quantiles):
        # Observed coverage: fraction of actuals below the quantile prediction
        raw_obs = (y_true < raw_preds[:, i]).mean()
        cal_obs = (y_true < calibrated_preds[:, i]).mean()

        # Absolute deviation from nominal
        raw_dev = abs(raw_obs - alpha)
        cal_dev = abs(cal_obs - alpha)

        # Improvement: reduction in absolute deviation (negative = better)
        improvement = (cal_dev - raw_dev) * 100

        results.append({
            'quantile': alpha,
            'expected': alpha,
            'raw_observed': raw_obs,
            'calibrated_observed': cal_obs,
            'raw_deviation_pp': raw_dev * 100,
            'cal_deviation_pp': cal_dev * 100,
            'improvement_pp': improvement
        })

    return pd.DataFrame(results)
