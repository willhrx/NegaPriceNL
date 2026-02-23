"""
Run Quantile Forecast Analysis for Model v10 (Conformalized QR)

Loads the trained v10 model and generates comprehensive analysis including:
- Raw vs calibrated performance comparison
- Calibration table showing convergence to nominal coverage
- CRPS, coverage, crossing rate metrics
- Probability-based metrics (P(price < 0), expected negative price)

Usage:
    python scripts/run_quantile_analysis_v10.py
"""

import sys
from pathlib import Path
import logging
import pickle

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_v10 import MODELS_DIR, FIGURES_DIR, QUANTILES_V10

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v10"

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_model_and_predictions():
    """Load saved v10 model, calibrator, and test predictions."""
    logger.info("Loading v10 model artifacts...")

    model_path = MODELS_DIR / f"quantile_regressor_{MODEL_VERSION}.pkl"
    calibrator_path = MODELS_DIR / f"static_calibrator_{MODEL_VERSION}.pkl"
    metrics_path = MODELS_DIR / f"test_metrics_{MODEL_VERSION}.pkl"
    preds_path = MODELS_DIR / f"test_predictions_{MODEL_VERSION}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run train_quantile_model_v10.py first."
        )

    model = joblib.load(model_path)
    logger.info(f"  Loaded model from {model_path}")

    from src.models.conformal_calibrator import ConformalCalibrator
    calibrator = ConformalCalibrator.load(calibrator_path)
    logger.info(f"  Loaded calibrator from {calibrator_path}")

    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    logger.info(f"  Loaded metrics from {metrics_path}")

    with open(preds_path, 'rb') as f:
        predictions = pickle.load(f)
    logger.info(f"  Loaded predictions from {preds_path}")

    return model, calibrator, metrics, predictions


def print_summary_table(metrics: dict, predictions: dict):
    """Print comprehensive summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("QUANTILE FORECAST ANALYSIS v10 — CONFORMALIZED QUANTILE REGRESSION")
    logger.info("=" * 80)

    raw_m = metrics['raw']
    static_m = metrics['static']
    rolling_m = metrics['rolling']

    logger.info(f"\n{'Metric':<30} {'Raw':<15} {'Static Cal':<15} {'Rolling Cal':<15}")
    logger.info("-" * 80)
    logger.info(f"{'Mean Quantile Loss':<30} {raw_m['mean_quantile_loss']:<15.4f} {static_m['mean_quantile_loss']:<15.4f} {rolling_m['mean_quantile_loss']:<15.4f}")
    logger.info(f"{'CRPS':<30} {raw_m['crps']:<15.4f} {static_m['crps']:<15.4f} {rolling_m['crps']:<15.4f}")
    logger.info(f"{'Coverage (80%)':<30} {raw_m['coverage_80pct']*100:<14.1f}% {static_m['coverage_80pct']*100:<14.1f}% {rolling_m['coverage_80pct']*100:<14.1f}%")
    logger.info(f"{'Crossing Rate':<30} {raw_m['crossing_rate']*100:<14.2f}% {static_m['crossing_rate']*100:<14.2f}% {rolling_m['crossing_rate']*100:<14.2f}%")
    logger.info(f"{'Mean Interval Width':<30} {raw_m.get('mean_width', 0):<14.1f}  {static_m.get('mean_width', 0):<14.1f}  {rolling_m.get('mean_width', 0):<14.1f}")

    # Calibration tables
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION TABLES")
    logger.info("=" * 80)

    logger.info("\nRAW (Uncalibrated):")
    logger.info("  Quantile   Expected   Observed   Deviation")
    logger.info("  " + "-" * 50)
    for alpha in QUANTILES_V10:
        obs = raw_m['calibration'].get(alpha, 0)
        dev = abs(obs - alpha) * 100
        status = "✓" if dev < 3 else ("~" if dev < 5 else "✗")
        logger.info(f"  q{alpha:.2f}     {alpha*100:5.1f}%     {obs*100:5.1f}%     {dev:5.1f}pp {status}")

    logger.info("\nROLLING CALIBRATED (30-day window):")
    logger.info("  Quantile   Expected   Observed   Deviation")
    logger.info("  " + "-" * 50)
    for alpha in QUANTILES_V10:
        obs = rolling_m['calibration'].get(alpha, 0)
        dev = abs(obs - alpha) * 100
        status = "✓" if dev < 3 else ("~" if dev < 5 else "✗")
        logger.info(f"  q{alpha:.2f}     {alpha*100:5.1f}%     {obs*100:5.1f}%     {dev:5.1f}pp {status}")

    # Improvement summary
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION IMPROVEMENT (Raw → Rolling)")
    logger.info("=" * 80)

    comp = metrics['calibration_comparison_rolling']
    logger.info("\n" + comp.to_string(index=False))

    # Success indicators
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS INDICATORS")
    logger.info("=" * 80)

    # Check median calibration using comparison dataframe
    comp_rolling = metrics['calibration_comparison_rolling']
    q50_row = comp_rolling[comp_rolling['quantile'] == 0.50].iloc[0]

    q50_raw_obs = q50_row['raw_observed']
    q50_rolling_obs = q50_row['calibrated_observed']

    q50_raw_dev = abs(q50_raw_obs - 0.50) * 100
    q50_rolling_dev = abs(q50_rolling_obs - 0.50) * 100

    logger.info(f"\nMedian (q50) Calibration Bias:")
    logger.info(f"  Raw:     {q50_raw_dev:.1f}pp deviation (observed {q50_raw_obs*100:.1f}%)")
    logger.info(f"  Rolling: {q50_rolling_dev:.1f}pp deviation (observed {q50_rolling_obs*100:.1f}%)")
    logger.info(f"  Improvement: {q50_raw_dev - q50_rolling_dev:+.1f}pp")

    if q50_rolling_dev < 3:
        logger.info("  ✓ EXCELLENT: q50 bias <3pp")
    elif q50_rolling_dev < 5:
        logger.info("  ~ GOOD: q50 bias <5pp")
    elif q50_rolling_dev < 10:
        logger.info("  ≈ ACCEPTABLE: q50 bias <10pp")
    else:
        logger.info("  ✗ NEEDS IMPROVEMENT: q50 bias >10pp")

    # Overall calibration quality
    mean_dev_rolling = comp_rolling['cal_deviation_pp'].mean()
    max_dev_rolling = comp_rolling['cal_deviation_pp'].max()

    logger.info(f"\nOverall Calibration (Rolling):")
    logger.info(f"  Mean deviation: {mean_dev_rolling:.1f}pp")
    logger.info(f"  Max deviation:  {max_dev_rolling:.1f}pp")

    if mean_dev_rolling < 3:
        logger.info("  ✓ EXCELLENT: All quantiles well-calibrated")
    elif mean_dev_rolling < 5:
        logger.info("  ~ GOOD: Most quantiles well-calibrated")
    else:
        logger.info("  ≈ ACCEPTABLE: Some calibration drift remains")

    # Crossing rate
    if rolling_m['crossing_rate'] == 0:
        logger.info("\n✓ Crossing rate: 0% (monotonicity enforced)")
    else:
        logger.info(f"\n~ Crossing rate: {rolling_m['crossing_rate']*100:.2f}%")


def plot_calibration_comparison(metrics: dict):
    """Plot calibration curves for raw vs calibrated."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use comparison dataframe
    comp_rolling = metrics['calibration_comparison_rolling']

    quantiles = comp_rolling['quantile'].values
    raw_obs = comp_rolling['raw_observed'].values
    rolling_obs = comp_rolling['calibrated_observed'].values

    # Plot 1: Raw vs Rolling
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    axes[0].plot(quantiles, raw_obs, 'o-', label='Raw (uncalibrated)', markersize=8, linewidth=2)
    axes[0].plot(quantiles, rolling_obs, 's-', label='Rolling calibrated', markersize=8, linewidth=2)
    axes[0].set_xlabel('Expected Coverage (Quantile Level)', fontsize=12)
    axes[0].set_ylabel('Observed Coverage', fontsize=12)
    axes[0].set_title('Calibration Curve: Raw vs Rolling', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Deviation from nominal
    raw_dev = comp_rolling['raw_deviation_pp'].values
    rolling_dev = comp_rolling['cal_deviation_pp'].values

    x = np.arange(len(quantiles))
    width = 0.35

    axes[1].bar(x - width/2, raw_dev, width, label='Raw', alpha=0.7)
    axes[1].bar(x + width/2, rolling_dev, width, label='Rolling', alpha=0.7)
    axes[1].axhline(3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='3pp threshold')
    axes[1].set_xlabel('Quantile', fontsize=12)
    axes[1].set_ylabel('Absolute Deviation (pp)', fontsize=12)
    axes[1].set_title('Calibration Deviation from Nominal', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'q{alpha:.2f}' for alpha in quantiles], rotation=45)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = FIGURES_DIR / f"calibration_comparison_{MODEL_VERSION}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nSaved calibration comparison plot to {output_path}")
    plt.close()


def plot_prediction_intervals(predictions: dict):
    """Plot prediction intervals for a sample period."""
    y_test = predictions['y_test']
    index_test = predictions['index_test']
    raw_preds = predictions['raw_preds']
    cal_preds = predictions['cal_preds_rolling']

    # Sample: First 30 days
    n_days = 30
    n_samples = n_days * 96
    if len(y_test) < n_samples:
        n_samples = len(y_test)

    y_sample = y_test[:n_samples]
    idx_sample = index_test[:n_samples]
    raw_sample = raw_preds[:n_samples]
    cal_sample = cal_preds[:n_samples]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot 1: Raw predictions
    axes[0].plot(idx_sample, y_sample, 'k.', markersize=2, label='Actual', alpha=0.6)
    axes[0].fill_between(idx_sample, raw_sample[:, 0], raw_sample[:, -1],
                          alpha=0.2, label='q05-q95 interval')
    axes[0].fill_between(idx_sample, raw_sample[:, 1], raw_sample[:, -2],
                          alpha=0.3, label='q10-q90 interval')
    axes[0].plot(idx_sample, raw_sample[:, 4], 'r-', linewidth=1.5, label='Median (q50)')
    axes[0].set_ylabel('Price (EUR/MWh)', fontsize=12)
    axes[0].set_title('Raw (Uncalibrated) Predictions', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Plot 2: Calibrated predictions
    axes[1].plot(idx_sample, y_sample, 'k.', markersize=2, label='Actual', alpha=0.6)
    axes[1].fill_between(idx_sample, cal_sample[:, 0], cal_sample[:, -1],
                          alpha=0.2, label='q05-q95 interval')
    axes[1].fill_between(idx_sample, cal_sample[:, 1], cal_sample[:, -2],
                          alpha=0.3, label='q10-q90 interval')
    axes[1].plot(idx_sample, cal_sample[:, 4], 'b-', linewidth=1.5, label='Median (q50)')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Price (EUR/MWh)', fontsize=12)
    axes[1].set_title('Rolling Calibrated Predictions (30-day window)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    output_path = FIGURES_DIR / f"prediction_intervals_{MODEL_VERSION}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved prediction intervals plot to {output_path}")
    plt.close()


def main():
    logger.info("=" * 80)
    logger.info("QUANTILE FORECAST ANALYSIS v10")
    logger.info("=" * 80)

    # Load artifacts
    model, calibrator, metrics, predictions = load_model_and_predictions()

    # Print summary
    print_summary_table(metrics, predictions)

    # Generate plots
    plot_calibration_comparison(metrics)
    plot_prediction_intervals(predictions)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
