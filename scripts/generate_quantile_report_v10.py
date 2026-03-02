"""
Generate Clean Quantile Model Report for v10 (Conformalized Quantile Regression)

Focuses exclusively on probabilistic forecasting performance:
- Calibration quality (expected vs observed coverage)
- CRPS (Continuous Ranked Probability Score)
- Quantile crossing rate
- Prediction interval statistics
- Per-quantile pinball loss

NO bidding strategies or economic analysis - pure model evaluation.

Usage:
    python scripts/generate_quantile_report_v10.py
"""

import sys
from pathlib import Path
import logging
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_v10 import MODELS_DIR, REPORTS_DIR, QUANTILES_V10

# Import directly from module to avoid __init__.py loading sklearn
import importlib.util
spec = importlib.util.spec_from_file_location(
    "regression_metrics",
    PROJECT_ROOT / "src" / "evaluation" / "regression_metrics.py"
)
regression_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression_metrics)

evaluate_quantile_forecast = regression_metrics.evaluate_quantile_forecast
compare_calibration = regression_metrics.compare_calibration
crps_quantile = regression_metrics.crps_quantile
mean_quantile_loss = regression_metrics.mean_quantile_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v10"


def load_model_artifacts():
    """Load v10 metrics and predictions (model not needed for report)."""
    logger.info("Loading v10 model artifacts...")

    metrics_path = MODELS_DIR / f"test_metrics_{MODEL_VERSION}.pkl"
    preds_path = MODELS_DIR / f"test_predictions_{MODEL_VERSION}.pkl"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics not found at {metrics_path}. "
            f"Run train_quantile_model_v10.py first."
        )

    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    logger.info(f"✓ Loaded metrics: {metrics_path.name}")

    with open(preds_path, 'rb') as f:
        predictions = pickle.load(f)
    logger.info(f"✓ Loaded predictions: {preds_path.name}")

    # Model not needed for report generation
    return None, metrics, predictions


def generate_report(model, metrics: dict, predictions: dict) -> str:
    """Generate comprehensive quantile model report."""

    report = []
    report.append("=" * 80)
    report.append("QUANTILE REGRESSION MODEL REPORT - v10")
    report.append("Conformalized Quantile Regression with Rolling Window Calibration")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Model architecture
    report.append("-" * 80)
    report.append("MODEL ARCHITECTURE")
    report.append("-" * 80)
    report.append(f"Model Type: XGBoost Quantile Regression Ensemble")
    report.append(f"Quantile Levels: {QUANTILES_V10}")
    report.append(f"Number of Quantiles: {len(QUANTILES_V10)}")
    report.append(f"Calibration Method: Conformal Prediction (Static + Rolling)")
    report.append(f"Rolling Window: 30 days")
    report.append("")

    # Test set info
    test_index = predictions['index_test']
    y_test = predictions['y_test']
    report.append(f"Test Period: {test_index[0].date()} to {test_index[-1].date()}")
    report.append(f"Test Samples: {len(test_index):,} quarter-hours ({len(test_index)/96:.0f} days)")
    report.append(f"Target Variable: Electricity Price (EUR/MWh)")
    report.append(f"Price Range: [{y_test.min():.2f}, {y_test.max():.2f}] EUR/MWh")
    report.append("")

    # Extract metrics
    raw_m = metrics['raw']
    static_m = metrics['static']
    rolling_m = metrics['rolling']

    # Summary metrics table
    report.append("-" * 80)
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 80)
    report.append("")
    report.append(f"{'Metric':<35} {'Raw':<15} {'Static Cal':<15} {'Rolling Cal':<15}")
    report.append("-" * 80)
    report.append(f"{'CRPS (lower is better)':<35} {raw_m['crps']:<15.4f} {static_m['crps']:<15.4f} {rolling_m['crps']:<15.4f}")
    report.append(f"{'Mean Quantile Loss':<35} {raw_m['mean_quantile_loss']:<15.4f} {static_m['mean_quantile_loss']:<15.4f} {rolling_m['mean_quantile_loss']:<15.4f}")
    report.append(f"{'80% Prediction Interval Coverage':<35} {raw_m['coverage_80pct']*100:<14.1f}% {static_m['coverage_80pct']*100:<14.1f}% {rolling_m['coverage_80pct']*100:<14.1f}%")
    report.append(f"{'Expected 80% Coverage':<35} 80.0%")
    report.append(f"{'Quantile Crossing Rate':<35} {raw_m['crossing_rate']*100:<14.2f}% {static_m['crossing_rate']*100:<14.2f}% {rolling_m['crossing_rate']*100:<14.2f}%")
    report.append(f"{'Mean Interval Width (EUR/MWh)':<35} {raw_m.get('mean_width', 0):<14.1f} {static_m.get('mean_width', 0):<14.1f} {rolling_m.get('mean_width', 0):<14.1f}")
    report.append("")

    # Calibration analysis
    report.append("-" * 80)
    report.append("CALIBRATION ANALYSIS")
    report.append("-" * 80)
    report.append("")
    report.append("A well-calibrated model should have observed coverage matching the expected")
    report.append("quantile level (e.g., 10% of actuals should fall below the q10 prediction).")
    report.append("")

    # Raw calibration
    report.append("RAW MODEL (Uncalibrated):")
    report.append(f"{'Quantile':<12} {'Expected':<12} {'Observed':<12} {'Deviation':<12}")
    report.append("-" * 50)

    # Safe access to calibration data
    if 'calibration' not in raw_m:
        report.append("  [Calibration data not available]")
        report.append("")
        raw_cal = None
    else:
        raw_cal = raw_m['calibration']

    if raw_cal is not None:
        for _, row in raw_cal.iterrows():
            alpha = row['alpha']
            expected = row['expected_below']
            observed = row['observed_below']
            deviation = (observed - expected) * 100
            report.append(f"q{alpha:<10.2f} {expected*100:<11.1f}% {observed*100:<11.1f}% {deviation:>+10.1f}pp")
        report.append("")

    # Rolling calibrated
    report.append("ROLLING CALIBRATED MODEL (Production):")
    report.append(f"{'Quantile':<12} {'Expected':<12} {'Observed':<12} {'Deviation':<12}")
    report.append("-" * 50)

    if 'calibration' not in rolling_m:
        report.append("  [Calibration data not available]")
        report.append("")
        rolling_cal = None
    else:
        rolling_cal = rolling_m['calibration']

    if rolling_cal is not None:
        for _, row in rolling_cal.iterrows():
            alpha = row['alpha']
            expected = row['expected_below']
            observed = row['observed_below']
            deviation = (observed - expected) * 100
            report.append(f"q{alpha:<10.2f} {expected*100:<11.1f}% {observed*100:<11.1f}% {deviation:>+10.1f}pp")
        report.append("")

    # Calibration comparison table
    report.append("-" * 80)
    report.append("CALIBRATION IMPROVEMENT (Raw vs Rolling)")
    report.append("-" * 80)
    report.append("")

    comp = metrics['calibration_comparison_rolling']
    report.append(f"{'Quantile':<12} {'Raw Dev':<12} {'Cal Dev':<12} {'Improvement':<12}")
    report.append("-" * 50)
    for _, row in comp.iterrows():
        quantile = row['quantile']
        raw_dev = row['raw_deviation_pp']
        cal_dev = row['cal_deviation_pp']
        improvement = row['improvement_pp']
        report.append(f"q{quantile:<10.2f} {raw_dev:>10.2f}pp {cal_dev:>10.2f}pp {improvement:>+10.2f}pp")
    report.append("")

    # Key calibration statistics
    mean_raw_dev = comp['raw_deviation_pp'].mean()
    mean_cal_dev = comp['cal_deviation_pp'].mean()
    max_raw_dev = comp['raw_deviation_pp'].max()
    max_cal_dev = comp['cal_deviation_pp'].max()

    report.append(f"Mean Calibration Deviation:")
    report.append(f"  Raw:     {mean_raw_dev:.2f}pp")
    report.append(f"  Rolling: {mean_cal_dev:.2f}pp")
    report.append(f"  Improvement: {mean_raw_dev - mean_cal_dev:+.2f}pp")
    report.append("")
    report.append(f"Maximum Calibration Deviation:")
    report.append(f"  Raw:     {max_raw_dev:.2f}pp")
    report.append(f"  Rolling: {max_cal_dev:.2f}pp")
    report.append("")

    # Per-quantile pinball loss
    report.append("-" * 80)
    report.append("PER-QUANTILE PINBALL LOSS")
    report.append("-" * 80)
    report.append("")
    report.append("Lower values indicate better fit for that quantile level.")
    report.append("")
    report.append(f"{'Quantile':<15} {'Raw':<15} {'Rolling Cal':<15} {'Improvement':<15}")
    report.append("-" * 60)

    raw_per_alpha = raw_m['quantile_loss_per_alpha']
    rolling_per_alpha = rolling_m['quantile_loss_per_alpha']

    for alpha in QUANTILES_V10:
        # Safe dictionary access with fallback
        raw_loss = raw_per_alpha.get(alpha, float('nan'))
        rolling_loss = rolling_per_alpha.get(alpha, float('nan'))
        improvement = raw_loss - rolling_loss
        report.append(f"q{alpha:<13.2f} {raw_loss:<15.4f} {rolling_loss:<15.4f} {improvement:>+14.4f}")
    report.append("")

    # Success criteria evaluation
    report.append("-" * 80)
    report.append("MODEL QUALITY ASSESSMENT")
    report.append("-" * 80)
    report.append("")

    # Median calibration (most critical)
    # Use np.isclose for floating point comparison
    q50_mask = np.isclose(comp['quantile'], 0.50)
    if not q50_mask.any():
        # Fallback: find closest to 0.50
        q50_row = comp.iloc[(comp['quantile'] - 0.50).abs().argmin()]
    else:
        q50_row = comp[q50_mask].iloc[0]
    q50_raw_dev = q50_row['raw_deviation_pp']
    q50_cal_dev = q50_row['cal_deviation_pp']

    report.append(f"Median (q50) Calibration:")
    report.append(f"  Raw deviation:     {q50_raw_dev:>6.2f}pp")
    report.append(f"  Rolling deviation: {q50_cal_dev:>6.2f}pp")

    if q50_cal_dev < 1:
        report.append(f"  Status: ✓ EXCELLENT (target <1pp)")
    elif q50_cal_dev < 3:
        report.append(f"  Status: ✓ GOOD (target <3pp)")
    elif q50_cal_dev < 5:
        report.append(f"  Status: ~ ACCEPTABLE (target <5pp)")
    else:
        report.append(f"  Status: ✗ NEEDS IMPROVEMENT")
    report.append("")

    # Overall calibration quality
    report.append(f"Overall Calibration Quality:")
    report.append(f"  Mean deviation: {mean_cal_dev:.2f}pp across all quantiles")

    if mean_cal_dev < 3:
        report.append(f"  Status: ✓ EXCELLENT - All quantiles well-calibrated")
    elif mean_cal_dev < 5:
        report.append(f"  Status: ✓ GOOD - Minor calibration drift")
    elif mean_cal_dev < 10:
        report.append(f"  Status: ~ ACCEPTABLE - Moderate calibration error")
    else:
        report.append(f"  Status: ✗ POOR - Significant calibration issues")
    report.append("")

    # CRPS assessment
    # Avoid division by zero
    if raw_m['crps'] > 0:
        crps_improvement = (raw_m['crps'] - rolling_m['crps']) / raw_m['crps'] * 100
    else:
        crps_improvement = 0.0

    report.append(f"Probabilistic Accuracy (CRPS):")
    report.append(f"  Raw CRPS:     {raw_m['crps']:.4f}")
    report.append(f"  Rolling CRPS: {rolling_m['crps']:.4f}")
    report.append(f"  Improvement:  {crps_improvement:+.1f}%")
    report.append("")

    # Quantile crossing
    report.append(f"Quantile Crossing Rate:")
    report.append(f"  Rolling: {rolling_m['crossing_rate']*100:.2f}%")
    if rolling_m['crossing_rate'] == 0:
        report.append(f"  Status: ✓ PERFECT - No quantile crossings (monotonicity enforced)")
    elif rolling_m['crossing_rate'] < 0.01:
        report.append(f"  Status: ✓ EXCELLENT - <1% crossing rate")
    else:
        report.append(f"  Status: ~ MINOR VIOLATIONS - {rolling_m['crossing_rate']*100:.1f}% crossings")
    report.append("")

    # Overall assessment
    report.append("-" * 80)
    report.append("OVERALL MODEL ASSESSMENT")
    report.append("-" * 80)
    report.append("")

    # Count excellent/good/acceptable metrics
    excellent_count = 0
    good_count = 0

    if q50_cal_dev < 1:
        excellent_count += 1
    elif q50_cal_dev < 3:
        good_count += 1

    if mean_cal_dev < 3:
        excellent_count += 1
    elif mean_cal_dev < 5:
        good_count += 1

    if rolling_m['crossing_rate'] == 0:
        excellent_count += 1
    elif rolling_m['crossing_rate'] < 0.01:
        good_count += 1

    if crps_improvement > 5:
        excellent_count += 1
    elif crps_improvement > 0:
        good_count += 1

    report.append(f"Key Performance Indicators:")
    report.append(f"  ✓ Excellent: {excellent_count}/4")
    report.append(f"  ✓ Good:      {good_count}/4")
    report.append("")

    if excellent_count >= 3:
        report.append("Overall Rating: ✓✓✓ EXCELLENT")
        report.append("Model is production-ready with exceptional calibration quality.")
    elif excellent_count + good_count >= 3:
        report.append("Overall Rating: ✓✓ GOOD")
        report.append("Model performs well with reliable probabilistic forecasts.")
    else:
        report.append("Overall Rating: ~ ACCEPTABLE")
        report.append("Model may benefit from further tuning or more calibration data.")

    report.append("")
    report.append("-" * 80)
    report.append("KEY STRENGTHS:")
    report.append("-" * 80)

    strengths = []
    if q50_cal_dev < 1:
        strengths.append("• Median forecast is exceptionally well-calibrated (<1pp deviation)")
    if mean_cal_dev < 3:
        strengths.append("• All quantile levels achieve excellent calibration (<3pp mean deviation)")
    if rolling_m['crossing_rate'] == 0:
        strengths.append("• Perfect monotonicity - no quantile crossings")
    if crps_improvement > 10:
        strengths.append(f"• Strong CRPS improvement from calibration ({crps_improvement:.1f}%)")
    if rolling_m['coverage_80pct'] >= 0.78 and rolling_m['coverage_80pct'] <= 0.82:
        strengths.append("• 80% prediction interval achieves target coverage")

    if not strengths:
        strengths.append("• Model produces valid quantile forecasts")

    for strength in strengths:
        report.append(strength)

    report.append("")
    report.append("-" * 80)
    report.append("RECOMMENDATIONS:")
    report.append("-" * 80)

    recommendations = []
    if mean_cal_dev > 5:
        recommendations.append("• Consider increasing calibration window size for more stable corrections")
    if rolling_m['crossing_rate'] > 0.01:
        recommendations.append("• Enforce monotonicity post-processing to eliminate quantile crossings")
    if crps_improvement < 5:
        recommendations.append("• Calibration provides limited CRPS improvement - review feature engineering")
    if abs(rolling_m['coverage_80pct'] - 0.80) > 0.05:
        recommendations.append("• Tune quantile levels or calibration to improve 80% interval coverage")

    if not recommendations:
        recommendations.append("✓ No critical issues identified - model is performing as expected")
        recommendations.append("• Consider monitoring calibration drift over time in production")
        recommendations.append("• Periodic recalibration (monthly/quarterly) recommended for sustained performance")
    else:
        for rec in recommendations:
            report.append(rec)

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Generate and save quantile model report."""
    logger.info("=" * 80)
    logger.info("QUANTILE MODEL REPORT GENERATOR - v10")
    logger.info("=" * 80)
    logger.info("")

    # Load artifacts
    model, metrics, predictions = load_model_artifacts()

    # Generate report
    logger.info("")
    logger.info("Generating quantile model report...")
    report_text = generate_report(model, metrics, predictions)

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "quantile_model_report_v10.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"✓ Report saved: {report_path}")
    logger.info("")

    # Print to console (handle encoding issues on Windows)
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        # Fallback: replace Unicode characters for console
        safe_text = report_text.replace('✓', '[OK]').replace('✗', '[X]').replace('•', '-')
        print("\n" + safe_text)

    logger.info("")
    logger.info("=" * 80)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
