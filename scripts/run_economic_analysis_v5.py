"""
Run Economic Analysis for Negative Price Predictions (v5)

This script:
1. Loads the v5 test data (2025, 15-min resolution)
2. Compares ML model v5 against benchmark strategies
3. Calculates economic value for a 10 MW solar park
4. Generates comparison report and visualizations
5. Includes v4 vs v5 comparison

Usage:
    py scripts/run_economic_analysis_v5.py
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    TEST_START_DATE, INSTALLED_SOLAR_CAPACITY_MW
)
from src.evaluation.economic_metrics import (
    calculate_economic_value,
    calculate_classification_metrics,
)
from src.evaluation.benchmark_strategies import (
    BaseStrategy,
    HeuristicStrategy,
    NeverCurtailStrategy,
    MLStrategy,
)
from src.evaluation.backtester import EconomicBacktester

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
ASSET_CAPACITY_MW = 10.0  # Hypothetical solar park size
MODEL_VERSION = "v5"
HOURS_PER_PERIOD = 0.25   # 15-min resolution


# === v5-specific benchmark strategies ===

class NaiveStrategyV5(BaseStrategy):
    """Predict negative if D-2 same quarter-hour was negative."""

    def __init__(self):
        super().__init__(name="Naive (D-2 Same QH)")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if 'is_negative_d2_same_qh' not in df.columns:
            raise ValueError("DataFrame must contain 'is_negative_d2_same_qh' column")
        return df['is_negative_d2_same_qh'].fillna(0).astype(int).values


class SolarThresholdStrategyV5(BaseStrategy):
    """
    Predict negative if forecast RES penetration > threshold AND
    forecast load < median.

    Uses D-1 safe forecast features instead of delivery-day actuals.
    """

    def __init__(self, res_threshold: float = 0.5):
        super().__init__(name=f"Solar Threshold (Forecast RES>{res_threshold*100:.0f}% + Low Load)")
        self.res_threshold = res_threshold

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if 'forecast_res_penetration' not in df.columns:
            raise ValueError("DataFrame must contain 'forecast_res_penetration' column")
        if 'load_forecast_mw' not in df.columns:
            raise ValueError("DataFrame must contain 'load_forecast_mw' column")

        median_load = df['load_forecast_mw'].median()

        is_risky = (
            (df['forecast_res_penetration'] > self.res_threshold) &
            (df['load_forecast_mw'] < median_load)
        )
        return is_risky.astype(int).values


def load_test_data() -> pd.DataFrame:
    """Load v5 feature matrix and filter to test period (2025)."""
    input_path = PROCESSED_DATA_DIR / "feature_matrix_v5.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {input_path}. "
            "Run create_feature_matrix_v5.py first."
        )

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Filter to test period (2025)
    test_start = pd.Timestamp(TEST_START_DATE, tz='UTC')
    test_df = df[df.index >= test_start].copy()

    # Fill NaN in generation/load actuals with 0 (no generation = no revenue impact)
    for col in ['solar_generation_mw', 'wind_generation_mw', 'load_mw']:
        if col in test_df.columns:
            n_nan = test_df[col].isna().sum()
            if n_nan > 0:
                logger.info(f"  Filling {n_nan} NaN in {col} with 0")
                test_df[col] = test_df[col].fillna(0)

    logger.info(f"  Loaded {len(test_df):,} test records (from {test_df.index.min()} to {test_df.index.max()})")
    logger.info(f"  Negative price QHs: {test_df['is_negative_price'].sum():,} ({test_df['is_negative_price'].mean()*100:.2f}%)")
    logger.info(f"  Resolution: 15-min ({HOURS_PER_PERIOD}h per period)")

    return test_df


def create_visualizations(results: pd.DataFrame, backtester: EconomicBacktester, test_df: pd.DataFrame):
    """Create economic analysis visualizations."""
    logger.info("Creating visualizations...")

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # 1. Strategy Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(14, 7))

    plot_df = results[results['strategy'] != 'Never Curtail'].copy()
    plot_df = plot_df.sort_values('capture_rate_pct', ascending=True)

    colors = ['#2ecc71' if 'ML' in s else '#3498db' for s in plot_df['strategy']]

    bars = ax.barh(plot_df['strategy'], plot_df['capture_rate_pct'], color=colors)

    ax.set_xlabel('Capture Rate (%)', fontsize=12)
    ax.set_title('Strategy Comparison: Capture Rate of Maximum Possible Savings\n(v5 - D-1 Safe Features, 15-min Resolution)', fontsize=14)
    x_min = min(0, plot_df['capture_rate_pct'].min() * 1.15)
    x_max = max(110, plot_df['capture_rate_pct'].max() * 1.15)
    ax.set_xlim(x_min, x_max)
    ax.axvline(x=0, color='black', linewidth=0.8)

    for bar, val in zip(bars, plot_df['capture_rate_pct']):
        offset = -2 if val < 0 else 2
        ha = 'right' if val < 0 else 'left'
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', ha=ha, fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'strategy_comparison_v5.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'strategy_comparison_v5.png'}")

    # 2. Economic Waterfall Chart
    fig, ax = plt.subplots(figsize=(12, 7))

    baseline = results[results['strategy'] == 'Never Curtail'].iloc[0]
    ml_row = results[results['strategy'].str.contains('ML')].iloc[0]

    categories = [
        'Baseline\n(No Curtailment)',
        'Losses from\nNegative Prices',
        'ML Avoided\nLosses',
        'ML Missed\nRevenue (FP)',
        'ML Paid to\nGenerate (FN)',
        'Final\nRevenue'
    ]

    baseline_rev = baseline['revenue_no_curtailment_eur']
    losses = ml_row['max_possible_savings_eur']
    avoided = ml_row['value_avoided_losses_eur']
    missed = ml_row['cost_missed_revenue_eur']
    paid = ml_row['cost_paid_to_generate_eur']
    final = ml_row['revenue_with_prediction_eur']

    values = [baseline_rev, -losses, avoided, -missed, -paid, 0]
    cumulative = [baseline_rev]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(final)

    colors_wf = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22', '#e74c3c', '#3498db']

    bars = ax.bar(categories, [abs(v) if i in [1,3,4] else v for i, v in enumerate(values)],
                  bottom=[0 if i in [0,5] else cumulative[i-1] if values[i] >= 0 else cumulative[i] for i in range(len(values))],
                  color=colors_wf, edgecolor='black', linewidth=0.5)

    ax.axhline(y=baseline_rev, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.axhline(y=final, color='green', linestyle='--', alpha=0.5, label='Final')

    ax.set_ylabel('Revenue (EUR)', fontsize=12)
    ax.set_title(f'Economic Value Waterfall: {ASSET_CAPACITY_MW} MW Solar Park\n(v5 - 2025 Test, 15-min Resolution)', fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    ax.text(0, baseline_rev + 5000, f'{baseline_rev:,.0f}', ha='center', fontsize=9)
    ax.text(5, final + 5000, f'{final:,.0f}', ha='center', fontsize=9)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'economic_waterfall_v5.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'economic_waterfall_v5.png'}")

    # 3. Confusion Matrix with Economic Values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = np.array([
        [ml_row['true_negatives'], ml_row['false_positives']],
        [ml_row['false_negatives'], ml_row['true_positives']]
    ])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted +', 'Predicted -'],
                yticklabels=['Actual +', 'Actual -'])
    axes[0].set_title('Classification Confusion Matrix\n(Count of Quarter-Hours)', fontsize=12)

    econ_cm = np.array([
        [ml_row['value_captured_revenue_eur'], ml_row['cost_missed_revenue_eur']],
        [ml_row['cost_paid_to_generate_eur'], ml_row['value_avoided_losses_eur']]
    ])

    annot_labels = [[f'{v:,.0f}' for v in row] for row in econ_cm]

    sns.heatmap(econ_cm, annot=annot_labels, fmt='', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Predicted +', 'Predicted -'],
                yticklabels=['Actual +', 'Actual -'])
    axes[1].set_title('Economic Confusion Matrix\n(EUR Impact)', fontsize=12)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'confusion_matrix_economic_v5.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'confusion_matrix_economic_v5.png'}")

    # 4. Cumulative Savings Over Time
    fig, ax = plt.subplots(figsize=(14, 6))

    ml_preds = backtester.get_strategy_predictions(ml_row['strategy'])
    test_df_copy = test_df.copy()
    test_df_copy['ml_pred'] = ml_preds

    # Scale national generation to asset capacity
    gen_max = test_df_copy['solar_generation_mw'].max()
    if gen_max > 0:
        test_df_copy['gen_scaled'] = test_df_copy['solar_generation_mw'] * (ASSET_CAPACITY_MW / gen_max)
    else:
        test_df_copy['gen_scaled'] = 0

    # Revenue calculations (with hours_per_period for 15-min)
    test_df_copy['rev_no_curtail'] = test_df_copy['gen_scaled'] * test_df_copy['price_eur_mwh'] * HOURS_PER_PERIOD
    test_df_copy['rev_ml'] = test_df_copy['gen_scaled'] * test_df_copy['price_eur_mwh'] * (1 - test_df_copy['ml_pred']) * HOURS_PER_PERIOD
    test_df_copy['rev_perfect'] = test_df_copy['gen_scaled'] * np.maximum(test_df_copy['price_eur_mwh'], 0) * HOURS_PER_PERIOD

    # Daily aggregation
    daily = test_df_copy.resample('D').agg({
        'rev_no_curtail': 'sum',
        'rev_ml': 'sum',
        'rev_perfect': 'sum',
    })

    daily['savings_ml'] = daily['rev_ml'] - daily['rev_no_curtail']
    daily['savings_perfect'] = daily['rev_perfect'] - daily['rev_no_curtail']

    # Cumulative
    daily['cum_savings_ml'] = daily['savings_ml'].cumsum()
    daily['cum_savings_perfect'] = daily['savings_perfect'].cumsum()

    ax.fill_between(daily.index, 0, daily['cum_savings_perfect'],
                    alpha=0.3, color='green', label='Perfect Foresight (Max)')
    ax.plot(daily.index, daily['cum_savings_ml'], color='blue', linewidth=2,
            label=f'ML Model {MODEL_VERSION}')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='No Curtailment')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Savings (EUR)', fontsize=12)
    ax.set_title(f'Cumulative Savings Over 2025 Test Period\n({ASSET_CAPACITY_MW} MW Solar Park, 15-min Resolution)', fontsize=14)
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'cumulative_savings_v5.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'cumulative_savings_v5.png'}")


def main():
    """Main economic analysis function."""
    logger.info("=" * 70)
    logger.info("NEGAPRICENL ECONOMIC ANALYSIS v5")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")
    logger.info(f"Asset Capacity: {ASSET_CAPACITY_MW} MW Solar Park")
    logger.info(f"Model Version: {MODEL_VERSION}")
    logger.info(f"Resolution: 15-min (hours_per_period={HOURS_PER_PERIOD})")

    try:
        # Load test data
        test_df = load_test_data()

        # Define model paths
        model_path = MODELS_DIR / f"gradient_boost_negative_price_{MODEL_VERSION}.pkl"
        threshold_path = MODELS_DIR / f"optimal_threshold_{MODEL_VERSION}.pkl"
        feature_cols_path = MODELS_DIR / f"feature_columns_{MODEL_VERSION}.pkl"

        # Check model exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Build v5-specific strategy list
        strategies = [
            NeverCurtailStrategy(),
            NaiveStrategyV5(),
            HeuristicStrategy(),
            SolarThresholdStrategyV5(res_threshold=0.5),
            MLStrategy(
                model_path=model_path,
                threshold_path=threshold_path,
                feature_columns_path=feature_cols_path,
                model_version=MODEL_VERSION
            ),
        ]

        # Create backtester with 15-min resolution
        backtester = EconomicBacktester(
            capacity_mw=ASSET_CAPACITY_MW,
            hours_per_period=HOURS_PER_PERIOD
        )
        backtester.set_strategies(strategies)

        # Run backtest
        logger.info("\nRunning economic backtest...")
        results = backtester.run(
            test_df,
            price_col='price_eur_mwh',
            generation_col='solar_generation_mw',
            target_col='is_negative_price'
        )

        # Generate and print report
        report = backtester.generate_summary_report(results)
        print("\n" + report)

        # Save results
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = REPORTS_DIR / f"economic_analysis_{MODEL_VERSION}.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"\nResults saved to: {csv_path}")

        report_path = REPORTS_DIR / f"economic_report_{MODEL_VERSION}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

        # v4 vs v5 comparison
        v4_metrics_path = MODELS_DIR / "test_metrics_v4.pkl"
        v4_report_path = REPORTS_DIR / "economic_analysis_v4.csv"

        comparison_lines = []
        comparison_lines.append("\n" + "=" * 70)
        comparison_lines.append("v4 vs v5 COMPARISON (Impact of D-1 12:00 Cutoff)")
        comparison_lines.append("=" * 70)

        if v4_metrics_path.exists():
            v4_metrics = joblib.load(v4_metrics_path)
            v5_metrics_path = MODELS_DIR / "test_metrics_v5.pkl"
            v5_metrics = joblib.load(v5_metrics_path) if v5_metrics_path.exists() else None

            comparison_lines.append(f"  {'':20} {'v4':>12} {'v5':>12} {'Delta':>12}")
            comparison_lines.append(f"  {'Features':<20} {'62':>12} {'55':>12} {'-7':>12}")
            comparison_lines.append(f"  {'Resolution':<20} {'Hourly':>12} {'15-min':>12} {'':>12}")

            if v5_metrics:
                comparison_lines.append(f"  {'Recall':<20} {v4_metrics['recall']*100:>11.1f}% {v5_metrics['recall']*100:>11.1f}% {(v5_metrics['recall']-v4_metrics['recall'])*100:>+11.1f}%")
                comparison_lines.append(f"  {'Precision':<20} {v4_metrics['precision']*100:>11.1f}% {v5_metrics['precision']*100:>11.1f}% {(v5_metrics['precision']-v4_metrics['precision'])*100:>+11.1f}%")
                comparison_lines.append(f"  {'F1':<20} {v4_metrics['f1']:>12.3f} {v5_metrics['f1']:>12.3f} {v5_metrics['f1']-v4_metrics['f1']:>+12.3f}")

        # Add economic comparison if v4 analysis exists
        if v4_report_path.exists():
            v4_econ = pd.read_csv(v4_report_path)
            v4_ml = v4_econ[v4_econ['strategy'].str.contains('ML')]
            v5_ml = results[results['strategy'].str.contains('ML')]

            if len(v4_ml) > 0 and len(v5_ml) > 0:
                v4_cr = v4_ml.iloc[0]['capture_rate_pct']
                v5_cr = v5_ml.iloc[0]['capture_rate_pct']
                comparison_lines.append(f"  {'Capture Rate':<20} {v4_cr:>11.1f}% {v5_cr:>11.1f}% {v5_cr-v4_cr:>+11.1f}%")
        else:
            comparison_lines.append("  (v4 economic analysis not available for comparison)")

        comparison_text = "\n".join(comparison_lines)
        logger.info(comparison_text)

        # Append comparison to report
        with open(report_path, 'a') as f:
            f.write("\n" + comparison_text)

        # Create visualizations
        create_visualizations(results, backtester, test_df)

        # Summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("ECONOMIC ANALYSIS v5 COMPLETE")
        logger.info("=" * 70)

        ml_row = results[results['strategy'].str.contains('ML')].iloc[0]
        baseline_row = results[results['strategy'] == 'Never Curtail'].iloc[0]

        logger.info(f"\nKey Results for {MODEL_VERSION} Model:")
        logger.info(f"  Baseline Revenue:     {baseline_row['revenue_no_curtailment_eur']:>15,.2f} EUR")
        logger.info(f"  ML Model Revenue:     {ml_row['revenue_with_prediction_eur']:>15,.2f} EUR")
        logger.info(f"  Savings Achieved:     {ml_row['savings_achieved_eur']:>15,.2f} EUR")
        logger.info(f"  Capture Rate:         {ml_row['capture_rate_pct']:>14.1f}%")
        logger.info(f"  Precision:            {ml_row['precision']*100:>14.1f}%")
        logger.info(f"  Recall:               {ml_row['recall']*100:>14.1f}%")

        # Check targets
        logger.info("\nTarget Achievement:")
        capture_target = 70
        logger.info(f"  Capture Rate (>{capture_target}%): {'PASS' if ml_row['capture_rate_pct'] > capture_target else 'FAIL'}")

        return results

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
