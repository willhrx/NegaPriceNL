"""
Run Economic Analysis for Negative Price Predictions

This script:
1. Loads the test data (2025)
2. Compares ML model v4 against benchmark strategies
3. Calculates economic value for a 10 MW solar park
4. Generates comparison report and visualizations

Usage:
    py scripts/run_economic_analysis.py
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
    NaiveStrategy,
    HeuristicStrategy,
    SolarThresholdStrategy,
    NeverCurtailStrategy,
    MLStrategy,
    get_all_strategies,
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
MODEL_VERSION = "v4"


def load_test_data() -> pd.DataFrame:
    """Load feature matrix and filter to test period (2025)."""
    input_path = PROCESSED_DATA_DIR / "feature_matrix.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {input_path}. "
            "Run create_feature_matrix.py first."
        )

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Filter to test period (2025)
    test_start = pd.Timestamp(TEST_START_DATE, tz='UTC')
    test_df = df[df.index >= test_start].copy()

    logger.info(f"  Loaded {len(test_df):,} test records (from {test_df.index.min()} to {test_df.index.max()})")
    logger.info(f"  Negative price hours: {test_df['is_negative_price'].sum():,} ({test_df['is_negative_price'].mean()*100:.2f}%)")

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
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by capture rate
    plot_df = results[results['strategy'] != 'Never Curtail'].copy()
    plot_df = plot_df.sort_values('capture_rate_pct', ascending=True)

    colors = ['#2ecc71' if 'ML' in s else '#3498db' for s in plot_df['strategy']]

    bars = ax.barh(plot_df['strategy'], plot_df['capture_rate_pct'], color=colors)

    ax.set_xlabel('Capture Rate (%)', fontsize=12)
    ax.set_title('Strategy Comparison: Capture Rate of Maximum Possible Savings\n(Higher is Better)', fontsize=14)
    ax.set_xlim(0, 110)

    # Add value labels
    for bar, val in zip(bars, plot_df['capture_rate_pct']):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'strategy_comparison.png'}")

    # 2. Economic Waterfall Chart
    fig, ax = plt.subplots(figsize=(12, 7))

    # Find key values
    baseline = results[results['strategy'] == 'Never Curtail'].iloc[0]
    ml_row = results[results['strategy'].str.contains('ML')].iloc[0]

    # Waterfall data
    categories = [
        'Baseline\n(No Curtailment)',
        'Losses from\nNegative Prices',
        'ML Avoided\nLosses',
        'ML Missed\nRevenue (FP)',
        'ML Paid to\nGenerate (FN)',
        'Final\nRevenue'
    ]

    baseline_rev = baseline['revenue_no_curtailment_eur']
    losses = ml_row['max_possible_savings_eur']  # What we could have saved
    avoided = ml_row['value_avoided_losses_eur']
    missed = ml_row['cost_missed_revenue_eur']
    paid = ml_row['cost_paid_to_generate_eur']
    final = ml_row['revenue_with_prediction_eur']

    # Create waterfall
    values = [baseline_rev, -losses, avoided, -missed, -paid, 0]
    cumulative = [baseline_rev]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(final)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22', '#e74c3c', '#3498db']

    bars = ax.bar(categories, [abs(v) if i in [1,3,4] else v for i, v in enumerate(values)],
                  bottom=[0 if i in [0,5] else cumulative[i-1] if values[i] >= 0 else cumulative[i] for i in range(len(values))],
                  color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(y=baseline_rev, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.axhline(y=final, color='green', linestyle='--', alpha=0.5, label='Final')

    ax.set_ylabel('Revenue (EUR)', fontsize=12)
    ax.set_title(f'Economic Value Waterfall: {ASSET_CAPACITY_MW} MW Solar Park (2025 Test Period)', fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Add value labels
    ax.text(0, baseline_rev + 5000, f'{baseline_rev:,.0f}', ha='center', fontsize=9)
    ax.text(5, final + 5000, f'{final:,.0f}', ha='center', fontsize=9)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'economic_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'economic_waterfall.png'}")

    # 3. Confusion Matrix with Economic Values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Classification confusion matrix
    ml_preds = backtester.get_strategy_predictions(ml_row['strategy'])
    y_true = test_df['is_negative_price'].values

    cm = np.array([
        [ml_row['true_negatives'], ml_row['false_positives']],
        [ml_row['false_negatives'], ml_row['true_positives']]
    ])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted +', 'Predicted -'],
                yticklabels=['Actual +', 'Actual -'])
    axes[0].set_title('Classification Confusion Matrix\n(Count of Hours)', fontsize=12)

    # Economic confusion matrix
    econ_cm = np.array([
        [ml_row['value_captured_revenue_eur'], ml_row['cost_missed_revenue_eur']],
        [ml_row['cost_paid_to_generate_eur'], ml_row['value_avoided_losses_eur']]
    ])

    # Format as EUR
    annot_labels = [[f'{v:,.0f}' for v in row] for row in econ_cm]

    sns.heatmap(econ_cm, annot=annot_labels, fmt='', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Predicted +', 'Predicted -'],
                yticklabels=['Actual +', 'Actual -'])
    axes[1].set_title('Economic Confusion Matrix\n(EUR Impact)', fontsize=12)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'confusion_matrix_economic.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'confusion_matrix_economic.png'}")

    # 4. Cumulative Savings Over Time
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate daily savings
    test_df_copy = test_df.copy()
    test_df_copy['ml_pred'] = ml_preds

    # Revenue calculations
    test_df_copy['rev_no_curtail'] = test_df_copy['solar_generation_mw'] * test_df_copy['price_eur_mwh']
    test_df_copy['rev_ml'] = test_df_copy['solar_generation_mw'] * test_df_copy['price_eur_mwh'] * (1 - test_df_copy['ml_pred'])
    test_df_copy['rev_perfect'] = test_df_copy['solar_generation_mw'] * np.maximum(test_df_copy['price_eur_mwh'], 0)

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
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Savings (EUR)', fontsize=12)
    ax.set_title(f'Cumulative Savings Over 2025 Test Period\n({ASSET_CAPACITY_MW} MW Solar Park)', fontsize=14)
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'cumulative_savings.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {FIGURES_DIR / 'cumulative_savings.png'}")


def main():
    """Main economic analysis function."""
    logger.info("=" * 70)
    logger.info("NEGAPRICENL ECONOMIC ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")
    logger.info(f"Asset Capacity: {ASSET_CAPACITY_MW} MW Solar Park")
    logger.info(f"Model Version: {MODEL_VERSION}")

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

        # Get all strategies
        strategies = get_all_strategies(
            model_path=model_path,
            threshold_path=threshold_path,
            feature_columns_path=feature_cols_path,
            model_version=MODEL_VERSION
        )

        # Create backtester
        backtester = EconomicBacktester(capacity_mw=ASSET_CAPACITY_MW)
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

        # Save comparison CSV
        csv_path = REPORTS_DIR / f"economic_analysis_{MODEL_VERSION}.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"\nResults saved to: {csv_path}")

        # Save report
        report_path = REPORTS_DIR / f"economic_report_{MODEL_VERSION}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

        # Create visualizations
        create_visualizations(results, backtester, test_df)

        # Summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("ECONOMIC ANALYSIS COMPLETE")
        logger.info("=" * 70)

        # Find ML model results
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
