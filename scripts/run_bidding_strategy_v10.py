"""
Bid Curve Strategy Backtesting for v10 Model

Compares multiple bidding strategies for the Dutch day-ahead auction market:

ML-Based Adaptive Strategies (use quantile forecasts):
1. Expected Value Bid - Risk-neutral optimal (trapezoidal integration)
2. Median Bid (q50) - Conservative, 50% curtailment risk
3. Quantile Bids (q10, q25, q40) - Risk-averse variants

Fixed Floor Strategies (non-adaptive baselines):
4. Fixed Floors (-€5, -€10, -€20) - Common industry practice

Benchmarks:
5. Perfect Foresight - Theoretical upper bound

All strategies evaluated on 2025 test year with:
- Auction clearing logic: generate if actual_price >= bid_price
- Capture rate: savings achieved / max possible savings
- Revenue accounting for curtailment decisions

Usage:
    python scripts/run_bidding_strategy_v10.py
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

from config.settings_v10 import (
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    QUANTILES_V10,
    TEST_START_DATE,
    TEST_END_DATE
)
from src.evaluation.bid_curve_strategies import (
    ExpectedValueBidStrategy,
    MedianBidStrategy,
    QuantileBidStrategy,
    FixedFloorBidStrategy,
    PerfectForesightBidStrategy
)
from src.evaluation.auction_backtester import AuctionBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v10"
CAPACITY_MW = 10.0  # 10 MW solar park
HOURS_PER_PERIOD = 0.25  # Quarter-hourly data

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_model_and_data():
    """
    Load v10 model and test data.

    Returns:
        model: Trained NegativePriceQuantileRegressor
        test_df: Test DataFrame with prices, generation, and features
    """
    logger.info("=" * 80)
    logger.info("LOADING MODEL AND DATA")
    logger.info("=" * 80)

    # Load model
    model_path = MODELS_DIR / f"quantile_regressor_{MODEL_VERSION}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run train_quantile_model_v10.py first."
        )

    model = joblib.load(model_path)
    logger.info(f"✓ Loaded model from {model_path}")

    # Load test predictions to get feature columns and indices
    preds_path = MODELS_DIR / f"test_predictions_{MODEL_VERSION}.pkl"
    with open(preds_path, 'rb') as f:
        predictions = pickle.load(f)

    logger.info(f"✓ Loaded test predictions from {preds_path}")

    # Get test indices
    test_index = predictions['index_test']
    y_test = predictions['y_test']

    logger.info(f"  Test period: {test_index[0].date()} to {test_index[-1].date()}")
    logger.info(f"  Test samples: {len(test_index):,} quarter-hours")

    # Load feature matrix (same as train script)
    input_path = PROCESSED_DATA_DIR / "feature_matrix_v7.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {input_path}. "
            "Run create_feature_matrix_v7.py first."
        )

    logger.info(f"Loading feature matrix from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Filter to test period
    test_mask = (df.index >= TEST_START_DATE) & (df.index <= TEST_END_DATE)
    test_df = df[test_mask].copy()

    logger.info(f"✓ Loaded test data: {len(test_df):,} samples")

    # Handle NaN values in generation data
    nan_count = test_df['solar_generation_mw'].isna().sum()
    if nan_count > 0:
        logger.warning(f"  Found {nan_count} NaN values in solar_generation_mw - filling with 0")
        test_df['solar_generation_mw'] = test_df['solar_generation_mw'].fillna(0)

    # Log data quality metrics
    logger.info(f"  NaN in solar_generation_mw: {test_df['solar_generation_mw'].isna().sum()}")
    logger.info(f"  NaN in price_eur_mwh: {test_df['price_eur_mwh'].isna().sum()}")

    # Verify required columns
    required_cols = ['price_eur_mwh', 'solar_generation_mw']
    missing = [col for col in required_cols if col not in test_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"  Price range: [{test_df['price_eur_mwh'].min():.1f}, "
               f"{test_df['price_eur_mwh'].max():.1f}] EUR/MWh")
    logger.info(f"  Negative price hours: {(test_df['price_eur_mwh'] < 0).sum():,} "
               f"({(test_df['price_eur_mwh'] < 0).mean()*100:.1f}%)")

    return model, test_df


def create_strategies(model):
    """
    Create all bidding strategies for comparison.

    Args:
        model: Trained quantile regressor

    Returns:
        List of strategy instances
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING BIDDING STRATEGIES")
    logger.info("=" * 80)

    strategies = []

    # ML-Based Adaptive Strategies
    logger.info("\nML-Based Adaptive Strategies (use quantile forecasts):")

    expected_value = ExpectedValueBidStrategy(model, QUANTILES_V10)
    strategies.append(expected_value)
    logger.info(f"  ✓ {expected_value.name}")

    median = MedianBidStrategy(model, QUANTILES_V10)
    strategies.append(median)
    logger.info(f"  ✓ {median.name}")

    for quantile in [0.10, 0.25, 0.40]:
        q_strategy = QuantileBidStrategy(model, QUANTILES_V10, quantile)
        strategies.append(q_strategy)
        logger.info(f"  ✓ {q_strategy.name}")

    # Fixed Floor Strategies
    logger.info("\nFixed Floor Strategies (non-adaptive baselines):")
    for floor in [-5.0, -10.0, -20.0]:
        fixed = FixedFloorBidStrategy(floor)
        strategies.append(fixed)
        logger.info(f"  ✓ {fixed.name}")

    # Perfect Foresight (upper bound)
    logger.info("\nBenchmark:")
    perfect = PerfectForesightBidStrategy()
    strategies.append(perfect)
    logger.info(f"  ✓ {perfect.name}")

    logger.info(f"\nTotal strategies: {len(strategies)}")

    return strategies


def run_backtests(test_df, strategies):
    """
    Run backtests for all strategies.

    Args:
        test_df: Test data with prices, generation, features
        strategies: List of strategy instances

    Returns:
        results_df: Comparison DataFrame
        backtester: AuctionBacktester instance (for accessing detailed results)
    """
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING BACKTESTS")
    logger.info("=" * 80)

    backtester = AuctionBacktester(
        capacity_mw=CAPACITY_MW,
        hours_per_period=HOURS_PER_PERIOD
    )

    results_df = backtester.run_multiple(
        df=test_df,
        strategies=strategies,
        price_col='price_eur_mwh',
        generation_col='solar_generation_mw'
    )

    return results_df, backtester


def print_results_table(results_df):
    """Print formatted results table."""
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY COMPARISON RESULTS")
    logger.info("=" * 80)

    # Key metrics table
    logger.info(f"\n{'Strategy':<35} {'Capture Rate':<15} {'Avg Bid (EUR/MWh)':<20} {'Curtail %':<12}")
    logger.info("-" * 85)

    for _, row in results_df.iterrows():
        strategy = row['strategy'][:34]  # Truncate if needed
        capture = f"{row['capture_rate_pct']:.1f}%"
        bid = f"{row['avg_bid_price_eur_mwh']:.2f}"
        curtail = f"{row['curtailment_rate_pct']:.1f}%"

        logger.info(f"{strategy:<35} {capture:<15} {bid:<20} {curtail:<12}")

    # Best strategy details
    logger.info("\n" + "=" * 80)
    logger.info("BEST STRATEGY BREAKDOWN")
    logger.info("=" * 80)

    best = results_df.iloc[0]
    logger.info(f"\nStrategy: {best['strategy']}")
    logger.info(f"  Capture Rate:           {best['capture_rate_pct']:.2f}%")
    logger.info(f"  Revenue with bids:      {best['revenue_with_bids_eur']:,.2f} EUR")
    logger.info(f"  Savings achieved:       {best['savings_achieved_eur']:,.2f} EUR")
    logger.info(f"  Avg bid price:          {best['avg_bid_price_eur_mwh']:.2f} EUR/MWh")
    logger.info(f"  Bid std deviation:      {best['bid_price_std_eur_mwh']:.2f} EUR/MWh")
    logger.info(f"  Curtailment rate:       {best['curtailment_rate_pct']:.1f}%")

    logger.info(f"\nValue Breakdown:")
    logger.info(f"  Avoided losses:         {best['value_avoided_losses_eur']:,.2f} EUR")
    logger.info(f"  Captured revenue:       {best['value_captured_revenue_eur']:,.2f} EUR")
    logger.info(f"  Missed revenue:         {best['cost_missed_revenue_eur']:,.2f} EUR")
    logger.info(f"  Paid to generate:       {best['cost_paid_to_generate_eur']:,.2f} EUR")


def plot_capture_rates(results_df, backtester):
    """Plot horizontal bar chart of capture rates."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color code: adaptive vs fixed vs perfect
    colors = []
    for strategy in results_df['strategy']:
        if 'Perfect' in strategy:
            colors.append('#2ecc71')  # Green for theoretical max
        elif 'Fixed Floor' in strategy:
            colors.append('#e74c3c')  # Red for fixed floors
        else:
            colors.append('#3498db')  # Blue for adaptive

    y_pos = np.arange(len(results_df))

    ax.barh(y_pos, results_df['capture_rate_pct'], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['strategy'], fontsize=10)
    ax.set_xlabel('Capture Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Bid Strategy Performance Comparison\n(Savings Achieved / Max Possible × 100%)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(row['capture_rate_pct'] + 1, i, f"{row['capture_rate_pct']:.1f}%",
               va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, label='Adaptive (ML-based)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Fixed Floor'),
        Patch(facecolor='#2ecc71', alpha=0.8, label='Perfect Foresight')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    output_path = FIGURES_DIR / f"bid_strategy_capture_rates_{MODEL_VERSION}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved capture rates plot to {output_path}")
    plt.close()


def plot_bid_distributions(results_df, backtester):
    """Box plot of bid price distributions."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract bid prices for each strategy (exclude Perfect Foresight)
    strategy_names = []
    bid_prices_list = []

    for _, row in results_df.iterrows():
        strategy_name = row['strategy']
        if 'Perfect' in strategy_name:
            continue  # Skip perfect foresight (too many outliers)

        details = backtester.get_strategy_details(strategy_name)
        if details and 'bid_prices' in details:
            strategy_names.append(strategy_name)
            bid_prices_list.append(details['bid_prices'])

    # Create box plot
    bp = ax.boxplot(bid_prices_list, labels=strategy_names, patch_artist=True)

    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        if 'Fixed Floor' in strategy_names[i]:
            patch.set_facecolor('#e74c3c')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.6)

    ax.set_ylabel('Bid Price (EUR/MWh)', fontsize=12, fontweight='bold')
    ax.set_title('Bid Price Distribution by Strategy', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Zero price')

    # Rotate x labels
    plt.xticks(rotation=45, ha='right', fontsize=9)

    plt.tight_layout()

    output_path = FIGURES_DIR / f"bid_price_distributions_{MODEL_VERSION}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved bid distributions plot to {output_path}")
    plt.close()


def plot_bid_vs_price_scatter(backtester, strategy_name='Expected Value (E[price])'):
    """Scatter plot: bid price vs actual price."""
    fig, ax = plt.subplots(figsize=(12, 10))

    details = backtester.get_strategy_details(strategy_name)
    if not details:
        logger.warning(f"No details found for {strategy_name}")
        return

    bid_prices = details['bid_prices']
    actual_prices = details['prices']
    cleared = details['cleared']

    # Sample for visualization (plot every 10th point to avoid overcrowding)
    sample_idx = np.arange(0, len(bid_prices), 10)

    # Separate cleared vs curtailed
    cleared_mask = cleared[sample_idx] == 1
    curtailed_mask = cleared[sample_idx] == 0

    ax.scatter(actual_prices[sample_idx][cleared_mask],
              bid_prices[sample_idx][cleared_mask],
              c='#2ecc71', alpha=0.4, s=10, label='Cleared (bid accepted)')
    ax.scatter(actual_prices[sample_idx][curtailed_mask],
              bid_prices[sample_idx][curtailed_mask],
              c='#e74c3c', alpha=0.4, s=10, label='Curtailed (bid rejected)')

    # Diagonal line: bid = price (decision boundary)
    lim_min = min(actual_prices.min(), bid_prices.min()) - 10
    lim_max = max(actual_prices.max(), bid_prices.max()) + 10
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=2,
           alpha=0.5, label='Decision boundary (bid = price)')

    # Quadrant regions
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Actual Market Price (EUR/MWh)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bid Price (EUR/MWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'Bid vs Actual Price: {strategy_name}\n'
                f'(sampled every 10th quarter-hour)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = FIGURES_DIR / f"bid_vs_price_scatter_{MODEL_VERSION}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved bid vs price scatter to {output_path}")
    plt.close()


def plot_revenue_waterfall(results_df):
    """Waterfall chart showing revenue progression."""
    fig, ax = plt.subplots(figsize=(14, 8))

    baseline = results_df.iloc[0]['revenue_no_curtailment_eur']
    perfect = results_df.iloc[0]['revenue_perfect_eur']

    # Show top 5 strategies + perfect
    top_strategies = results_df.head(5)

    revenues = [baseline]
    labels = ['Baseline\n(No Curtailment)']

    for _, row in top_strategies.iterrows():
        if 'Perfect' not in row['strategy']:
            revenues.append(row['revenue_with_bids_eur'])
            labels.append(row['strategy'].replace(' ', '\n', 1))  # Wrap long names

    revenues.append(perfect)
    labels.append('Perfect\nForesight')

    # Calculate deltas
    deltas = [revenues[0]] + [revenues[i] - revenues[i-1] for i in range(1, len(revenues))]

    # Colors: baseline gray, strategies blue/red, perfect green
    colors = ['#95a5a6']  # Gray for baseline
    for i in range(1, len(revenues) - 1):
        colors.append('#3498db' if deltas[i] >= 0 else '#e74c3c')
    colors.append('#2ecc71')  # Green for perfect

    # Create waterfall
    x_pos = np.arange(len(revenues))
    bottoms = np.zeros(len(revenues))

    for i in range(1, len(revenues)):
        bottoms[i] = revenues[i-1]

    ax.bar(x_pos, revenues, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Add connecting lines
    for i in range(len(revenues) - 1):
        ax.plot([x_pos[i] + 0.4, x_pos[i+1] - 0.4],
               [revenues[i], revenues[i]],
               'k--', linewidth=1, alpha=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9, ha='center')
    ax.set_ylabel('Revenue (EUR)', fontsize=12, fontweight='bold')
    ax.set_title('Revenue Progression Across Strategies\n(2025 Test Year, 10 MW Park)',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, rev in enumerate(revenues):
        ax.text(i, rev + 5000, f"{rev/1000:.0f}k", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = FIGURES_DIR / f"revenue_waterfall_{MODEL_VERSION}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved revenue waterfall to {output_path}")
    plt.close()


def save_results(results_df):
    """Save results to CSV."""
    # Remove array columns (already in backtester.results)
    results_to_save = results_df.copy()

    output_path = REPORTS_DIR / f"bid_strategy_comparison_{MODEL_VERSION}.csv"
    results_to_save.to_csv(output_path, index=False)
    logger.info(f"✓ Saved results CSV to {output_path}")


def main():
    logger.info("=" * 80)
    logger.info("BID CURVE STRATEGY BACKTESTING - v10 MODEL")
    logger.info("=" * 80)
    logger.info(f"Test Period: {TEST_START_DATE.date()} to {TEST_END_DATE.date()}")
    logger.info(f"Asset Capacity: {CAPACITY_MW} MW")
    logger.info("")

    # Step 1: Load model and data
    model, test_df = load_model_and_data()

    # Step 2: Create strategies
    strategies = create_strategies(model)

    # Step 3: Run backtests
    results_df, backtester = run_backtests(test_df, strategies)

    # Step 4: Print results
    print_results_table(results_df)

    # Step 5: Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    plot_capture_rates(results_df, backtester)
    plot_bid_distributions(results_df, backtester)
    plot_bid_vs_price_scatter(backtester)
    plot_revenue_waterfall(results_df)

    # Step 6: Save results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    save_results(results_df)

    # Summary report
    report = backtester.generate_summary_report(results_df)
    report_path = REPORTS_DIR / f"bid_strategy_report_{MODEL_VERSION}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"✓ Saved text report to {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nBest Strategy: {results_df.iloc[0]['strategy']}")
    logger.info(f"Capture Rate: {results_df.iloc[0]['capture_rate_pct']:.2f}%")
    logger.info(f"Savings: {results_df.iloc[0]['savings_achieved_eur']:,.2f} EUR")


if __name__ == "__main__":
    main()
