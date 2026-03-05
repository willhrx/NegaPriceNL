"""
BESS Arbitrage Simulation Runner

Entry point for running the BESS + Wind portfolio simulation.
Compares multiple bidding strategies against actual 2025 prices.

Usage:
    python scripts/run_bess_simulation.py
"""

import sys
from pathlib import Path
import logging
import pickle
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_bess import (
    WIND_FARM_CAPACITY_MW,
    NL_INSTALLED_WIND_CAPACITY_MW,
    BESS_POWER_MW,
    BESS_ENERGY_MWH,
    BESS_SOC_MIN_PCT,
    BESS_SOC_MAX_PCT,
    BESS_INITIAL_SOC_MWH,
    BESS_CHARGE_EFF,
    BESS_DISCHARGE_EFF,
    BESS_MAX_DAILY_CYCLES,
    CHARGE_BID_QUANTILE,
    DISCHARGE_ASK_QUANTILE,
    CHARGE_BID_QUANTILE_AGGRESSIVE,
    DISCHARGE_ASK_QUANTILE_AGGRESSIVE,
    HIGH_RES_PENETRATION_THRESHOLD,
    MTU_DURATION_HOURS,
    MTUS_PER_DAY,
    EPEX_PRICE_FLOOR,
    EPEX_PRICE_CAP,
    BESS_FIGURES_DIR,
    BESS_REPORTS_DIR,
)
from config.settings_v10 import (
    MODELS_DIR,
    DATA_DIR,
    QUANTILES_V10,
    TEST_START_DATE,
    TEST_END_DATE,
)

from src.simulation.portfolio_backtester import PortfolioBacktester, SimulationConfig
from src.simulation.metrics import (
    calculate_metrics,
    outcomes_to_dataframe,
    generate_report,
    generate_comparison_report,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_model_and_data():
    """Load v10 quantile model and feature matrix."""
    logger.info("Loading model and data...")

    # Load model
    model_path = MODELS_DIR / "quantile_regressor_v10.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train_quantile_model_v10.py first."
        )
    model = joblib.load(model_path)
    logger.info(f"  Loaded model: {model_path.name}")

    # Load feature matrix
    feature_matrix_path = DATA_DIR / "processed" / "feature_matrix_v7.csv"
    if not feature_matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {feature_matrix_path}. "
            "Run create_feature_matrix_v7.py first."
        )

    df = pd.read_csv(feature_matrix_path, parse_dates=['datetime'])
    df = df.set_index('datetime')
    logger.info(f"  Loaded feature matrix: {len(df):,} rows")

    # Filter to test period
    test_mask = (df.index >= TEST_START_DATE) & (df.index <= TEST_END_DATE)
    test_df = df[test_mask].copy()
    logger.info(f"  Test period: {len(test_df):,} rows ({len(test_df)//96} days)")

    # Handle missing values
    if test_df['wind_generation_mw'].isna().sum() > 0:
        logger.warning("  Filling NaN in wind_generation_mw with 0")
        test_df['wind_generation_mw'] = test_df['wind_generation_mw'].fillna(0)

    return model, test_df


def get_feature_columns(test_df: pd.DataFrame) -> list:
    """Get feature column names (exclude target and actuals)."""
    exclude = [
        'price_eur_mwh',
        'is_negative_price',
        'price_is_15min',
        'solar_generation_mw',
        'wind_generation_mw',
        'load_mw',
        'flow_NL_DE_mw',
        # Raw NTC columns
        'ntc_da_nl_be_mw', 'ntc_da_be_nl_mw',
        'ntc_da_nl_delu_mw', 'ntc_da_delu_nl_mw',
        'ntc_da_nl_gb_mw', 'ntc_da_gb_nl_mw',
        'ntc_da_nl_no2_mw', 'ntc_da_no2_nl_mw',
    ]
    feature_cols = [c for c in test_df.columns if c not in exclude]
    return feature_cols


def create_visualizations(
    outcomes_df: pd.DataFrame,
    strategy_metrics: dict,
    output_dir: Path,
):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cumulative BESS P&L
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative_pnl = outcomes_df['bess_net_pnl_eur'].cumsum()
    ax.plot(cumulative_pnl.index, cumulative_pnl.values, linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative BESS P&L (EUR)')
    ax.set_title('Cumulative BESS Arbitrage P&L (Conservative Strategy)')
    ax.ticklabel_format(style='plain', axis='y')
    fig.tight_layout()
    fig.savefig(output_dir / 'cumulative_pnl.png', dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: cumulative_pnl.png")

    # 2. Daily P&L distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(outcomes_df['bess_net_pnl_eur'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', label='Zero')
    ax.axvline(
        x=outcomes_df['bess_net_pnl_eur'].mean(),
        color='g',
        linestyle='--',
        label=f"Mean: {outcomes_df['bess_net_pnl_eur'].mean():.0f} EUR"
    )
    ax.set_xlabel('Daily BESS P&L (EUR)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Daily BESS P&L')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'daily_pnl_distribution.png', dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: daily_pnl_distribution.png")

    # 3. Strategy comparison
    if len(strategy_metrics) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(strategy_metrics.keys())
        values = [m.incremental_bess_value_eur for m in strategy_metrics.values()]

        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
        bars = ax.barh(names, values, color=colors, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Incremental BESS Value (EUR)')
        ax.set_title('Strategy Comparison: BESS Value vs Wind-Only')

        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.annotate(
                f'{val:,.0f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5 if width > 0 else -5, 0),
                textcoords='offset points',
                ha='left' if width > 0 else 'right',
                va='center',
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(output_dir / 'strategy_comparison.png', dpi=150)
        plt.close(fig)
        logger.info(f"  Saved: strategy_comparison.png")

    # 4. Monthly summary
    outcomes_df['month'] = outcomes_df.index.month
    monthly = outcomes_df.groupby('month').agg({
        'bess_net_pnl_eur': 'sum',
        'energy_discharged_mwh': 'sum',
        'wind_revenue_eur': 'sum',
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Monthly BESS P&L
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in monthly['bess_net_pnl_eur']]
    axes[0].bar(monthly.index, monthly['bess_net_pnl_eur'], color=colors, edgecolor='black')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('BESS Net P&L (EUR)')
    axes[0].set_title('Monthly BESS P&L')
    axes[0].set_xticks(range(1, 13))

    # Monthly cycles
    monthly['cycles'] = monthly['energy_discharged_mwh'] / BESS_ENERGY_MWH
    axes[1].bar(monthly.index, monthly['cycles'], color='#3498db', edgecolor='black')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Equivalent Cycles')
    axes[1].set_title('Monthly BESS Cycling')
    axes[1].set_xticks(range(1, 13))

    fig.tight_layout()
    fig.savefig(output_dir / 'monthly_summary.png', dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: monthly_summary.png")


def main():
    """Main entry point for BESS simulation."""
    parser = argparse.ArgumentParser(description='Run BESS arbitrage simulation')
    parser.add_argument('--bess-power', type=float, default=BESS_POWER_MW)
    parser.add_argument('--bess-energy', type=float, default=BESS_ENERGY_MWH)
    parser.add_argument('--wind-capacity', type=float, default=WIND_FARM_CAPACITY_MW)
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BESS ARBITRAGE SIMULATION")
    logger.info("=" * 80)
    logger.info(f"Wind Farm: {args.wind_capacity} MW")
    logger.info(f"BESS: {args.bess_power} MW / {args.bess_energy} MWh")
    logger.info("")

    # Load data
    model, test_df = load_model_and_data()
    feature_columns = get_feature_columns(test_df)
    logger.info(f"  Using {len(feature_columns)} features")

    # Create configurations for different strategies
    configs = {
        'Wind Only': None,  # Special case - no BESS
        'BESS Perfect Foresight': 'perfect',  # Special case
        'BESS Conservative (q25/q75)': SimulationConfig(
            wind_capacity_mw=args.wind_capacity,
            bess_power_mw=args.bess_power,
            bess_energy_mwh=args.bess_energy,
            charge_quantile=CHARGE_BID_QUANTILE,
            discharge_quantile=DISCHARGE_ASK_QUANTILE,
        ),
        'BESS Aggressive (q40/q60)': SimulationConfig(
            wind_capacity_mw=args.wind_capacity,
            bess_power_mw=args.bess_power,
            bess_energy_mwh=args.bess_energy,
            charge_quantile=CHARGE_BID_QUANTILE_AGGRESSIVE,
            discharge_quantile=DISCHARGE_ASK_QUANTILE_AGGRESSIVE,
        ),
    }

    # Run simulations
    all_outcomes = {}
    all_metrics = {}

    for strategy_name, config in configs.items():
        logger.info(f"\nRunning: {strategy_name}")

        if config is None:
            # Wind only baseline
            base_config = SimulationConfig(wind_capacity_mw=args.wind_capacity)
            backtester = PortfolioBacktester(model, QUANTILES_V10, base_config)
            outcomes = backtester.run_wind_only(
                test_df,
                price_column='price_eur_mwh',
                wind_column='wind_generation_mw',
            )
        elif config == 'perfect':
            # Perfect foresight
            base_config = SimulationConfig(
                wind_capacity_mw=args.wind_capacity,
                bess_power_mw=args.bess_power,
                bess_energy_mwh=args.bess_energy,
            )
            backtester = PortfolioBacktester(model, QUANTILES_V10, base_config)
            outcomes = backtester.run_perfect_foresight(
                test_df,
                feature_columns=feature_columns,
                price_column='price_eur_mwh',
                wind_column='wind_generation_mw',
            )
        else:
            # Regular strategy
            backtester = PortfolioBacktester(model, QUANTILES_V10, config)
            outcomes = backtester.run(
                test_df,
                feature_columns=feature_columns,
                price_column='price_eur_mwh',
                wind_column='wind_generation_mw',
                res_penetration_column='forecast_res_penetration',
            )

        all_outcomes[strategy_name] = outcomes

    # Calculate metrics
    wind_only_outcomes = all_outcomes['Wind Only']

    for strategy_name, outcomes in all_outcomes.items():
        metrics = calculate_metrics(
            outcomes,
            wind_only_outcomes=wind_only_outcomes,
            bess_power_mw=args.bess_power,
            bess_energy_mwh=args.bess_energy,
        )
        all_metrics[strategy_name] = metrics
        logger.info(
            f"  {strategy_name}: Portfolio = {metrics.total_portfolio_revenue_eur:,.0f} EUR, "
            f"BESS Value = {metrics.incremental_bess_value_eur:,.0f} EUR"
        )

    # Generate reports
    logger.info("\nGenerating reports...")
    BESS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Comparison report
    comparison_report = generate_comparison_report(all_metrics)
    print("\n" + comparison_report)

    # Detailed report for conservative strategy
    conservative_metrics = all_metrics['BESS Conservative (q25/q75)']
    detailed_report = generate_report(conservative_metrics, 'BESS Conservative (q25/q75)')

    report_path = BESS_REPORTS_DIR / 'bess_simulation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(comparison_report)
        f.write("\n\n")
        f.write(detailed_report)
    logger.info(f"  Saved: {report_path}")

    # Save outcomes to CSV
    conservative_outcomes = all_outcomes['BESS Conservative (q25/q75)']
    outcomes_df = outcomes_to_dataframe(conservative_outcomes)
    outcomes_csv_path = BESS_REPORTS_DIR / 'bess_daily_outcomes.csv'
    outcomes_df.to_csv(outcomes_csv_path)
    logger.info(f"  Saved: {outcomes_csv_path}")

    # Create visualizations
    logger.info("\nCreating visualizations...")
    create_visualizations(outcomes_df, all_metrics, BESS_FIGURES_DIR)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best strategy: {max(all_metrics.items(), key=lambda x: x[1].incremental_bess_value_eur)[0]}")
    logger.info(f"Conservative BESS Value: {conservative_metrics.incremental_bess_value_eur:,.0f} EUR")
    logger.info(f"Perfect Foresight Upper Bound: {all_metrics['BESS Perfect Foresight'].incremental_bess_value_eur:,.0f} EUR")


if __name__ == '__main__':
    main()
