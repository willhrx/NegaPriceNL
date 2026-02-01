"""
Economic backtester for negative price prediction strategies.

Runs backtest on historical data to evaluate the economic value
of different prediction strategies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from src.evaluation.economic_metrics import (
    calculate_economic_value,
    calculate_classification_metrics,
    summarize_by_period
)
from src.evaluation.benchmark_strategies import BaseStrategy, get_all_strategies

logger = logging.getLogger(__name__)


class EconomicBacktester:
    """
    Run economic backtest comparing multiple prediction strategies.

    Parameters
    ----------
    capacity_mw : float
        Solar asset capacity in MW (for scaling generation)
    """

    def __init__(self, capacity_mw: float = 10.0):
        self.capacity_mw = capacity_mw
        self.results = {}
        self.strategies = []

    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the backtest."""
        self.strategies.append(strategy)

    def set_strategies(self, strategies: List[BaseStrategy]):
        """Set all strategies at once."""
        self.strategies = strategies

    def run(
        self,
        df: pd.DataFrame,
        price_col: str = 'price_eur_mwh',
        generation_col: str = 'solar_generation_mw',
        target_col: str = 'is_negative_price'
    ) -> pd.DataFrame:
        """
        Run backtest on all strategies.

        Parameters
        ----------
        df : pd.DataFrame
            Test data with features, prices, generation, and target
        price_col : str
            Column name for electricity prices
        generation_col : str
            Column name for solar generation
        target_col : str
            Column name for actual negative price indicator

        Returns
        -------
        pd.DataFrame
            Comparison of all strategies
        """
        if not self.strategies:
            raise ValueError("No strategies added. Call add_strategy() or set_strategies() first.")

        # Extract required data
        prices = df[price_col].values
        generation = df[generation_col].values
        y_true = df[target_col].values

        results = []

        for strategy in self.strategies:
            logger.info(f"Running backtest for: {strategy.name}")

            try:
                # Generate predictions
                y_pred = strategy.predict(df)

                # Calculate economic metrics
                econ_metrics = calculate_economic_value(
                    y_true=y_true,
                    y_pred=y_pred,
                    prices=prices,
                    generation=generation,
                    capacity_mw=self.capacity_mw
                )

                # Calculate classification metrics
                class_metrics = calculate_classification_metrics(y_true, y_pred)

                # Combine results
                result = {
                    'strategy': strategy.name,
                    **econ_metrics,
                    **class_metrics
                }
                results.append(result)

                # Store detailed results
                self.results[strategy.name] = {
                    'predictions': y_pred,
                    'economic_metrics': econ_metrics,
                    'classification_metrics': class_metrics
                }

            except Exception as e:
                logger.error(f"Error running {strategy.name}: {e}")
                continue

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)

        # Sort by capture rate (descending)
        if 'capture_rate_pct' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('capture_rate_pct', ascending=False)

        return comparison_df

    def get_strategy_predictions(self, strategy_name: str) -> Optional[np.ndarray]:
        """Get predictions for a specific strategy."""
        if strategy_name in self.results:
            return self.results[strategy_name]['predictions']
        return None

    def generate_summary_report(self, comparison_df: pd.DataFrame) -> str:
        """
        Generate a text summary report of backtest results.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Results from run()

        Returns
        -------
        str
            Formatted report
        """
        report = []
        report.append("=" * 70)
        report.append("ECONOMIC BACKTEST RESULTS")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now()}")
        report.append(f"Asset Capacity: {self.capacity_mw} MW")
        report.append("")

        # Find baseline (Never Curtail)
        baseline_row = comparison_df[comparison_df['strategy'].str.contains('Never', case=False)]
        if len(baseline_row) > 0:
            baseline_revenue = baseline_row.iloc[0]['revenue_no_curtailment_eur']
            report.append(f"Baseline Revenue (No Curtailment): {baseline_revenue:,.2f} EUR")

        # Find perfect foresight reference
        if len(comparison_df) > 0:
            max_savings = comparison_df.iloc[0]['max_possible_savings_eur']
            report.append(f"Maximum Possible Savings: {max_savings:,.2f} EUR")
            report.append("")

        report.append("-" * 70)
        report.append("STRATEGY COMPARISON (sorted by Capture Rate)")
        report.append("-" * 70)

        for _, row in comparison_df.iterrows():
            report.append(f"\n{row['strategy']}")
            report.append(f"  Revenue:      {row['revenue_with_prediction_eur']:>12,.2f} EUR")
            report.append(f"  Savings:      {row['savings_achieved_eur']:>12,.2f} EUR")
            report.append(f"  Capture Rate: {row['capture_rate_pct']:>11.1f}%")
            report.append(f"  Precision:    {row['precision']*100:>11.1f}%")
            report.append(f"  Recall:       {row['recall']*100:>11.1f}%")
            report.append(f"  F1 Score:     {row['f1_score']:>11.3f}")

        report.append("")
        report.append("-" * 70)
        report.append("COST BREAKDOWN (Best Strategy)")
        report.append("-" * 70)

        if len(comparison_df) > 0:
            best = comparison_df.iloc[0]
            report.append(f"Strategy: {best['strategy']}")
            report.append(f"  Avoided Losses (TP):     {best['value_avoided_losses_eur']:>12,.2f} EUR")
            report.append(f"  Captured Revenue (TN):   {best['value_captured_revenue_eur']:>12,.2f} EUR")
            report.append(f"  Missed Revenue (FP):     {best['cost_missed_revenue_eur']:>12,.2f} EUR")
            report.append(f"  Paid to Generate (FN):   {best['cost_paid_to_generate_eur']:>12,.2f} EUR")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)


def run_full_backtest(
    test_data_path: Path,
    model_path: Path,
    threshold_path: Path,
    feature_columns_path: Path,
    capacity_mw: float = 10.0,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convenience function to run full backtest with all strategies.

    Parameters
    ----------
    test_data_path : Path
        Path to test data CSV
    model_path : Path
        Path to trained model
    threshold_path : Path
        Path to optimal threshold
    feature_columns_path : Path
        Path to feature columns list
    capacity_mw : float
        Asset capacity
    output_dir : Path, optional
        Directory to save results

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)

    # Get strategies
    strategies = get_all_strategies(
        model_path=model_path,
        threshold_path=threshold_path,
        feature_columns_path=feature_columns_path
    )

    # Create backtester
    backtester = EconomicBacktester(capacity_mw=capacity_mw)
    backtester.set_strategies(strategies)

    # Run backtest
    logger.info("Running backtest...")
    results = backtester.run(df)

    # Generate report
    report = backtester.generate_summary_report(results)
    print(report)

    # Save results if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison CSV
        csv_path = output_dir / "strategy_comparison.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # Save report
        report_path = output_dir / "backtest_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    return results
