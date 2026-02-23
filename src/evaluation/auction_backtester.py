"""
Auction Backtester for Bid Curve Strategies

Extends economic backtesting to handle continuous bid prices (EUR/MWh)
rather than binary curtail/generate decisions.

Key Differences from Binary Backtester:
1. Strategies return bid prices (EUR/MWh) not binary decisions
2. Clearing logic: generate if actual_price >= bid_price, else curtailed
3. Revenue calculated based on actual clearing prices

This simulates the real auction mechanism: operators submit price floors,
and the market clearing price determines who produces.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from .bid_curve_strategies import BidCurveStrategy

logger = logging.getLogger(__name__)


class AuctionBacktester:
    """
    Backtest bid curve strategies against actual auction clearing prices.

    Simulates the day-ahead auction mechanism where:
    1. Operator submits bid price (EUR/MWh floor)
    2. Market clears at actual price
    3. If actual_price >= bid_price → generate (bid accepted)
    4. If actual_price < bid_price → curtail (bid rejected)

    Attributes:
        capacity_mw: Solar asset capacity in MW
        hours_per_period: Duration of each time period in hours (0.25 for 15-min)
    """

    def __init__(self, capacity_mw: float = 10.0, hours_per_period: float = 0.25):
        """
        Initialize auction backtester.

        Args:
            capacity_mw: Asset capacity in MW (for scaling generation)
            hours_per_period: Period duration (0.25 for quarter-hourly data)
        """
        self.capacity_mw = capacity_mw
        self.hours_per_period = hours_per_period
        self.results = {}

    def run(
        self,
        df: pd.DataFrame,
        strategy: BidCurveStrategy,
        price_col: str = 'price_eur_mwh',
        generation_col: str = 'solar_generation_mw'
    ) -> Dict:
        """
        Backtest a single bid curve strategy.

        Args:
            df: Test data containing prices, generation, and features
            strategy: BidCurveStrategy instance
            price_col: Column name for electricity prices
            generation_col: Column name for solar generation

        Returns:
            Dictionary with comprehensive metrics:
            - revenue_no_curtailment_eur: Baseline (never curtail)
            - revenue_with_bids_eur: Revenue using bid strategy
            - revenue_perfect_eur: Upper bound (perfect foresight)
            - savings_achieved_eur: Improvement vs baseline
            - max_possible_savings_eur: Theoretical maximum
            - capture_rate_pct: Savings achieved / max possible × 100%
            - avg_bid_price_eur_mwh: Mean bid price
            - bid_price_std_eur_mwh: Bid price standard deviation
            - curtailment_rate_pct: % of quarter-hours curtailed
            - bid_prices: Array of all bid prices (for analysis)
            - cleared: Binary array (1=cleared, 0=curtailed)
        """
        # Extract required columns
        prices = df[price_col].values
        generation = df[generation_col].values

        # Get bid prices from strategy
        logger.info(f"Computing bid prices for: {strategy.name}")
        bid_prices = strategy.get_bid_prices(df)

        # Validate bid prices
        if len(bid_prices) != len(prices):
            raise ValueError(
                f"Bid prices length ({len(bid_prices)}) doesn't match "
                f"prices length ({len(prices)})"
            )

        # Simulate auction clearing
        # Generate if actual price >= bid price, else curtailed
        cleared = (prices >= bid_prices).astype(int)
        curtailed = 1 - cleared

        logger.info(f"  Auction clearing: {cleared.sum():,}/{len(cleared):,} bids accepted "
                   f"({cleared.mean()*100:.1f}%)")

        # Scale generation to asset capacity
        gen_scaled = generation * self.capacity_mw

        # === Revenue Calculations ===

        # 1. Baseline: Never curtail (produce always)
        revenue_baseline = np.sum(gen_scaled * prices * self.hours_per_period)

        # 2. With bid strategy: Only earn revenue when cleared
        revenue_with_bids = np.sum(gen_scaled * prices * cleared * self.hours_per_period)

        # 3. Perfect foresight: Curtail only when price < 0
        optimal_cleared = (prices >= 0).astype(int)
        revenue_perfect = np.sum(gen_scaled * prices * optimal_cleared * self.hours_per_period)

        # === Savings Metrics ===

        actual_savings = revenue_with_bids - revenue_baseline
        max_savings = revenue_perfect - revenue_baseline
        capture_rate = (actual_savings / max_savings * 100) if max_savings != 0 else 0

        # === Bid Price Statistics ===

        bid_stats = {
            'avg_bid_price_eur_mwh': bid_prices.mean(),
            'bid_price_std_eur_mwh': bid_prices.std(),
            'bid_price_min_eur_mwh': bid_prices.min(),
            'bid_price_max_eur_mwh': bid_prices.max(),
            'bid_price_p10': np.percentile(bid_prices, 10),
            'bid_price_p25': np.percentile(bid_prices, 25),
            'bid_price_p50': np.percentile(bid_prices, 50),
            'bid_price_p75': np.percentile(bid_prices, 75),
            'bid_price_p90': np.percentile(bid_prices, 90),
        }

        # === Clearing Statistics ===

        curtailment_rate = curtailed.mean() * 100
        cleared_rate = cleared.mean() * 100

        # How often were we cleared when price was positive?
        positive_mask = prices >= 0
        cleared_when_positive = cleared[positive_mask].mean() * 100 if positive_mask.sum() > 0 else 0

        # How often were we curtailed when price was negative?
        negative_mask = prices < 0
        curtailed_when_negative = curtailed[negative_mask].mean() * 100 if negative_mask.sum() > 0 else 0

        # === Cost Breakdown ===

        # Avoided losses: When we were curtailed and price was negative
        avoided_losses_mask = (curtailed == 1) & (prices < 0)
        avoided_losses = np.sum(gen_scaled[avoided_losses_mask] *
                               np.abs(prices[avoided_losses_mask]) *
                               self.hours_per_period)

        # Captured revenue: When we were cleared and price was positive
        captured_revenue_mask = (cleared == 1) & (prices >= 0)
        captured_revenue = np.sum(gen_scaled[captured_revenue_mask] *
                                  prices[captured_revenue_mask] *
                                  self.hours_per_period)

        # Missed revenue: When we were curtailed but price was positive
        missed_revenue_mask = (curtailed == 1) & (prices >= 0)
        missed_revenue = np.sum(gen_scaled[missed_revenue_mask] *
                               prices[missed_revenue_mask] *
                               self.hours_per_period)

        # Paid to generate: When we were cleared but price was negative
        paid_to_generate_mask = (cleared == 1) & (prices < 0)
        paid_to_generate = np.sum(gen_scaled[paid_to_generate_mask] *
                                 np.abs(prices[paid_to_generate_mask]) *
                                 self.hours_per_period)

        # === Compile Results ===

        results = {
            'strategy': strategy.name,
            'revenue_no_curtailment_eur': revenue_baseline,
            'revenue_with_bids_eur': revenue_with_bids,
            'revenue_perfect_eur': revenue_perfect,
            'savings_achieved_eur': actual_savings,
            'max_possible_savings_eur': max_savings,
            'capture_rate_pct': capture_rate,
            'curtailment_rate_pct': curtailment_rate,
            'cleared_rate_pct': cleared_rate,
            'cleared_when_positive_pct': cleared_when_positive,
            'curtailed_when_negative_pct': curtailed_when_negative,
            'total_generation_mwh': np.sum(gen_scaled * self.hours_per_period),
            'quarters_analyzed': len(prices),
            'negative_price_quarters': int((prices < 0).sum()),
            'value_avoided_losses_eur': avoided_losses,
            'value_captured_revenue_eur': captured_revenue,
            'cost_missed_revenue_eur': missed_revenue,
            'cost_paid_to_generate_eur': paid_to_generate,
            **bid_stats,
        }

        # Store detailed arrays for analysis
        results['bid_prices'] = bid_prices
        results['cleared'] = cleared
        results['prices'] = prices

        self.results[strategy.name] = results

        logger.info(f"  Capture rate: {capture_rate:.1f}%")
        logger.info(f"  Avg bid: {bid_stats['avg_bid_price_eur_mwh']:.1f} EUR/MWh")
        logger.info(f"  Curtailment rate: {curtailment_rate:.1f}%")

        return results

    def run_multiple(
        self,
        df: pd.DataFrame,
        strategies: List[BidCurveStrategy],
        price_col: str = 'price_eur_mwh',
        generation_col: str = 'solar_generation_mw'
    ) -> pd.DataFrame:
        """
        Backtest multiple strategies and return comparison DataFrame.

        Args:
            df: Test data
            strategies: List of BidCurveStrategy instances
            price_col: Price column name
            generation_col: Generation column name

        Returns:
            DataFrame with one row per strategy, sorted by capture rate
        """
        results = []

        for strategy in strategies:
            logger.info(f"\nBacktesting: {strategy.name}")
            try:
                result = self.run(df, strategy, price_col, generation_col)
                # Remove array columns for DataFrame
                result_summary = {k: v for k, v in result.items()
                                if not isinstance(v, np.ndarray)}
                results.append(result_summary)
            except Exception as e:
                logger.error(f"Error running {strategy.name}: {e}")
                continue

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)

        # Sort by capture rate (descending)
        if 'capture_rate_pct' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('capture_rate_pct', ascending=False)

        return comparison_df

    def generate_summary_report(self, comparison_df: pd.DataFrame) -> str:
        """
        Generate text summary report of backtest results.

        Args:
            comparison_df: Results from run_multiple()

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("BID CURVE STRATEGY BACKTEST RESULTS")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Asset Capacity: {self.capacity_mw} MW")
        report.append(f"Period Duration: {self.hours_per_period} hours (quarter-hourly)")
        report.append("")

        # Baseline and theoretical maximum
        if len(comparison_df) > 0:
            baseline_revenue = comparison_df.iloc[0]['revenue_no_curtailment_eur']
            max_savings = comparison_df.iloc[0]['max_possible_savings_eur']
            perfect_revenue = comparison_df.iloc[0]['revenue_perfect_eur']

            report.append(f"Baseline Revenue (No Curtailment): {baseline_revenue:>15,.2f} EUR")
            report.append(f"Perfect Foresight Revenue:         {perfect_revenue:>15,.2f} EUR")
            report.append(f"Maximum Possible Savings:          {max_savings:>15,.2f} EUR")
            report.append("")

        report.append("-" * 80)
        report.append("STRATEGY COMPARISON (sorted by Capture Rate)")
        report.append("-" * 80)
        report.append("")

        for _, row in comparison_df.iterrows():
            report.append(f"{row['strategy']}")
            report.append(f"  Revenue:          {row['revenue_with_bids_eur']:>14,.2f} EUR")
            report.append(f"  Savings:          {row['savings_achieved_eur']:>14,.2f} EUR")
            report.append(f"  Capture Rate:     {row['capture_rate_pct']:>13.1f}%")
            report.append(f"  Avg Bid Price:    {row['avg_bid_price_eur_mwh']:>13.2f} EUR/MWh")
            report.append(f"  Bid Std Dev:      {row['bid_price_std_eur_mwh']:>13.2f} EUR/MWh")
            report.append(f"  Curtailment Rate: {row['curtailment_rate_pct']:>13.1f}%")
            report.append("")

        # Best strategy breakdown
        report.append("-" * 80)
        report.append("VALUE BREAKDOWN (Best Strategy)")
        report.append("-" * 80)

        if len(comparison_df) > 0:
            best = comparison_df.iloc[0]
            report.append(f"Strategy: {best['strategy']}")
            report.append("")
            report.append(f"  Avoided Losses:       {best['value_avoided_losses_eur']:>14,.2f} EUR  (curtailed when negative)")
            report.append(f"  Captured Revenue:     {best['value_captured_revenue_eur']:>14,.2f} EUR  (cleared when positive)")
            report.append(f"  Missed Revenue:       {best['cost_missed_revenue_eur']:>14,.2f} EUR  (curtailed when positive)")
            report.append(f"  Paid to Generate:     {best['cost_paid_to_generate_eur']:>14,.2f} EUR  (cleared when negative)")
            report.append("")
            report.append(f"  Net Value:            {best['savings_achieved_eur']:>14,.2f} EUR")

        # Adaptive vs Fixed Floor Comparison
        report.append("")
        report.append("-" * 80)
        report.append("ADAPTIVE vs FIXED FLOOR STRATEGIES")
        report.append("-" * 80)

        # Find best adaptive and best fixed floor
        adaptive_mask = comparison_df['strategy'].str.contains('Expected|Median|Quantile', case=False, regex=True)
        fixed_mask = comparison_df['strategy'].str.contains('Fixed Floor', case=False)

        if adaptive_mask.any() and fixed_mask.any():
            best_adaptive = comparison_df[adaptive_mask].iloc[0]
            best_fixed = comparison_df[fixed_mask].iloc[0]

            report.append(f"\nBest Adaptive Strategy: {best_adaptive['strategy']}")
            report.append(f"  Capture Rate: {best_adaptive['capture_rate_pct']:.1f}%")
            report.append(f"  Avg Bid: {best_adaptive['avg_bid_price_eur_mwh']:.2f} EUR/MWh")

            report.append(f"\nBest Fixed Floor Strategy: {best_fixed['strategy']}")
            report.append(f"  Capture Rate: {best_fixed['capture_rate_pct']:.1f}%")
            report.append(f"  Avg Bid: {best_fixed['avg_bid_price_eur_mwh']:.2f} EUR/MWh")

            improvement = best_adaptive['capture_rate_pct'] - best_fixed['capture_rate_pct']
            report.append(f"\nAdaptive Advantage: +{improvement:.1f}pp capture rate")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def get_strategy_details(self, strategy_name: str) -> Optional[Dict]:
        """Get detailed results for a specific strategy."""
        return self.results.get(strategy_name)


def run_bid_strategy_backtest(
    test_data_path: Path,
    strategies: List[BidCurveStrategy],
    capacity_mw: float = 10.0,
    hours_per_period: float = 0.25,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convenience function to run bid strategy backtest.

    Args:
        test_data_path: Path to test data CSV
        strategies: List of BidCurveStrategy instances
        capacity_mw: Asset capacity in MW
        hours_per_period: Period duration (0.25 for quarter-hourly)
        output_dir: Directory to save results (optional)

    Returns:
        Comparison DataFrame
    """
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)

    # Create backtester
    backtester = AuctionBacktester(capacity_mw=capacity_mw, hours_per_period=hours_per_period)

    # Run backtest
    logger.info("Running bid strategy backtest...")
    results = backtester.run_multiple(df, strategies)

    # Generate report
    report = backtester.generate_summary_report(results)
    print(report)

    # Save results if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison CSV
        csv_path = output_dir / "bid_strategy_comparison.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        # Save report
        report_path = output_dir / "bid_backtest_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")

    return results
