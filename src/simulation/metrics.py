"""
Simulation Metrics and Reporting for BESS Arbitrage

Aggregates daily outcomes into portfolio-level metrics and generates reports.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

from .market import DailyOutcome

logger = logging.getLogger(__name__)


@dataclass
class SimulationMetrics:
    """
    Aggregated metrics from BESS simulation.

    Attributes
    ----------
    total_wind_revenue_eur : float
        Total wind generation revenue
    total_bess_charge_cost_eur : float
        Total cost of charging
    total_bess_discharge_revenue_eur : float
        Total revenue from discharging
    total_bess_net_pnl_eur : float
        Net BESS P&L
    total_portfolio_revenue_eur : float
        Total portfolio revenue
    incremental_bess_value_eur : float
        Value added by BESS vs wind-only
    bess_value_per_mw_year_eur : float
        BESS value per MW of storage per year
    avg_daily_bess_pnl_eur : float
        Average daily BESS P&L
    avg_spread_captured_eur_mwh : float
        Average spread between discharge and charge prices
    total_unfulfilled_mwh : float
        Total unfulfilled volume
    unfulfilled_pct : float
        Unfulfilled as % of total discharge volume
    total_energy_charged_mwh : float
        Total energy charged
    total_energy_discharged_mwh : float
        Total energy discharged
    avg_daily_cycles : float
        Average daily equivalent cycles
    n_days : int
        Number of days in simulation
    n_active_days : int
        Days with non-zero BESS activity
    """

    total_wind_revenue_eur: float
    total_bess_charge_cost_eur: float
    total_bess_discharge_revenue_eur: float
    total_bess_net_pnl_eur: float
    total_portfolio_revenue_eur: float
    incremental_bess_value_eur: float
    bess_value_per_mw_year_eur: float
    avg_daily_bess_pnl_eur: float
    avg_spread_captured_eur_mwh: float
    total_unfulfilled_mwh: float
    unfulfilled_pct: float
    total_energy_charged_mwh: float
    total_energy_discharged_mwh: float
    avg_daily_cycles: float
    n_days: int
    n_active_days: int


def calculate_metrics(
    outcomes: List[DailyOutcome],
    wind_only_outcomes: Optional[List[DailyOutcome]] = None,
    bess_power_mw: float = 25.0,
    bess_energy_mwh: float = 50.0,
) -> SimulationMetrics:
    """
    Calculate aggregated metrics from simulation outcomes.

    Parameters
    ----------
    outcomes : List[DailyOutcome]
        Daily outcomes from BESS strategy
    wind_only_outcomes : Optional[List[DailyOutcome]]
        Wind-only baseline outcomes for comparison
    bess_power_mw : float
        BESS power capacity for per-MW calculations
    bess_energy_mwh : float
        BESS energy capacity for cycle calculations

    Returns
    -------
    SimulationMetrics
        Aggregated metrics
    """
    n_days = len(outcomes)

    # Aggregate totals
    total_wind_revenue = sum(o.wind_revenue_eur for o in outcomes)
    total_charge_cost = sum(o.bess_charge_cost_eur for o in outcomes)
    total_discharge_revenue = sum(o.bess_discharge_revenue_eur for o in outcomes)
    total_bess_pnl = sum(o.bess_net_pnl_eur for o in outcomes)
    total_portfolio = sum(o.total_portfolio_revenue_eur for o in outcomes)
    total_unfulfilled = sum(o.unfulfilled_mwh for o in outcomes)
    total_charged = sum(o.energy_charged_mwh for o in outcomes)
    total_discharged = sum(o.energy_discharged_mwh for o in outcomes)

    # Count active days
    n_active = sum(
        1 for o in outcomes
        if o.n_charge_mtus > 0 or o.n_discharge_mtus > 0
    )

    # Calculate incremental value vs wind-only
    if wind_only_outcomes is not None:
        wind_only_revenue = sum(o.total_portfolio_revenue_eur for o in wind_only_outcomes)
        incremental_value = total_portfolio - wind_only_revenue
    else:
        incremental_value = total_bess_pnl

    # Per-MW per-year value
    bess_value_per_mw = incremental_value / bess_power_mw

    # Average daily metrics
    avg_daily_pnl = total_bess_pnl / n_days if n_days > 0 else 0

    # Average spread captured
    if total_discharged > 0 and total_charged > 0:
        avg_discharge_price = total_discharge_revenue / total_discharged
        avg_charge_price = total_charge_cost / total_charged
        avg_spread = avg_discharge_price - avg_charge_price
    else:
        avg_spread = 0

    # Unfulfilled percentage
    total_planned_discharge = total_discharged + total_unfulfilled
    unfulfilled_pct = (
        (total_unfulfilled / total_planned_discharge * 100)
        if total_planned_discharge > 0
        else 0
    )

    # Average daily cycles
    total_cycles = total_discharged / bess_energy_mwh
    avg_daily_cycles = total_cycles / n_days if n_days > 0 else 0

    return SimulationMetrics(
        total_wind_revenue_eur=total_wind_revenue,
        total_bess_charge_cost_eur=total_charge_cost,
        total_bess_discharge_revenue_eur=total_discharge_revenue,
        total_bess_net_pnl_eur=total_bess_pnl,
        total_portfolio_revenue_eur=total_portfolio,
        incremental_bess_value_eur=incremental_value,
        bess_value_per_mw_year_eur=bess_value_per_mw,
        avg_daily_bess_pnl_eur=avg_daily_pnl,
        avg_spread_captured_eur_mwh=avg_spread,
        total_unfulfilled_mwh=total_unfulfilled,
        unfulfilled_pct=unfulfilled_pct,
        total_energy_charged_mwh=total_charged,
        total_energy_discharged_mwh=total_discharged,
        avg_daily_cycles=avg_daily_cycles,
        n_days=n_days,
        n_active_days=n_active,
    )


def outcomes_to_dataframe(outcomes: List[DailyOutcome]) -> pd.DataFrame:
    """
    Convert list of daily outcomes to DataFrame.

    Parameters
    ----------
    outcomes : List[DailyOutcome]
        Daily outcomes from simulation

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per day
    """
    records = []
    for o in outcomes:
        records.append({
            'date': o.date,
            'wind_revenue_eur': o.wind_revenue_eur,
            'bess_charge_cost_eur': o.bess_charge_cost_eur,
            'bess_discharge_revenue_eur': o.bess_discharge_revenue_eur,
            'bess_net_pnl_eur': o.bess_net_pnl_eur,
            'total_portfolio_revenue_eur': o.total_portfolio_revenue_eur,
            'unfulfilled_mwh': o.unfulfilled_mwh,
            'n_charge_mtus': o.n_charge_mtus,
            'n_discharge_mtus': o.n_discharge_mtus,
            'energy_charged_mwh': o.energy_charged_mwh,
            'energy_discharged_mwh': o.energy_discharged_mwh,
            'cycles': o.energy_discharged_mwh / 50.0,  # Assuming 50 MWh
            'end_soc_mwh': o.soc_timeseries[-1] if len(o.soc_timeseries) > 0 else 0,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index('date')
    return df


def generate_report(
    metrics: SimulationMetrics,
    strategy_name: str = "BESS Conservative (q25/q75)",
) -> str:
    """
    Generate text report for simulation results.

    Parameters
    ----------
    metrics : SimulationMetrics
        Aggregated simulation metrics
    strategy_name : str
        Name of the strategy for the report header

    Returns
    -------
    str
        Formatted report text
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"BESS ARBITRAGE SIMULATION REPORT")
    lines.append(f"Strategy: {strategy_name}")
    lines.append("=" * 80)
    lines.append("")

    lines.append("-" * 80)
    lines.append("PORTFOLIO SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Simulation Period:           {metrics.n_days} days")
    lines.append(f"Days with BESS Activity:     {metrics.n_active_days}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("REVENUE BREAKDOWN")
    lines.append("-" * 80)
    lines.append(f"Wind Revenue:                {metrics.total_wind_revenue_eur:>15,.2f} EUR")
    lines.append(f"BESS Discharge Revenue:      {metrics.total_bess_discharge_revenue_eur:>15,.2f} EUR")
    lines.append(f"BESS Charge Cost:            {metrics.total_bess_charge_cost_eur:>15,.2f} EUR")
    lines.append(f"BESS Net P&L:                {metrics.total_bess_net_pnl_eur:>15,.2f} EUR")
    lines.append(f"                             {'-' * 15}")
    lines.append(f"Total Portfolio Revenue:     {metrics.total_portfolio_revenue_eur:>15,.2f} EUR")
    lines.append("")

    lines.append("-" * 80)
    lines.append("BESS VALUE METRICS")
    lines.append("-" * 80)
    lines.append(f"Incremental BESS Value:      {metrics.incremental_bess_value_eur:>15,.2f} EUR")
    lines.append(f"BESS Value per MW per Year:  {metrics.bess_value_per_mw_year_eur:>15,.2f} EUR/MW/yr")
    lines.append(f"Average Daily BESS P&L:      {metrics.avg_daily_bess_pnl_eur:>15,.2f} EUR/day")
    lines.append(f"Average Spread Captured:     {metrics.avg_spread_captured_eur_mwh:>15,.2f} EUR/MWh")
    lines.append("")

    lines.append("-" * 80)
    lines.append("OPERATIONAL METRICS")
    lines.append("-" * 80)
    lines.append(f"Total Energy Charged:        {metrics.total_energy_charged_mwh:>15,.2f} MWh")
    lines.append(f"Total Energy Discharged:     {metrics.total_energy_discharged_mwh:>15,.2f} MWh")
    lines.append(f"Average Daily Cycles:        {metrics.avg_daily_cycles:>15,.3f}")
    lines.append(f"Total Unfulfilled Volume:    {metrics.total_unfulfilled_mwh:>15,.2f} MWh")
    lines.append(f"Unfulfilled Percentage:      {metrics.unfulfilled_pct:>15,.2f} %")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_comparison_report(
    strategies: Dict[str, SimulationMetrics],
) -> str:
    """
    Generate comparison report for multiple strategies.

    Parameters
    ----------
    strategies : Dict[str, SimulationMetrics]
        Dictionary mapping strategy name to metrics

    Returns
    -------
    str
        Formatted comparison report
    """
    lines = []
    lines.append("=" * 100)
    lines.append("BESS STRATEGY COMPARISON")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = f"{'Strategy':<30} {'Total P&L':>15} {'BESS Value':>15} {'Avg Spread':>12} {'Cycles/Day':>12}"
    lines.append(header)
    lines.append("-" * 100)

    # Sort by total portfolio revenue
    sorted_strategies = sorted(
        strategies.items(),
        key=lambda x: x[1].total_portfolio_revenue_eur,
        reverse=True,
    )

    for name, m in sorted_strategies:
        row = (
            f"{name:<30} "
            f"{m.total_portfolio_revenue_eur:>15,.0f} "
            f"{m.incremental_bess_value_eur:>15,.0f} "
            f"{m.avg_spread_captured_eur_mwh:>12,.2f} "
            f"{m.avg_daily_cycles:>12,.3f}"
        )
        lines.append(row)

    lines.append("-" * 100)
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)
