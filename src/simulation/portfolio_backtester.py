"""
Portfolio Backtester for BESS Arbitrage Simulation

Main simulation loop that ties together all components:
- Quantile model predictions
- LP dispatch optimization
- Bid construction
- Auction clearing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

from .assets import WindFarm, BatteryStorage
from .optimiser import DailyDispatchOptimiser, DailySchedule
from .bid_builder import BidBuilder, DailyBids
from .market import AuctionSimulator, DailyOutcome

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for portfolio simulation."""

    # Asset parameters
    wind_capacity_mw: float = 50.0
    wind_national_capacity_mw: float = 11700.0
    bess_power_mw: float = 25.0
    bess_energy_mwh: float = 50.0
    bess_soc_min_pct: float = 0.10
    bess_soc_max_pct: float = 0.90
    bess_initial_soc_mwh: float = 25.0
    bess_charge_eff: float = 0.922
    bess_discharge_eff: float = 0.922
    bess_max_daily_cycles: float = 1.5

    # Bidding strategy
    charge_quantile: float = 0.25
    discharge_quantile: float = 0.75
    price_floor: float = -500.0
    price_cap: float = 4000.0
    aggressive_res_threshold: Optional[float] = 0.80

    # Market
    mtu_duration_h: float = 0.25
    n_mtus: int = 96


class PortfolioBacktester:
    """
    Main simulation loop for BESS arbitrage backtesting.

    Runs day-by-day simulation:
    1. Generate quantile forecasts
    2. Optimize dispatch using median forecast
    3. Build bids using quantile thresholds
    4. Clear against actual prices
    5. Carry SoC forward to next day

    Parameters
    ----------
    model : object
        Trained quantile regression model with predict() method
    quantiles : List[float]
        Quantile levels used by the model
    config : SimulationConfig
        Simulation configuration
    """

    def __init__(
        self,
        model,
        quantiles: List[float],
        config: SimulationConfig,
    ):
        self.model = model
        self.quantiles = quantiles
        self.config = config

        # Find median quantile index
        self.median_idx = self._find_quantile_index(0.50)

        # Initialize components
        self.wind_farm = WindFarm(
            capacity_mw=config.wind_capacity_mw,
            national_capacity_mw=config.wind_national_capacity_mw,
        )

        self.battery = BatteryStorage(
            power_mw=config.bess_power_mw,
            energy_mwh=config.bess_energy_mwh,
            soc_min_pct=config.bess_soc_min_pct,
            soc_max_pct=config.bess_soc_max_pct,
            initial_soc_mwh=config.bess_initial_soc_mwh,
            charge_eff=config.bess_charge_eff,
            discharge_eff=config.bess_discharge_eff,
        )

        self.optimiser = DailyDispatchOptimiser(
            power_mw=config.bess_power_mw,
            energy_mwh=config.bess_energy_mwh,
            soc_min_mwh=config.bess_energy_mwh * config.bess_soc_min_pct,
            soc_max_mwh=config.bess_energy_mwh * config.bess_soc_max_pct,
            charge_eff=config.bess_charge_eff,
            discharge_eff=config.bess_discharge_eff,
            max_daily_cycles=config.bess_max_daily_cycles,
            mtu_duration_h=config.mtu_duration_h,
            n_mtus=config.n_mtus,
        )

        self.bid_builder = BidBuilder(
            quantiles=quantiles,
            charge_quantile=config.charge_quantile,
            discharge_quantile=config.discharge_quantile,
            price_floor=config.price_floor,
            price_cap=config.price_cap,
            aggressive_res_threshold=config.aggressive_res_threshold,
        )

        self.market = AuctionSimulator(mtu_duration_h=config.mtu_duration_h)

    def _find_quantile_index(self, target_q: float) -> int:
        """Find index of target quantile."""
        for i, q in enumerate(self.quantiles):
            if abs(q - target_q) < 1e-6:
                return i
        return int(np.argmin([abs(q - target_q) for q in self.quantiles]))

    def run(
        self,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        price_column: str = 'price_eur_mwh',
        wind_column: str = 'wind_generation_mw',
        res_penetration_column: Optional[str] = 'forecast_res_penetration',
    ) -> List[DailyOutcome]:
        """
        Run full backtest over test period.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data with datetime index, features, prices, wind
        feature_columns : List[str]
            Feature column names for model prediction
        price_column : str
            Column name for actual prices
        wind_column : str
            Column name for national wind generation
        res_penetration_column : Optional[str]
            Column name for RES penetration (for aggressive charging)

        Returns
        -------
        List[DailyOutcome]
            Results for each day
        """
        # Reset battery to initial state
        self.battery.reset(self.config.bess_initial_soc_mwh)

        # Group by date
        test_df = test_df.copy()
        test_df['_date'] = test_df.index.date
        daily_groups = test_df.groupby('_date')

        outcomes: List[DailyOutcome] = []
        n_days = len(daily_groups)

        logger.info(f"Running backtest for {n_days} days...")

        for i, (date, day_df) in enumerate(daily_groups):
            if len(day_df) != self.config.n_mtus:
                logger.warning(
                    f"Day {date} has {len(day_df)} MTUs, expected {self.config.n_mtus}. Skipping."
                )
                continue

            # Extract data for this day
            X = day_df[feature_columns].values
            actual_prices = day_df[price_column].values
            national_wind = day_df[wind_column].values

            # Get RES penetration if available
            res_penetration = None
            if (
                res_penetration_column is not None
                and res_penetration_column in day_df.columns
            ):
                res_penetration = day_df[res_penetration_column].values

            # Scale wind to farm level
            actual_wind = self.wind_farm.get_generation(national_wind)
            wind_forecast = actual_wind  # Perfect wind foresight assumption

            # Generate quantile forecasts
            quantile_forecasts = self.model.predict(X, enforce_monotonicity=True)

            # Extract median forecast for LP
            median_forecast = quantile_forecasts[:, self.median_idx]

            # Optimize dispatch
            schedule = self.optimiser.optimise(
                price_forecast=median_forecast,
                initial_soc_mwh=self.battery.soc_mwh,
            )

            # Build bids
            bids = self.bid_builder.build_bids(
                schedule=schedule,
                quantile_forecasts=quantile_forecasts,
                wind_forecast=wind_forecast,
                res_penetration=res_penetration,
            )

            # Clear against actual prices
            outcome = self.market.clear(
                bids=bids,
                actual_prices=actual_prices,
                actual_wind=actual_wind,
                battery=self.battery,
                date=pd.Timestamp(date),
            )

            outcomes.append(outcome)

            # Log progress
            if (i + 1) % 30 == 0:
                logger.info(
                    f"  Day {i + 1}/{n_days}: BESS P&L = {outcome.bess_net_pnl_eur:,.0f} EUR"
                )

        logger.info(f"Backtest complete: {len(outcomes)} days processed")
        return outcomes

    def run_wind_only(
        self,
        test_df: pd.DataFrame,
        price_column: str = 'price_eur_mwh',
        wind_column: str = 'wind_generation_mw',
    ) -> List[DailyOutcome]:
        """
        Run wind-only baseline (no BESS).

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data
        price_column : str
            Column name for prices
        wind_column : str
            Column name for wind generation

        Returns
        -------
        List[DailyOutcome]
            Results for each day
        """
        test_df = test_df.copy()
        test_df['_date'] = test_df.index.date
        daily_groups = test_df.groupby('_date')

        outcomes: List[DailyOutcome] = []

        for date, day_df in daily_groups:
            if len(day_df) != self.config.n_mtus:
                continue

            actual_prices = day_df[price_column].values
            national_wind = day_df[wind_column].values
            actual_wind = self.wind_farm.get_generation(national_wind)

            outcome = self.market.clear_wind_only(
                actual_prices=actual_prices,
                actual_wind=actual_wind,
                date=pd.Timestamp(date),
            )
            outcomes.append(outcome)

        return outcomes

    def run_perfect_foresight(
        self,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        price_column: str = 'price_eur_mwh',
        wind_column: str = 'wind_generation_mw',
    ) -> List[DailyOutcome]:
        """
        Run with perfect price foresight (upper bound benchmark).

        Uses actual prices for LP optimization instead of forecasts.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data
        feature_columns : List[str]
            Feature columns (not used for forecasting, but needed for structure)
        price_column : str
            Column name for actual prices
        wind_column : str
            Column name for wind generation

        Returns
        -------
        List[DailyOutcome]
            Results for each day
        """
        # Reset battery
        self.battery.reset(self.config.bess_initial_soc_mwh)

        test_df = test_df.copy()
        test_df['_date'] = test_df.index.date
        daily_groups = test_df.groupby('_date')

        outcomes: List[DailyOutcome] = []

        for date, day_df in daily_groups:
            if len(day_df) != self.config.n_mtus:
                continue

            actual_prices = day_df[price_column].values
            national_wind = day_df[wind_column].values
            actual_wind = self.wind_farm.get_generation(national_wind)

            # Optimize with perfect foresight
            schedule = self.optimiser.optimise(
                price_forecast=actual_prices,  # Use actual prices
                initial_soc_mwh=self.battery.soc_mwh,
            )

            # Build bids at actual prices (always clear)
            bids = DailyBids(
                wind_sell_volume=actual_wind,
                wind_sell_price=np.full(self.config.n_mtus, self.config.price_floor),
                bess_charge_volume=schedule.charge_mw,
                bess_charge_price=np.full(self.config.n_mtus, self.config.price_cap),
                bess_discharge_volume=schedule.discharge_mw,
                bess_discharge_price=np.full(self.config.n_mtus, self.config.price_floor),
            )

            # Clear
            outcome = self.market.clear(
                bids=bids,
                actual_prices=actual_prices,
                actual_wind=actual_wind,
                battery=self.battery,
                date=pd.Timestamp(date),
            )
            outcomes.append(outcome)

        return outcomes
