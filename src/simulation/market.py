"""
Auction Clearing Simulator for BESS Arbitrage

Simulates day-ahead auction clearing given bids and actual prices.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
import pandas as pd

from .assets import BatteryStorage
from .bid_builder import DailyBids


@dataclass
class DailyOutcome:
    """
    Results from simulating one day of auction clearing.

    Attributes
    ----------
    date : pd.Timestamp
        Date of the trading day
    wind_revenue_eur : float
        Revenue from wind generation
    bess_charge_cost_eur : float
        Cost of charging (positive = expense)
    bess_discharge_revenue_eur : float
        Revenue from discharging
    bess_net_pnl_eur : float
        Net BESS P&L (discharge revenue - charge cost)
    total_portfolio_revenue_eur : float
        Total portfolio revenue (wind + BESS net)
    soc_timeseries : np.ndarray
        SoC at end of each MTU (97 values including start)
    charge_cleared : np.ndarray
        Boolean mask of MTUs where charge cleared (96,)
    discharge_cleared : np.ndarray
        Boolean mask of MTUs where discharge cleared (96,)
    unfulfilled_mwh : float
        Volume that cleared but couldn't execute due to SoC limits
    n_charge_mtus : int
        Number of MTUs with successful charging
    n_discharge_mtus : int
        Number of MTUs with successful discharging
    energy_charged_mwh : float
        Total energy charged (stored in battery)
    energy_discharged_mwh : float
        Total energy discharged (delivered to grid)
    """

    date: pd.Timestamp
    wind_revenue_eur: float
    bess_charge_cost_eur: float
    bess_discharge_revenue_eur: float
    bess_net_pnl_eur: float
    total_portfolio_revenue_eur: float
    soc_timeseries: np.ndarray
    charge_cleared: np.ndarray
    discharge_cleared: np.ndarray
    unfulfilled_mwh: float
    n_charge_mtus: int
    n_discharge_mtus: int
    energy_charged_mwh: float
    energy_discharged_mwh: float
    actual_charge_mw: np.ndarray  # Actual charge power per MTU (96,)
    actual_discharge_mw: np.ndarray  # Actual discharge power per MTU (96,)

    @property
    def cycles(self) -> float:
        """Number of equivalent full cycles."""
        # Cycles based on discharge (one cycle = full discharge)
        # Note: This is a simplification; real cycle counting is more complex
        return self.energy_discharged_mwh / 50.0  # Assuming 50 MWh battery


class AuctionSimulator:
    """
    Simulates auction clearing for wind + BESS portfolio.

    Clearing logic:
    - Wind: Always clears (bid at price floor)
    - BESS charge: Clears if actual_price ≤ charge_bid_price
    - BESS discharge: Clears if actual_price ≥ discharge_ask_price

    Sequential clearing matters for SoC tracking.

    Parameters
    ----------
    mtu_duration_h : float
        Duration of each MTU (hours), default 0.25
    """

    def __init__(self, mtu_duration_h: float = 0.25):
        self.mtu_duration_h = mtu_duration_h

    def clear(
        self,
        bids: DailyBids,
        actual_prices: np.ndarray,
        actual_wind: np.ndarray,
        battery: BatteryStorage,
        date: pd.Timestamp,
    ) -> DailyOutcome:
        """
        Simulate auction clearing for one day.

        Parameters
        ----------
        bids : DailyBids
            Bid structure from BidBuilder
        actual_prices : np.ndarray
            Actual clearing prices (EUR/MWh), shape (96,)
        actual_wind : np.ndarray
            Actual wind generation (MW), shape (96,)
        battery : BatteryStorage
            Battery object (stateful - will be modified)
        date : pd.Timestamp
            Date for this clearing

        Returns
        -------
        DailyOutcome
            Results of the day's clearing
        """
        n_mtus = bids.n_mtus
        dt = self.mtu_duration_h

        # Track outcomes
        wind_revenue = 0.0
        bess_charge_cost = 0.0
        bess_discharge_revenue = 0.0
        unfulfilled_mwh = 0.0
        energy_charged = 0.0
        energy_discharged = 0.0

        charge_cleared = np.zeros(n_mtus, dtype=bool)
        discharge_cleared = np.zeros(n_mtus, dtype=bool)
        actual_charge_mw = np.zeros(n_mtus)
        actual_discharge_mw = np.zeros(n_mtus)

        # Track SoC over the day
        soc_history = [battery.soc_mwh]

        # Process each MTU sequentially (order matters for SoC)
        for t in range(n_mtus):
            price = actual_prices[t]

            # --- Wind ---
            # Wind always clears (bid at price floor)
            wind_mwh = actual_wind[t] * dt
            wind_revenue += wind_mwh * price

            # --- BESS Charge ---
            # Clears if actual price ≤ charge bid price
            if (
                bids.bess_charge_volume[t] > 0
                and price <= bids.bess_charge_price[t]
            ):
                charge_cleared[t] = True
                requested_power = bids.bess_charge_volume[t]

                # Attempt to charge
                actual_stored = battery.charge(requested_power, dt)
                energy_charged += actual_stored
                actual_charge_mw[t] = actual_stored / dt  # MWh → MW

                # Cost = energy drawn from grid × price
                # Energy from grid = stored / charge_efficiency
                energy_from_grid = actual_stored / battery.charge_eff
                bess_charge_cost += energy_from_grid * price

                # Track unfulfilled if SoC limited the charge
                expected_stored = (
                    requested_power * dt * battery.charge_eff
                )
                if actual_stored < expected_stored * 0.99:
                    unfulfilled_mwh += expected_stored - actual_stored

            # --- BESS Discharge ---
            # Clears if actual price ≥ discharge ask price
            if (
                bids.bess_discharge_volume[t] > 0
                and price >= bids.bess_discharge_price[t]
            ):
                discharge_cleared[t] = True
                requested_power = bids.bess_discharge_volume[t]

                # Attempt to discharge
                actual_delivered = battery.discharge(requested_power, dt)
                energy_discharged += actual_delivered
                actual_discharge_mw[t] = actual_delivered / dt  # MWh → MW

                # Revenue = energy delivered to grid × price
                bess_discharge_revenue += actual_delivered * price

                # Track unfulfilled if SoC limited the discharge
                expected_delivered = requested_power * dt
                if actual_delivered < expected_delivered * 0.99:
                    unfulfilled_mwh += expected_delivered - actual_delivered

            # Record SoC at end of this MTU
            soc_history.append(battery.soc_mwh)

        # Calculate net P&L
        bess_net_pnl = bess_discharge_revenue - bess_charge_cost
        total_portfolio_revenue = wind_revenue + bess_net_pnl

        return DailyOutcome(
            date=date,
            wind_revenue_eur=wind_revenue,
            bess_charge_cost_eur=bess_charge_cost,
            bess_discharge_revenue_eur=bess_discharge_revenue,
            bess_net_pnl_eur=bess_net_pnl,
            total_portfolio_revenue_eur=total_portfolio_revenue,
            soc_timeseries=np.array(soc_history),
            charge_cleared=charge_cleared,
            discharge_cleared=discharge_cleared,
            unfulfilled_mwh=unfulfilled_mwh,
            n_charge_mtus=int(charge_cleared.sum()),
            n_discharge_mtus=int(discharge_cleared.sum()),
            energy_charged_mwh=energy_charged,
            energy_discharged_mwh=energy_discharged,
            actual_charge_mw=actual_charge_mw,
            actual_discharge_mw=actual_discharge_mw,
        )

    def clear_wind_only(
        self,
        actual_prices: np.ndarray,
        actual_wind: np.ndarray,
        date: pd.Timestamp,
    ) -> DailyOutcome:
        """
        Baseline scenario: wind only, no BESS.

        Parameters
        ----------
        actual_prices : np.ndarray
            Actual clearing prices (EUR/MWh)
        actual_wind : np.ndarray
            Actual wind generation (MW)
        date : pd.Timestamp
            Date for this clearing

        Returns
        -------
        DailyOutcome
            Results with zero BESS activity
        """
        n_mtus = len(actual_prices)
        dt = self.mtu_duration_h

        # Wind revenue only
        wind_revenue = np.sum(actual_wind * actual_prices * dt)

        return DailyOutcome(
            date=date,
            wind_revenue_eur=wind_revenue,
            bess_charge_cost_eur=0.0,
            bess_discharge_revenue_eur=0.0,
            bess_net_pnl_eur=0.0,
            total_portfolio_revenue_eur=wind_revenue,
            soc_timeseries=np.zeros(n_mtus + 1),
            charge_cleared=np.zeros(n_mtus, dtype=bool),
            discharge_cleared=np.zeros(n_mtus, dtype=bool),
            unfulfilled_mwh=0.0,
            n_charge_mtus=0,
            n_discharge_mtus=0,
            energy_charged_mwh=0.0,
            energy_discharged_mwh=0.0,
            actual_charge_mw=np.zeros(n_mtus),
            actual_discharge_mw=np.zeros(n_mtus),
        )
