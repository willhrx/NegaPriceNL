"""
Bid Builder for BESS Arbitrage

Converts LP schedule + quantile forecasts into auction bid prices.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .optimiser import DailySchedule


@dataclass
class DailyBids:
    """
    Daily bid structure for wind + BESS portfolio.

    Attributes
    ----------
    wind_sell_volume : np.ndarray
        Wind generation to sell each MTU (MW), shape (96,)
    wind_sell_price : np.ndarray
        Wind sell price each MTU (EUR/MWh), typically at price floor
    bess_charge_volume : np.ndarray
        BESS charge volume each MTU (MW), 0 where no charge planned
    bess_charge_price : np.ndarray
        BESS charge bid price (EUR/MWh), buy if clearing ≤ this
    bess_discharge_volume : np.ndarray
        BESS discharge volume each MTU (MW)
    bess_discharge_price : np.ndarray
        BESS discharge ask price (EUR/MWh), sell if clearing ≥ this
    """

    wind_sell_volume: np.ndarray
    wind_sell_price: np.ndarray
    bess_charge_volume: np.ndarray
    bess_charge_price: np.ndarray
    bess_discharge_volume: np.ndarray
    bess_discharge_price: np.ndarray

    @property
    def n_mtus(self) -> int:
        """Number of market time units."""
        return len(self.wind_sell_volume)


class BidBuilder:
    """
    Converts LP schedule and quantile forecasts into auction bids.

    Strategy:
    - Wind: Always sell at price floor (never curtail)
    - BESS charge: Bid at q25 forecast (buy if price ≤ q25)
    - BESS discharge: Ask at q75 forecast (sell if price ≥ q75)

    Optional aggressive negative price strategy:
    - When RES penetration > threshold, charge at price floor
      to capture negative price charging opportunities.

    Parameters
    ----------
    quantiles : List[float]
        List of quantile levels (e.g., [0.05, 0.10, ..., 0.95])
    charge_quantile : float
        Quantile for charge bid (default 0.25)
    discharge_quantile : float
        Quantile for discharge ask (default 0.75)
    price_floor : float
        Market price floor (EUR/MWh)
    price_cap : float
        Market price cap (EUR/MWh)
    aggressive_res_threshold : Optional[float]
        RES penetration threshold for aggressive charging (e.g., 0.8)
    """

    def __init__(
        self,
        quantiles: List[float],
        charge_quantile: float = 0.25,
        discharge_quantile: float = 0.75,
        price_floor: float = -500.0,
        price_cap: float = 4000.0,
        aggressive_res_threshold: Optional[float] = None,
    ):
        self.quantiles = quantiles
        self.charge_quantile = charge_quantile
        self.discharge_quantile = discharge_quantile
        self.price_floor = price_floor
        self.price_cap = price_cap
        self.aggressive_res_threshold = aggressive_res_threshold

        # Find quantile indices
        self.charge_q_idx = self._find_quantile_index(charge_quantile)
        self.discharge_q_idx = self._find_quantile_index(discharge_quantile)

    def _find_quantile_index(self, target_q: float) -> int:
        """Find index of target quantile in quantiles list."""
        for i, q in enumerate(self.quantiles):
            if abs(q - target_q) < 1e-6:
                return i
        # If exact match not found, find closest
        idx = int(np.argmin([abs(q - target_q) for q in self.quantiles]))
        return idx

    def build_bids(
        self,
        schedule: DailySchedule,
        quantile_forecasts: np.ndarray,
        wind_forecast: np.ndarray,
        res_penetration: Optional[np.ndarray] = None,
    ) -> DailyBids:
        """
        Build auction bids from LP schedule and quantile forecasts.

        Parameters
        ----------
        schedule : DailySchedule
            Optimal charge/discharge schedule from LP
        quantile_forecasts : np.ndarray
            Quantile price forecasts, shape (96, n_quantiles)
        wind_forecast : np.ndarray
            Wind generation forecast (MW), shape (96,)
        res_penetration : Optional[np.ndarray]
            RES penetration forecast, shape (96,), for aggressive strategy

        Returns
        -------
        DailyBids
            Complete bid structure for the day
        """
        n_mtus = len(schedule.charge_mw)

        # Wind: always sell at price floor (never curtail)
        wind_sell_volume = wind_forecast.copy()
        wind_sell_price = np.full(n_mtus, self.price_floor)

        # BESS charge bids
        bess_charge_volume = schedule.charge_mw.copy()
        bess_charge_price = np.zeros(n_mtus)

        # BESS discharge asks
        bess_discharge_volume = schedule.discharge_mw.copy()
        bess_discharge_price = np.zeros(n_mtus)

        for t in range(n_mtus):
            # Charge bid price
            if bess_charge_volume[t] > 0:
                base_charge_price = quantile_forecasts[t, self.charge_q_idx]

                # Aggressive strategy: charge at floor during high RES
                if (
                    self.aggressive_res_threshold is not None
                    and res_penetration is not None
                    and res_penetration[t] > self.aggressive_res_threshold
                ):
                    bess_charge_price[t] = self.price_floor
                else:
                    bess_charge_price[t] = base_charge_price
            else:
                # No charge planned - set price to floor (won't be used)
                bess_charge_price[t] = self.price_floor

            # Discharge ask price
            if bess_discharge_volume[t] > 0:
                bess_discharge_price[t] = quantile_forecasts[
                    t, self.discharge_q_idx
                ]
            else:
                # No discharge planned - set price to cap (won't be used)
                bess_discharge_price[t] = self.price_cap

        return DailyBids(
            wind_sell_volume=wind_sell_volume,
            wind_sell_price=wind_sell_price,
            bess_charge_volume=bess_charge_volume,
            bess_charge_price=bess_charge_price,
            bess_discharge_volume=bess_discharge_volume,
            bess_discharge_price=bess_discharge_price,
        )

    def build_fixed_floor_bids(
        self,
        charge_floor: float,
        discharge_floor: float,
        schedule: DailySchedule,
        wind_forecast: np.ndarray,
    ) -> DailyBids:
        """
        Build bids with fixed price thresholds (naive strategy).

        Parameters
        ----------
        charge_floor : float
            Fixed charge bid price (EUR/MWh)
        discharge_floor : float
            Fixed discharge ask price (EUR/MWh)
        schedule : DailySchedule
            Charge/discharge schedule
        wind_forecast : np.ndarray
            Wind generation forecast (MW)

        Returns
        -------
        DailyBids
            Bid structure with fixed prices
        """
        n_mtus = len(schedule.charge_mw)

        return DailyBids(
            wind_sell_volume=wind_forecast.copy(),
            wind_sell_price=np.full(n_mtus, self.price_floor),
            bess_charge_volume=schedule.charge_mw.copy(),
            bess_charge_price=np.full(n_mtus, charge_floor),
            bess_discharge_volume=schedule.discharge_mw.copy(),
            bess_discharge_price=np.full(n_mtus, discharge_floor),
        )
