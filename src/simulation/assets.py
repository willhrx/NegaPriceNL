"""
Physical Asset Models for BESS Simulation

Classes:
- WindFarm: Scales national wind generation to farm-level output
- BatteryStorage: Stateful battery with SoC tracking and efficiency losses
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindFarm:
    """
    Wind farm model that scales national wind generation to farm output.

    Assumes farm generation is proportional to national wind output,
    which is a simplification (real farms have location-specific profiles).

    Parameters
    ----------
    capacity_mw : float
        Nameplate capacity of the wind farm (MW)
    national_capacity_mw : float
        Total installed wind capacity in the country (MW)
    """

    capacity_mw: float
    national_capacity_mw: float

    @property
    def share(self) -> float:
        """Farm's share of national wind capacity."""
        return self.capacity_mw / self.national_capacity_mw

    def get_generation(self, national_wind_mw: np.ndarray) -> np.ndarray:
        """
        Scale national wind generation to farm output.

        Parameters
        ----------
        national_wind_mw : np.ndarray
            National wind generation timeseries (MW)

        Returns
        -------
        np.ndarray
            Farm-level generation timeseries (MW)
        """
        return national_wind_mw * self.share

    def get_forecast(self, national_wind_forecast_mw: np.ndarray) -> np.ndarray:
        """
        Scale national wind forecast to farm-level forecast.

        Parameters
        ----------
        national_wind_forecast_mw : np.ndarray
            National wind forecast timeseries (MW)

        Returns
        -------
        np.ndarray
            Farm-level forecast timeseries (MW)
        """
        return national_wind_forecast_mw * self.share


class BatteryStorage:
    """
    Stateful battery storage model with SoC tracking and efficiency losses.

    The battery enforces:
    - SoC floor and ceiling constraints
    - Charge/discharge power limits
    - Round-trip efficiency (split between charge and discharge)

    Parameters
    ----------
    power_mw : float
        Maximum charge/discharge power (MW)
    energy_mwh : float
        Nameplate energy capacity (MWh)
    soc_min_pct : float
        Minimum SoC as fraction (e.g., 0.10 for 10%)
    soc_max_pct : float
        Maximum SoC as fraction (e.g., 0.90 for 90%)
    initial_soc_mwh : float
        Initial state of charge (MWh)
    charge_eff : float
        One-way charge efficiency (e.g., 0.922 for 85% RTE)
    discharge_eff : float
        One-way discharge efficiency (e.g., 0.922 for 85% RTE)
    """

    def __init__(
        self,
        power_mw: float,
        energy_mwh: float,
        soc_min_pct: float = 0.10,
        soc_max_pct: float = 0.90,
        initial_soc_mwh: Optional[float] = None,
        charge_eff: float = 0.922,
        discharge_eff: float = 0.922,
    ):
        self.power_mw = power_mw
        self.energy_mwh = energy_mwh
        self.soc_min_mwh = energy_mwh * soc_min_pct
        self.soc_max_mwh = energy_mwh * soc_max_pct
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff

        # Initialize SoC
        if initial_soc_mwh is None:
            initial_soc_mwh = energy_mwh * 0.5  # Default to 50%
        self._soc_mwh = np.clip(initial_soc_mwh, self.soc_min_mwh, self.soc_max_mwh)

        # Track history for analysis
        self._soc_history: list[float] = [self._soc_mwh]

    @property
    def soc_mwh(self) -> float:
        """Current state of charge (MWh)."""
        return self._soc_mwh

    @property
    def soc_pct(self) -> float:
        """Current state of charge as percentage."""
        return self._soc_mwh / self.energy_mwh

    @property
    def usable_capacity_mwh(self) -> float:
        """Usable capacity between SoC floor and ceiling."""
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def soc_history(self) -> np.ndarray:
        """History of SoC values."""
        return np.array(self._soc_history)

    def available_charge_mw(self, duration_h: float = 0.25) -> float:
        """
        Maximum charge power available given current SoC headroom.

        Parameters
        ----------
        duration_h : float
            Duration of the charging period (hours)

        Returns
        -------
        float
            Maximum charge power (MW) that won't exceed SoC ceiling
        """
        # How much energy can we store?
        headroom_mwh = self.soc_max_mwh - self._soc_mwh
        # Account for efficiency: grid energy needed = stored / efficiency
        max_energy_from_grid = headroom_mwh / self.charge_eff
        # Convert to power
        max_power = max_energy_from_grid / duration_h
        return min(max_power, self.power_mw)

    def available_discharge_mw(self, duration_h: float = 0.25) -> float:
        """
        Maximum discharge power available given current SoC.

        Parameters
        ----------
        duration_h : float
            Duration of the discharging period (hours)

        Returns
        -------
        float
            Maximum discharge power (MW) that won't go below SoC floor
        """
        # How much energy can we extract?
        available_mwh = self._soc_mwh - self.soc_min_mwh
        # Account for efficiency: delivered = extracted * efficiency
        max_energy_delivered = available_mwh * self.discharge_eff
        # Convert to power
        max_power = max_energy_delivered / duration_h
        return min(max_power, self.power_mw)

    def charge(self, power_mw: float, duration_h: float) -> float:
        """
        Attempt to charge at given power for given duration.

        Parameters
        ----------
        power_mw : float
            Requested charge power (MW)
        duration_h : float
            Duration of charging (hours)

        Returns
        -------
        float
            Actual energy stored (MWh), after efficiency and SoC limits
        """
        # Clamp to available power
        actual_power = min(power_mw, self.available_charge_mw(duration_h))
        actual_power = min(actual_power, self.power_mw)

        if actual_power <= 0:
            return 0.0

        # Energy drawn from grid
        energy_from_grid = actual_power * duration_h

        # Energy stored (after losses)
        energy_stored = energy_from_grid * self.charge_eff

        # Update SoC
        new_soc = self._soc_mwh + energy_stored
        new_soc = np.clip(new_soc, self.soc_min_mwh, self.soc_max_mwh)
        actual_stored = new_soc - self._soc_mwh
        self._soc_mwh = new_soc
        self._soc_history.append(self._soc_mwh)

        return actual_stored

    def discharge(self, power_mw: float, duration_h: float) -> float:
        """
        Attempt to discharge at given power for given duration.

        Parameters
        ----------
        power_mw : float
            Requested discharge power (MW)
        duration_h : float
            Duration of discharging (hours)

        Returns
        -------
        float
            Actual energy delivered to grid (MWh), after efficiency and SoC limits
        """
        # Clamp to available power
        actual_power = min(power_mw, self.available_discharge_mw(duration_h))
        actual_power = min(actual_power, self.power_mw)

        if actual_power <= 0:
            return 0.0

        # Energy delivered to grid
        energy_delivered = actual_power * duration_h

        # Energy extracted from battery (before losses)
        energy_extracted = energy_delivered / self.discharge_eff

        # Update SoC
        new_soc = self._soc_mwh - energy_extracted
        new_soc = np.clip(new_soc, self.soc_min_mwh, self.soc_max_mwh)
        actual_extracted = self._soc_mwh - new_soc
        self._soc_mwh = new_soc
        self._soc_history.append(self._soc_mwh)

        # Actual delivered = extracted * efficiency
        actual_delivered = actual_extracted * self.discharge_eff

        return actual_delivered

    def reset(self, soc_mwh: Optional[float] = None) -> None:
        """
        Reset battery state for new simulation run.

        Parameters
        ----------
        soc_mwh : float, optional
            Initial SoC for reset. If None, resets to 50%.
        """
        if soc_mwh is None:
            soc_mwh = self.energy_mwh * 0.5
        self._soc_mwh = np.clip(soc_mwh, self.soc_min_mwh, self.soc_max_mwh)
        self._soc_history = [self._soc_mwh]

    def __repr__(self) -> str:
        return (
            f"BatteryStorage(power={self.power_mw}MW, "
            f"energy={self.energy_mwh}MWh, "
            f"SoC={self.soc_pct:.1%})"
        )
