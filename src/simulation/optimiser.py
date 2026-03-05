"""
Daily Dispatch Optimizer for BESS Arbitrage

Solves a linear program to find optimal charge/discharge schedule
given price forecasts and battery constraints.

Uses cvxpy for clean constraint syntax and fast solving.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


@dataclass
class DailySchedule:
    """
    Optimal daily charge/discharge schedule from LP optimizer.

    Attributes
    ----------
    charge_mw : np.ndarray
        Charge power for each MTU (96,)
    discharge_mw : np.ndarray
        Discharge power for each MTU (96,)
    planned_soc : np.ndarray
        Planned SoC trajectory (97,) - includes initial SoC
    expected_revenue : float
        Expected revenue from objective function (EUR)
    solver_status : str
        Solver status ('optimal', 'infeasible', etc.)
    """

    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    planned_soc: np.ndarray
    expected_revenue: float
    solver_status: str

    @property
    def net_position(self) -> np.ndarray:
        """Net grid position: discharge - charge (MW)."""
        return self.discharge_mw - self.charge_mw

    @property
    def is_optimal(self) -> bool:
        """Whether solver found optimal solution."""
        return self.solver_status == 'optimal'


class DailyDispatchOptimiser:
    """
    LP-based daily dispatch optimizer for battery storage.

    Formulation:
        max  Σ_t price(t) × [d(t) - c(t)] × duration
        s.t. c(t) ≤ power_mw
             d(t) ≤ power_mw
             SoC(t) = SoC(t-1) + η_c×c(t)×duration - (1/η_d)×d(t)×duration
             SoC_min ≤ SoC(t) ≤ SoC_max
             Σ_t d(t) × duration / energy_mwh ≤ max_cycles  (cycle constraint)

    Parameters
    ----------
    power_mw : float
        Maximum charge/discharge power (MW)
    energy_mwh : float
        Battery energy capacity (MWh)
    soc_min_mwh : float
        Minimum SoC (MWh)
    soc_max_mwh : float
        Maximum SoC (MWh)
    charge_eff : float
        Charge efficiency (e.g., 0.922)
    discharge_eff : float
        Discharge efficiency (e.g., 0.922)
    max_daily_cycles : float
        Maximum daily cycles (e.g., 1.5)
    mtu_duration_h : float
        Duration of each market time unit (hours)
    n_mtus : int
        Number of MTUs per day (default 96)
    """

    def __init__(
        self,
        power_mw: float,
        energy_mwh: float,
        soc_min_mwh: float,
        soc_max_mwh: float,
        charge_eff: float = 0.922,
        discharge_eff: float = 0.922,
        max_daily_cycles: float = 1.5,
        mtu_duration_h: float = 0.25,
        n_mtus: int = 96,
    ):
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required for the LP optimizer. "
                "Install with: pip install cvxpy"
            )

        self.power_mw = power_mw
        self.energy_mwh = energy_mwh
        self.soc_min_mwh = soc_min_mwh
        self.soc_max_mwh = soc_max_mwh
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.max_daily_cycles = max_daily_cycles
        self.mtu_duration_h = mtu_duration_h
        self.n_mtus = n_mtus

    def optimise(
        self,
        price_forecast: np.ndarray,
        initial_soc_mwh: float,
        verbose: bool = False,
    ) -> DailySchedule:
        """
        Solve LP for optimal daily charge/discharge schedule.

        Parameters
        ----------
        price_forecast : np.ndarray
            Price forecast for each MTU (n_mtus,), in EUR/MWh
        initial_soc_mwh : float
            Starting SoC for the day (MWh)
        verbose : bool
            Whether to print solver output

        Returns
        -------
        DailySchedule
            Optimal schedule with charge/discharge arrays
        """
        n = self.n_mtus
        dt = self.mtu_duration_h

        # Validate inputs
        if len(price_forecast) != n:
            raise ValueError(f"Expected {n} prices, got {len(price_forecast)}")

        # Clamp initial SoC to valid range
        initial_soc_mwh = np.clip(
            initial_soc_mwh, self.soc_min_mwh, self.soc_max_mwh
        )

        # Decision variables
        c = cp.Variable(n, nonneg=True)  # charge power (MW)
        d = cp.Variable(n, nonneg=True)  # discharge power (MW)

        # SoC variables (we track SoC at end of each period)
        soc = cp.Variable(n, nonneg=True)

        # Objective: maximize revenue from price arbitrage
        # Revenue = Σ_t price(t) × (discharge(t) - charge(t)) × duration
        # Note: charging at negative prices = earning money
        revenue = cp.sum(cp.multiply(price_forecast, (d - c))) * dt
        objective = cp.Maximize(revenue)

        # Constraints
        constraints = []

        # Power limits
        constraints.append(c <= self.power_mw)
        constraints.append(d <= self.power_mw)

        # SoC evolution: soc[t] = soc[t-1] + η_c*c[t]*dt - (1/η_d)*d[t]*dt
        for t in range(n):
            if t == 0:
                prev_soc = initial_soc_mwh
            else:
                prev_soc = soc[t - 1]

            energy_in = self.charge_eff * c[t] * dt
            energy_out = (1 / self.discharge_eff) * d[t] * dt
            constraints.append(soc[t] == prev_soc + energy_in - energy_out)

        # SoC bounds
        constraints.append(soc >= self.soc_min_mwh)
        constraints.append(soc <= self.soc_max_mwh)

        # Cycle constraint: total discharge ≤ max_cycles × capacity
        max_discharge_mwh = self.max_daily_cycles * self.energy_mwh
        constraints.append(cp.sum(d) * dt <= max_discharge_mwh)

        # No simultaneous charge and discharge (optional, helps convergence)
        # This is implicitly handled by the optimizer seeking to maximize revenue

        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, verbose=verbose)
        except cp.SolverError:
            # Fallback to SCS if ECOS fails
            try:
                problem.solve(solver=cp.SCS, verbose=verbose)
            except cp.SolverError:
                logger.warning("Both ECOS and SCS solvers failed")
                return self._empty_schedule(initial_soc_mwh, "solver_error")

        # Check solution status
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            logger.warning(f"LP solver status: {problem.status}")
            return self._empty_schedule(initial_soc_mwh, problem.status)

        # Extract solution
        charge_mw = np.array(c.value).flatten()
        discharge_mw = np.array(d.value).flatten()
        soc_trajectory = np.array(soc.value).flatten()

        # Clean up small numerical noise
        charge_mw = np.maximum(charge_mw, 0)
        discharge_mw = np.maximum(discharge_mw, 0)

        # Small values are numerical noise
        charge_mw[charge_mw < 1e-6] = 0
        discharge_mw[discharge_mw < 1e-6] = 0

        # Build full SoC trajectory (including initial)
        planned_soc = np.zeros(n + 1)
        planned_soc[0] = initial_soc_mwh
        planned_soc[1:] = soc_trajectory

        return DailySchedule(
            charge_mw=charge_mw,
            discharge_mw=discharge_mw,
            planned_soc=planned_soc,
            expected_revenue=float(problem.value),
            solver_status=problem.status,
        )

    def _empty_schedule(
        self, initial_soc_mwh: float, status: str
    ) -> DailySchedule:
        """Return an empty schedule when optimization fails."""
        n = self.n_mtus
        return DailySchedule(
            charge_mw=np.zeros(n),
            discharge_mw=np.zeros(n),
            planned_soc=np.full(n + 1, initial_soc_mwh),
            expected_revenue=0.0,
            solver_status=status,
        )

    def optimise_with_actual_prices(
        self,
        actual_prices: np.ndarray,
        initial_soc_mwh: float,
    ) -> DailySchedule:
        """
        Solve LP with actual prices (perfect foresight benchmark).

        This provides the theoretical upper bound on achievable revenue.
        """
        return self.optimise(actual_prices, initial_soc_mwh)
