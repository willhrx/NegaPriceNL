"""
BESS Arbitrage Simulation Package

Provides components for simulating a BRP portfolio with wind + battery storage
trading in the Dutch day-ahead electricity market.

Components:
- assets: WindFarm and BatteryStorage physical models
- optimiser: Daily LP dispatch optimization
- bid_builder: Quantile forecast to bid price conversion
- market: Auction clearing simulation
- portfolio_backtester: Main simulation loop
- metrics: P&L aggregation and reporting
"""

from .assets import WindFarm, BatteryStorage
from .optimiser import DailyDispatchOptimiser, DailySchedule
from .bid_builder import BidBuilder, DailyBids
from .market import AuctionSimulator, DailyOutcome
from .portfolio_backtester import PortfolioBacktester
from .metrics import SimulationMetrics

__all__ = [
    'WindFarm',
    'BatteryStorage',
    'DailyDispatchOptimiser',
    'DailySchedule',
    'BidBuilder',
    'DailyBids',
    'AuctionSimulator',
    'DailyOutcome',
    'PortfolioBacktester',
    'SimulationMetrics',
]
