"""
Economic evaluation module for negative price predictions.

This module provides tools to:
1. Calculate economic value of predictions for solar asset operators
2. Compare ML model against benchmark strategies
3. Run backtests on historical data
"""

from src.evaluation.economic_metrics import (
    calculate_economic_value,
    calculate_confusion_costs,
    calculate_capture_rate,
)
from src.evaluation.benchmark_strategies import (
    NaiveStrategy,
    HeuristicStrategy,
    SolarThresholdStrategy,
    MLStrategy,
)
from src.evaluation.backtester import EconomicBacktester

__all__ = [
    'calculate_economic_value',
    'calculate_confusion_costs',
    'calculate_capture_rate',
    'NaiveStrategy',
    'HeuristicStrategy',
    'SolarThresholdStrategy',
    'MLStrategy',
    'EconomicBacktester',
]
