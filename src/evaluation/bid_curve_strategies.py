"""
Bid Curve Strategies for Day-Ahead Auction Market

Implements probabilistic bidding strategies that convert quantile forecasts
into optimal auction bids (EUR/MWh price floors).

Unlike binary strategies (curtail/generate), bid curves let the market clearing
price determine the outcome: generate if market_price >= bid_price, else curtailed.

Strategy Variants:
1. ExpectedValueBidStrategy: Bid at E[price] (risk-neutral optimal)
2. MedianBidStrategy: Bid at median (q50, conservative)
3. QuantileBidStrategy: Bid at specific quantile (risk-adjusted)
4. FixedFloorBidStrategy: Fixed price floor (e.g., -€5/MWh)

Example:
    >>> from src.models.quantile_regressor import NegativePriceQuantileRegressor
    >>> model = NegativePriceQuantileRegressor.load('model_v10.pkl')
    >>> strategy = ExpectedValueBidStrategy(model, quantiles=[0.1, 0.5, 0.9])
    >>> bid_prices = strategy.calculate_bid_price(X)  # EUR/MWh bids
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from abc import abstractmethod

from .benchmark_strategies import BaseStrategy


class BidCurveStrategy(BaseStrategy):
    """
    Abstract base class for bid curve strategies.

    Converts quantile price forecasts into optimal auction bids.

    Attributes:
        model: Trained NegativePriceQuantileRegressor
        quantiles: List of quantile levels (e.g., [0.05, 0.10, ..., 0.95])
        bid_method: Method for calculating bid price
        bid_prices_: Computed bid prices (EUR/MWh) - set after predict()
    """

    def __init__(self, model, quantiles: List[float], bid_method: str, name: str):
        """
        Initialize bid curve strategy.

        Args:
            model: Trained quantile regressor with predict() method
            quantiles: List of quantile levels used by model
            bid_method: Identifier for bid calculation method
            name: Human-readable strategy name
        """
        super().__init__(name=name)
        self.model = model
        self.quantiles = quantiles
        self.bid_method = bid_method
        self.bid_prices_ = None  # Populated after predict()

        # Store feature names from model if available
        self.feature_columns = getattr(model, 'feature_names', None)

    @abstractmethod
    def calculate_bid_price(self, quantile_preds: np.ndarray) -> np.ndarray:
        """
        Convert quantile predictions to optimal bid price.

        Args:
            quantile_preds: Quantile forecasts, shape (n_samples, n_quantiles)
                           Each row is [q05, q10, ..., q95] in EUR/MWh

        Returns:
            bid_prices: Shape (n_samples,) - EUR/MWh bid for each quarter-hour
        """
        pass

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate binary curtail decisions based on bid prices.

        This method is called by backtester. Since we don't know actual prices
        here, we return zeros (placeholder). The AuctionBacktester will use
        calculate_bid_price() instead to simulate auction clearing.

        Args:
            df: DataFrame with features

        Returns:
            Binary array (1=curtailed, 0=generated) - placeholder zeros
        """
        # Get calibrated quantile predictions
        if self.feature_columns is not None:
            X = df[self.feature_columns]
        else:
            X = df

        quantile_preds = self.model.predict(X, enforce_monotonicity=True)

        # Calculate bid prices
        self.bid_prices_ = self.calculate_bid_price(quantile_preds)

        # Placeholder - actual clearing determined by AuctionBacktester
        return np.zeros(len(df), dtype=int)

    def get_bid_prices(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get bid prices directly without predict() call.

        Used by AuctionBacktester to access bid prices.

        Args:
            df: DataFrame with features

        Returns:
            Bid prices in EUR/MWh, shape (n_samples,)
        """
        if self.feature_columns is not None:
            X = df[self.feature_columns]
        else:
            X = df

        quantile_preds = self.model.predict(X, enforce_monotonicity=True)
        return self.calculate_bid_price(quantile_preds)


class ExpectedValueBidStrategy(BidCurveStrategy):
    """
    Bid at expected price E[price] - optimal under risk neutrality.

    Computes E[price] via trapezoidal integration across quantile levels:
        E[price] ≈ Σ [(q_i + q_{i+1})/2] × (α_{i+1} - α_i)

    This is the theoretically optimal bid if the operator has no risk aversion
    and maximizes expected revenue.

    Example:
        >>> strategy = ExpectedValueBidStrategy(model, quantiles=[0.1, 0.5, 0.9])
        >>> bid = strategy.calculate_bid_price(np.array([[-10, 0, 10]]))
        >>> # E[price] = -10*0.4 + 0*0.0 + 10*0.4 = 0 EUR/MWh
    """

    def __init__(self, model, quantiles: List[float]):
        super().__init__(
            model=model,
            quantiles=quantiles,
            bid_method='expected_value',
            name='Expected Value (E[price])'
        )

    def calculate_bid_price(self, quantile_preds: np.ndarray) -> np.ndarray:
        """
        Calculate expected price using trapezoidal integration.

        For quantiles α = [0.05, 0.10, ..., 0.95] with predictions q:
            E[price] ≈ Σ [(q_i + q_{i+1})/2] × (α_{i+1} - α_i)

        Args:
            quantile_preds: Shape (n_samples, n_quantiles)

        Returns:
            Expected prices, shape (n_samples,)
        """
        n_samples = quantile_preds.shape[0]
        expected_prices = np.zeros(n_samples)

        for i in range(len(self.quantiles) - 1):
            alpha_i = self.quantiles[i]
            alpha_next = self.quantiles[i + 1]
            q_i = quantile_preds[:, i]
            q_next = quantile_preds[:, i + 1]

            # Trapezoidal area: average of two quantiles × probability mass
            prob_mass = alpha_next - alpha_i
            avg_price = (q_i + q_next) / 2
            expected_prices += avg_price * prob_mass

        return expected_prices


class MedianBidStrategy(BidCurveStrategy):
    """
    Conservative strategy: bid at median (q50).

    Bidding at the median means:
    - 50% chance of being cleared (market price >= bid)
    - 50% chance of being curtailed (market price < bid)

    More conservative than expected value, suitable for risk-averse operators.
    """

    def __init__(self, model, quantiles: List[float]):
        super().__init__(
            model=model,
            quantiles=quantiles,
            bid_method='median',
            name='Median (q50)'
        )

        # Find index of q50
        try:
            self.q50_idx = quantiles.index(0.50)
        except ValueError:
            raise ValueError("Quantiles list must contain 0.50 for median bidding")

    def calculate_bid_price(self, quantile_preds: np.ndarray) -> np.ndarray:
        """Return median (q50) as bid price."""
        return quantile_preds[:, self.q50_idx]


class QuantileBidStrategy(BidCurveStrategy):
    """
    Bid at a specific quantile level (risk-adjusted).

    Bidding at quantile α means:
    - (1-α)% chance of being cleared
    - α% chance of being curtailed

    Examples:
        q10 bid: 90% chance of clearing (very conservative)
        q25 bid: 75% chance of clearing (conservative)
        q40 bid: 60% chance of clearing (moderate)
        q50 bid: 50% chance of clearing (neutral)

    Lower quantiles → more conservative → more curtailment → lower revenue variance
    """

    def __init__(self, model, quantiles: List[float], target_quantile: float):
        """
        Initialize quantile-based bid strategy.

        Args:
            model: Trained quantile regressor
            quantiles: List of available quantile levels
            target_quantile: Quantile level to bid at (e.g., 0.10, 0.25, 0.40)
        """
        if target_quantile not in quantiles:
            raise ValueError(
                f"Target quantile {target_quantile} not in available quantiles {quantiles}"
            )

        super().__init__(
            model=model,
            quantiles=quantiles,
            bid_method=f'quantile_{target_quantile:.2f}',
            name=f'Quantile q{target_quantile:.2f}'
        )
        self.target_quantile = target_quantile
        self.target_idx = quantiles.index(target_quantile)

    def calculate_bid_price(self, quantile_preds: np.ndarray) -> np.ndarray:
        """Return target quantile as bid price."""
        return quantile_preds[:, self.target_idx]


class FixedFloorBidStrategy(BaseStrategy):
    """
    Fixed price floor strategy (non-adaptive).

    Always bid the same price regardless of forecast. Common industry practice:
    - Floor = -€5/MWh: Accept mild negative prices
    - Floor = -€10/MWh: Accept moderate negative prices
    - Floor = -€20/MWh: Accept severe negative prices (rare)

    Advantages:
    - Simple, easy to implement
    - Predictable behavior
    - No ML model required

    Disadvantages:
    - Doesn't adapt to forecast distribution
    - May over-curtail when prices expected to be only slightly negative
    - May under-curtail when prices expected to be very negative

    Example:
        >>> strategy = FixedFloorBidStrategy(floor_price=-5.0)
        >>> bid = strategy.get_bid_prices(X)  # Always -5.0 EUR/MWh
    """

    def __init__(self, floor_price: float):
        """
        Initialize fixed floor strategy.

        Args:
            floor_price: Fixed bid price in EUR/MWh (typically negative)
        """
        super().__init__(name=f'Fixed Floor ({floor_price:+.0f} EUR/MWh)')
        self.floor_price = floor_price
        self.bid_prices_ = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Placeholder for BaseStrategy interface."""
        self.bid_prices_ = np.full(len(df), self.floor_price)
        return np.zeros(len(df), dtype=int)

    def get_bid_prices(self, df: pd.DataFrame) -> np.ndarray:
        """Return fixed floor price for all samples."""
        return np.full(len(df), self.floor_price)


class PerfectForesightBidStrategy(BaseStrategy):
    """
    Theoretical upper bound: bid exactly at actual price.

    This strategy has perfect knowledge of future prices and bids just below
    the clearing price to maximize revenue:
    - If price >= 0: bid at 0 (capture positive revenue)
    - If price < 0: bid at price (get curtailed, avoid negative revenue)

    Provides the theoretical maximum savings achievable.
    Used as upper bound benchmark.
    """

    def __init__(self):
        super().__init__(name='Perfect Foresight (Theoretical Max)')
        self.bid_prices_ = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate optimal decisions with perfect foresight.

        Args:
            df: Must contain 'price_eur_mwh' column

        Returns:
            Binary array: 1 if should curtail (price < 0), 0 otherwise
        """
        if 'price_eur_mwh' not in df.columns:
            raise ValueError("DataFrame must contain 'price_eur_mwh' for perfect foresight")

        prices = df['price_eur_mwh'].values

        # Bid at actual price (or just below)
        self.bid_prices_ = prices - 0.01  # Bid slightly below to ensure clearing when profitable

        # Curtail when price is negative
        return (prices < 0).astype(int)

    def get_bid_prices(self, df: pd.DataFrame) -> np.ndarray:
        """Return actual prices as 'bids' for theoretical maximum."""
        if 'price_eur_mwh' not in df.columns:
            raise ValueError("DataFrame must contain 'price_eur_mwh' for perfect foresight")

        return df['price_eur_mwh'].values - 0.01


# Factory function for easy strategy creation
def create_bid_strategy(
    strategy_type: str,
    model=None,
    quantiles: Optional[List[float]] = None,
    **kwargs
) -> BaseStrategy:
    """
    Factory function to create bid strategies.

    Args:
        strategy_type: One of 'expected_value', 'median', 'quantile', 'fixed_floor'
        model: Trained quantile regressor (required for ML strategies)
        quantiles: List of quantile levels (required for ML strategies)
        **kwargs: Additional strategy-specific parameters:
            - target_quantile: float for QuantileBidStrategy
            - floor_price: float for FixedFloorBidStrategy

    Returns:
        Initialized strategy instance

    Example:
        >>> strategy = create_bid_strategy('expected_value', model, quantiles)
        >>> strategy = create_bid_strategy('fixed_floor', floor_price=-5.0)
    """
    if strategy_type == 'expected_value':
        if model is None or quantiles is None:
            raise ValueError("Model and quantiles required for expected_value strategy")
        return ExpectedValueBidStrategy(model, quantiles)

    elif strategy_type == 'median':
        if model is None or quantiles is None:
            raise ValueError("Model and quantiles required for median strategy")
        return MedianBidStrategy(model, quantiles)

    elif strategy_type == 'quantile':
        if model is None or quantiles is None:
            raise ValueError("Model and quantiles required for quantile strategy")
        target_quantile = kwargs.get('target_quantile')
        if target_quantile is None:
            raise ValueError("target_quantile parameter required for quantile strategy")
        return QuantileBidStrategy(model, quantiles, target_quantile)

    elif strategy_type == 'fixed_floor':
        floor_price = kwargs.get('floor_price')
        if floor_price is None:
            raise ValueError("floor_price parameter required for fixed_floor strategy")
        return FixedFloorBidStrategy(floor_price)

    elif strategy_type == 'perfect_foresight':
        return PerfectForesightBidStrategy()

    else:
        raise ValueError(
            f"Unknown strategy_type '{strategy_type}'. "
            f"Must be one of: expected_value, median, quantile, fixed_floor, perfect_foresight"
        )
