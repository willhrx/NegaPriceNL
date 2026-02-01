"""
Benchmark strategies for negative price prediction.

These strategies serve as baselines to compare against the ML model:
1. NaiveStrategy: Predict based on yesterday's same hour
2. HeuristicStrategy: Rule-based on weekend/solar peak/season
3. SolarThresholdStrategy: Domain knowledge (high solar + low load)
4. MLStrategy: Trained classifier
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional


class BaseStrategy(ABC):
    """Abstract base class for prediction strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features (must have appropriate columns)

        Returns
        -------
        np.ndarray
            Binary predictions (1 = predict negative price, 0 = predict positive)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class NaiveStrategy(BaseStrategy):
    """
    Predict negative if yesterday's same hour was negative.

    Uses the is_negative_lag_24h feature (lagged 24 hours).
    Simple but captures temporal persistence of negative price events.
    """

    def __init__(self):
        super().__init__(name="Naive (Yesterday Same Hour)")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict negative if 24h ago was negative."""
        if 'is_negative_lag_24h' not in df.columns:
            raise ValueError("DataFrame must contain 'is_negative_lag_24h' column")

        # Handle NaN at start of series (first 24 hours)
        predictions = df['is_negative_lag_24h'].fillna(0).astype(int).values
        return predictions


class HeuristicStrategy(BaseStrategy):
    """
    Rule-based prediction using domain knowledge.

    Predict negative if:
    - Hour is during solar peak (10:00-14:00)
    - AND it's a weekend (lower demand)
    - AND it's spring/summer (March-September, high solar)

    This captures the typical negative price pattern in the Netherlands.
    """

    def __init__(self):
        super().__init__(name="Heuristic (Weekend + Solar Peak + Season)")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict negative based on time/season heuristics."""
        required_cols = ['hour', 'is_weekend', 'month']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")

        is_risky = (
            (df['hour'] >= 10) & (df['hour'] <= 14) &  # Solar peak hours
            (df['is_weekend'] == 1) &                   # Weekend (lower demand)
            (df['month'] >= 3) & (df['month'] <= 9)     # Spring/summer
        )
        return is_risky.astype(int).values


class SolarThresholdStrategy(BaseStrategy):
    """
    Domain knowledge: high renewable penetration + low load.

    Predict negative if:
    - RES penetration > threshold (e.g., 0.5 = 50% of load from renewables)
    - AND load is below median

    This directly captures the fundamental driver of negative prices.
    """

    def __init__(self, res_threshold: float = 0.5):
        super().__init__(name=f"Solar Threshold (RES>{res_threshold*100:.0f}% + Low Load)")
        self.res_threshold = res_threshold

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict negative based on RES penetration threshold."""
        if 'res_penetration' not in df.columns:
            raise ValueError("DataFrame must contain 'res_penetration' column")
        if 'load_mw' not in df.columns:
            raise ValueError("DataFrame must contain 'load_mw' column")

        median_load = df['load_mw'].median()

        is_risky = (
            (df['res_penetration'] > self.res_threshold) &
            (df['load_mw'] < median_load)
        )
        return is_risky.astype(int).values


class AlwaysCurtailStrategy(BaseStrategy):
    """
    Always predict negative (always curtail).

    Useful as a baseline to show the cost of over-curtailment.
    """

    def __init__(self):
        super().__init__(name="Always Curtail")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Always predict negative."""
        return np.ones(len(df), dtype=int)


class NeverCurtailStrategy(BaseStrategy):
    """
    Never predict negative (never curtail).

    This is the baseline "do nothing" strategy.
    """

    def __init__(self):
        super().__init__(name="Never Curtail")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Never predict negative."""
        return np.zeros(len(df), dtype=int)


class MLStrategy(BaseStrategy):
    """
    ML model predictions using trained classifier.

    Loads a trained model and threshold, generates probability predictions,
    and applies the threshold to create binary predictions.

    Supports both:
    - NegativePriceXGBoost wrapper (saved as dict with 'model' key)
    - Direct sklearn model objects
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold_path: Optional[Path] = None,
        feature_columns_path: Optional[Path] = None,
        model_version: str = "v4"
    ):
        super().__init__(name=f"ML Model ({model_version})")
        self.model_version = model_version
        self.model = None  # The underlying sklearn model
        self.threshold = None
        self.feature_columns = None

        # Load model artifacts if paths provided
        if model_path and threshold_path:
            self.load_model(model_path, threshold_path, feature_columns_path)

    def load_model(
        self,
        model_path: Path,
        threshold_path: Path,
        feature_columns_path: Optional[Path] = None
    ):
        """Load trained model and threshold."""
        model_data = joblib.load(model_path)

        # Handle NegativePriceXGBoost wrapper (saved as dict)
        if isinstance(model_data, dict) and 'model' in model_data:
            self.model = model_data['model']  # Extract the sklearn model
            # Use feature names from wrapper if available
            if 'feature_names' in model_data:
                self.feature_columns = model_data['feature_names']
            # Threshold might be in the dict too
            if 'threshold' in model_data and threshold_path is None:
                self.threshold = model_data['threshold']
        else:
            # Direct sklearn model
            self.model = model_data

        # Load threshold from separate file
        self.threshold = joblib.load(threshold_path)

        # Load feature columns if provided
        if feature_columns_path and feature_columns_path.exists():
            self.feature_columns = joblib.load(feature_columns_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Select feature columns
        if self.feature_columns is not None:
            X = df[self.feature_columns]
        else:
            # Exclude known non-feature columns
            exclude_cols = ['price_eur_mwh', 'is_negative_price', 'datetime', 'Actual Aggregated']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            X = df[feature_cols]

        # Generate probability predictions
        proba = self.model.predict_proba(X)[:, 1]

        # Apply threshold
        predictions = (proba >= self.threshold).astype(int)
        return predictions

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw probability predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.feature_columns is not None:
            X = df[self.feature_columns]
        else:
            exclude_cols = ['price_eur_mwh', 'is_negative_price', 'datetime', 'Actual Aggregated']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            X = df[feature_cols]

        return self.model.predict_proba(X)[:, 1]


def get_all_strategies(
    model_path: Optional[Path] = None,
    threshold_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
    model_version: str = "v4"
) -> list:
    """
    Get list of all benchmark strategies.

    Parameters
    ----------
    model_path : Path, optional
        Path to trained ML model
    threshold_path : Path, optional
        Path to optimal threshold
    feature_columns_path : Path, optional
        Path to feature column list
    model_version : str
        Model version label

    Returns
    -------
    list
        List of strategy instances
    """
    strategies = [
        NeverCurtailStrategy(),
        NaiveStrategy(),
        HeuristicStrategy(),
        SolarThresholdStrategy(res_threshold=0.5),
    ]

    # Add ML strategy if model provided
    if model_path and threshold_path:
        ml_strategy = MLStrategy(
            model_path=model_path,
            threshold_path=threshold_path,
            feature_columns_path=feature_columns_path,
            model_version=model_version
        )
        strategies.append(ml_strategy)

    return strategies
