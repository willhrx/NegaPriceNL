"""
XGBoost Quantile Regressor for Price Distribution Forecasting

Trains 5 separate XGBRegressor models (one per quantile alpha) to predict
the 10th, 25th, 50th, 75th, and 90th percentiles of the day-ahead price
distribution for each quarter-hour.

From these quantile forecasts, derives:
- P(price < 0) and P(price < -10 EUR)
- Expected price conditional on being negative
- Prediction interval width as a confidence measure
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RANDOM_STATE

DEFAULT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


class NegativePriceQuantileRegressor:
    """
    Multi-quantile XGBoost regressor for electricity price forecasting.

    Trains one XGBRegressor per quantile level, each with
    objective='reg:quantileerror'. Provides methods to derive
    probability estimates and prediction intervals from the
    quantile forecasts.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.params = params or self._default_params()
        self.models: Dict[float, xgb.XGBRegressor] = {}
        self.feature_names: Optional[List[str]] = None

    def _default_params(self) -> Dict[str, Any]:
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True,
    ) -> 'NegativePriceQuantileRegressor':
        self.feature_names = list(X_train.columns)

        for alpha in self.quantiles:
            params = self.params.copy()
            params['objective'] = 'reg:quantileerror'
            params['quantile_alpha'] = alpha

            if X_val is not None and y_val is not None:
                params['early_stopping_rounds'] = early_stopping_rounds

            model = xgb.XGBRegressor(**params)

            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False

            model.fit(X_train, y_train, **fit_kwargs)
            self.models[alpha] = model

            if verbose:
                n_trees = model.get_booster().num_boosted_rounds()
                print(f"  Quantile {alpha:.2f}: {n_trees} trees")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict all quantiles.

        Returns
        -------
        np.ndarray of shape (n_samples, n_quantiles)
            Price forecasts in EUR/MWh for each quantile.
        """
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")

        predictions = np.column_stack([
            self.models[alpha].predict(X) for alpha in self.quantiles
        ])
        return predictions

    def predict_quantile(self, X: pd.DataFrame, alpha: float) -> np.ndarray:
        """Predict a single quantile."""
        if alpha not in self.models:
            raise ValueError(f"Quantile {alpha} not trained. Available: {list(self.models.keys())}")
        return self.models[alpha].predict(X)

    def derive_probabilities(
        self, X: pd.DataFrame, threshold: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Derive probability estimates from quantile forecasts.

        Uses linear interpolation between quantile levels to estimate
        the probability that the price falls below a given threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        threshold : float
            Price threshold in EUR/MWh (default 0.0 for negative prices)

        Returns
        -------
        dict with keys:
            'p_below_threshold': P(price < threshold) per sample
            'p_negative': P(price < 0) per sample
            'p_below_minus10': P(price < -10) per sample
            'median_forecast': q50 forecast per sample
            'interval_width': q90 - q10 per sample
            'expected_negative_price': conditional expectation E[price | price < 0]
        """
        preds = self.predict(X)  # (n, 5)
        alphas = np.array(self.quantiles)

        p_negative = self._interpolate_probability(preds, alphas, 0.0)
        p_below_minus10 = self._interpolate_probability(preds, alphas, -10.0)

        # Median forecast (q50 column)
        q50_idx = self.quantiles.index(0.50)
        median_forecast = preds[:, q50_idx]

        # Interval width
        q10_idx = self.quantiles.index(0.10)
        q90_idx = self.quantiles.index(0.90)
        interval_width = preds[:, q90_idx] - preds[:, q10_idx]

        # Conditional expectation E[price | price < 0]
        # Approximate from quantiles below zero
        expected_neg = self._conditional_expectation_negative(preds, alphas)

        return {
            'p_negative': p_negative,
            'p_below_minus10': p_below_minus10,
            'median_forecast': median_forecast,
            'interval_width': interval_width,
            'expected_negative_price': expected_neg,
            'quantile_predictions': preds,
        }

    @staticmethod
    def _interpolate_probability(
        preds: np.ndarray, alphas: np.ndarray, threshold: float
    ) -> np.ndarray:
        """
        Estimate P(price < threshold) by interpolating between quantiles.

        For each sample, find where the quantile forecasts cross the threshold
        and linearly interpolate to get the probability.
        """
        n_samples = preds.shape[0]
        probs = np.zeros(n_samples)

        for i in range(n_samples):
            quantile_values = preds[i]

            if threshold <= quantile_values[0]:
                # Below all quantiles — extrapolate: prob < lowest alpha
                probs[i] = alphas[0] * (threshold / quantile_values[0]) if quantile_values[0] < 0 else 0.0
                probs[i] = max(0.0, probs[i])
            elif threshold >= quantile_values[-1]:
                # Above all quantiles
                probs[i] = 1.0 - (1.0 - alphas[-1]) * 0.5  # Conservative upper bound
                probs[i] = min(1.0, probs[i])
            else:
                # Interpolate between the two quantiles that bracket the threshold
                for j in range(len(alphas) - 1):
                    if quantile_values[j] <= threshold <= quantile_values[j + 1]:
                        # Linear interpolation
                        frac = (threshold - quantile_values[j]) / (quantile_values[j + 1] - quantile_values[j])
                        probs[i] = alphas[j] + frac * (alphas[j + 1] - alphas[j])
                        break

        return probs

    @staticmethod
    def _conditional_expectation_negative(
        preds: np.ndarray, alphas: np.ndarray
    ) -> np.ndarray:
        """
        Approximate E[price | price < 0] from quantile forecasts.

        Uses the average of quantile forecasts that are below zero,
        weighted by the quantile spacing.
        """
        n_samples = preds.shape[0]
        expected = np.full(n_samples, np.nan)

        for i in range(n_samples):
            neg_mask = preds[i] < 0
            if neg_mask.any():
                # Average of negative quantile forecasts
                expected[i] = preds[i][neg_mask].mean()

        return expected

    def predict_binary(self, X: pd.DataFrame, price_threshold: float = 0.0) -> np.ndarray:
        """
        Derive binary curtailment decisions from median forecast.

        Curtail (1) if median price forecast < threshold, else generate (0).
        Compatible with the existing EconomicBacktester pipeline.
        """
        q50_idx = self.quantiles.index(0.50)
        median = self.models[self.quantiles[q50_idx]].predict(X)
        return (median < price_threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Aggregate feature importance across all quantile models."""
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")

        importances = np.zeros(len(self.feature_names))
        for model in self.models.values():
            importances += model.feature_importances_

        importances /= len(self.models)

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=False)

    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """Permutation importance using the median (q50) model."""
        q50_idx = self.quantiles.index(0.50)
        model = self.models[self.quantiles[q50_idx]]

        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': result.importances_mean,
            'importance_std': result.importances_std,
        }).sort_values('importance', ascending=False)

    def save(self, path: Path) -> None:
        if not self.models:
            raise ValueError("Model not trained. Call fit() first.")

        model_data = {
            'models': self.models,
            'params': self.params,
            'quantiles': self.quantiles,
            'feature_names': self.feature_names,
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Path) -> 'NegativePriceQuantileRegressor':
        model_data = joblib.load(path)

        instance = cls(
            quantiles=model_data['quantiles'],
            params=model_data['params'],
        )
        instance.models = model_data['models']
        instance.feature_names = model_data['feature_names']

        return instance
