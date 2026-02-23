"""
Conformalized Quantile Regression (CQR) Calibrator

Implements the static conformal calibration approach from the Dexter Energy
whitepaper on probabilistic electricity price forecasting.

Key concept:
- Standard quantile regression produces raw quantile estimates q̂_α(x)
- These are only well-calibrated if test distribution matches training distribution
- CQR computes conformity scores on a calibration set to correct for distribution shift

Algorithm:
1. Train quantile models on training data → raw predictions q̂_α(x)
2. Predict on calibration set and compute residuals: r_i = y_cal_i - q̂_α(x_cal_i)
3. For each quantile α, compute correction: Δ_α = quantile(residuals, α)
4. Apply to test predictions: q̃_α(x) = q̂_α(x) + Δ_α

This provides distribution-free marginal coverage guarantees under exchangeability.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """
    Static conformal calibrator for quantile predictions.

    For each quantile α, computes correction Δ_α such that exactly α fraction
    of calibration actuals fall below the corrected quantile.

    Example:
        >>> calibrator = ConformalCalibrator(quantiles=[0.1, 0.5, 0.9])
        >>> calibrator.calibrate(y_cal, raw_preds_cal)
        >>> corrected_preds = calibrator.apply(raw_preds_test)
    """

    def __init__(self, quantiles: List[float]):
        """
        Initialize conformal calibrator.

        Args:
            quantiles: List of quantile levels (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
        """
        self.quantiles = quantiles
        self.corrections = {}  # α → correction value (EUR/MWh)
        self.is_calibrated = False

    def calibrate(self, y_cal: np.ndarray, quantile_preds_cal: np.ndarray):
        """
        Compute per-quantile corrections from calibration set.

        Args:
            y_cal: Actual values on calibration set (n_samples,)
            quantile_preds_cal: Raw quantile predictions (n_samples, n_quantiles)

        For each quantile α:
            residual_i = y_cal_i - q̂_α(x_cal_i)
            correction_α = quantile(residuals, α)

        The corrected prediction becomes:
            q̃_α(x) = q̂_α(x) + correction_α

        Intuition: If the raw model predicts too high (positive bias), residuals
        will be negative on average, so corrections will be negative, shifting
        the distribution downward.
        """
        if len(y_cal) != quantile_preds_cal.shape[0]:
            raise ValueError(
                f"y_cal length ({len(y_cal)}) must match quantile_preds_cal rows "
                f"({quantile_preds_cal.shape[0]})"
            )

        if quantile_preds_cal.shape[1] != len(self.quantiles):
            raise ValueError(
                f"quantile_preds_cal columns ({quantile_preds_cal.shape[1]}) must match "
                f"number of quantiles ({len(self.quantiles)})"
            )

        logger.info(f"Computing conformity scores on {len(y_cal):,} calibration samples...")

        for i, alpha in enumerate(self.quantiles):
            # Residuals: actual - predicted
            residuals = y_cal - quantile_preds_cal[:, i]

            # The α-th quantile of the residuals
            # This is the shift needed to achieve nominal coverage
            self.corrections[alpha] = np.quantile(residuals, alpha)

            logger.info(
                f"  q{alpha:.2f}: correction = {self.corrections[alpha]:+7.2f} EUR/MWh "
                f"(residual range: [{residuals.min():.1f}, {residuals.max():.1f}])"
            )

        self.is_calibrated = True

    def apply(self, quantile_preds_raw: np.ndarray) -> np.ndarray:
        """
        Apply calibration corrections to raw quantile predictions.

        Args:
            quantile_preds_raw: Raw predictions (n_samples, n_quantiles)

        Returns:
            Corrected predictions with monotonicity enforced (n_samples, n_quantiles)

        Raises:
            ValueError: If calibrator not fitted
        """
        if not self.is_calibrated:
            raise ValueError("Calibrator not fitted. Call calibrate() first.")

        if quantile_preds_raw.shape[1] != len(self.quantiles):
            raise ValueError(
                f"quantile_preds_raw columns ({quantile_preds_raw.shape[1]}) must match "
                f"number of quantiles ({len(self.quantiles)})"
            )

        corrected = quantile_preds_raw.copy()

        # Apply per-quantile corrections
        for i, alpha in enumerate(self.quantiles):
            corrected[:, i] += self.corrections[alpha]

        # Enforce monotonicity (quantiles must not cross)
        # This is critical for valid probability distributions
        for j in range(corrected.shape[0]):
            corrected[j] = np.sort(corrected[j])

        return corrected

    def get_corrections_summary(self) -> pd.DataFrame:
        """
        Return summary of corrections for logging and analysis.

        Returns:
            DataFrame with columns ['quantile', 'correction_eur_mwh']
        """
        if not self.is_calibrated:
            raise ValueError("Calibrator not fitted. Call calibrate() first.")

        return pd.DataFrame([
            {'quantile': alpha, 'correction_eur_mwh': self.corrections[alpha]}
            for alpha in self.quantiles
        ])

    def save(self, filepath: str):
        """
        Save calibrator to pickle file.

        Args:
            filepath: Path to save file
        """
        import pickle
        if not self.is_calibrated:
            raise ValueError("Cannot save uncalibrated calibrator")

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Saved calibrator to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'ConformalCalibrator':
        """
        Load calibrator from pickle file.

        Args:
            filepath: Path to saved calibrator

        Returns:
            Loaded ConformalCalibrator instance
        """
        import pickle
        with open(filepath, 'rb') as f:
            calibrator = pickle.load(f)

        logger.info(f"Loaded calibrator from {filepath}")
        return calibrator
