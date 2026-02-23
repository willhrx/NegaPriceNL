"""
Rolling-Window Conformal Calibration

Implements production-realistic backtesting via rolling calibration windows.

Motivation (from Dexter Energy whitepaper):
- Static calibration uses a fixed calibration set (e.g., Sep-Dec 2024)
- In production, the model would be recalibrated daily as new actuals arrive
- Rolling calibration simulates this by using a sliding window of recent data

Algorithm:
For each test day D:
1. Gather calibration window: [D - window_days, D - 2] actuals
   (D-1 is the latest known at D-1 12:00 CET auction time)
2. Compute conformity scores on this window
3. Apply corrections to day D's predictions
4. Falls back to static calibration for days without enough history

This approach:
- Adapts to regime shifts (e.g., if prices drop in March, April calibration reflects this)
- Provides more realistic performance estimates for production deployment
- Matches the retraining cadence assumed in operational planning
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from src.models.conformal_calibrator import ConformalCalibrator

logger = logging.getLogger(__name__)


class RollingConformalCalibrator:
    """
    Rolling-window conformal calibration for production-realistic backtesting.

    For each test day D, calibrate using [D-window, D-2] actuals.
    Falls back to static corrections for days without enough calibration history.

    Example:
        >>> roller = RollingConformalCalibrator(quantiles=[0.1, 0.5, 0.9], window_days=30)
        >>> calibrated_preds = roller.calibrate_and_predict(
        ...     y_all, preds_all, index_all, test_start, static_corrections
        ... )
    """

    def __init__(self, quantiles: List[float], window_days: int = 30):
        """
        Initialize rolling conformal calibrator.

        Args:
            quantiles: List of quantile levels
            window_days: Size of rolling calibration window (days)
        """
        self.quantiles = quantiles
        self.window_days = window_days
        self.window_qh = window_days * 96  # 15-min periods in window

        logger.info(f"Initialized RollingConformalCalibrator:")
        logger.info(f"  Window size: {window_days} days ({self.window_qh:,} QH periods)")
        logger.info(f"  Quantiles: {quantiles}")

    def calibrate_and_predict(
        self,
        y_all: np.ndarray,
        preds_all: np.ndarray,
        index_all: pd.DatetimeIndex,
        test_start: pd.Timestamp,
        static_corrections: Optional[Dict[float, float]] = None
    ) -> np.ndarray:
        """
        Walk through test period day by day, applying rolling calibration.

        Args:
            y_all: All actual values (train + val + cal + test), shape (n_total,)
            preds_all: All raw predictions, shape (n_total, n_quantiles)
            index_all: Datetime index for all samples
            test_start: Start of test period
            static_corrections: Fallback corrections for insufficient history
                               (dict mapping quantile → correction value)

        Returns:
            Calibrated predictions for test period, shape (n_test, n_quantiles)

        Note:
            - Only test period predictions are modified
            - Training/val/cal predictions remain unchanged (returned as-is)
        """
        logger.info("\n" + "=" * 70)
        logger.info("ROLLING CONFORMAL CALIBRATION")
        logger.info("=" * 70)
        logger.info(f"Test start: {test_start}")
        logger.info(f"Window: {self.window_days} days")

        calibrated_preds = preds_all.copy()
        test_mask = index_all >= test_start

        # Get unique test days
        unique_days = sorted(index_all[test_mask].normalize().unique())
        logger.info(f"Test days: {len(unique_days)}")

        days_using_static = 0
        days_using_rolling = 0

        for day_idx, day in enumerate(unique_days):
            # Mask for current day's quarter-hours
            day_mask = index_all.normalize() == day
            if not day_mask.any():
                continue

            # Calibration window: [day - window_days, day - 1 day)
            # We use up to D-2 because D-1 prices are the latest known at auction time
            cal_end = day
            cal_start = day - pd.Timedelta(days=self.window_days)
            cal_mask = (index_all >= cal_start) & (index_all < cal_end)

            cal_samples = cal_mask.sum()

            # Check if we have enough calibration data
            if cal_samples < 96:  # Less than 1 day of QH data
                # Fall back to static calibration
                if static_corrections:
                    for i, alpha in enumerate(self.quantiles):
                        calibrated_preds[day_mask, i] += static_corrections[alpha]
                    days_using_static += 1

                    if day_idx < 5:  # Log first few days
                        logger.info(
                            f"  {day.date()}: Static fallback ({cal_samples} cal samples < 96)"
                        )
                else:
                    logger.warning(
                        f"  {day.date()}: Insufficient cal data ({cal_samples} samples) "
                        f"and no static fallback provided - predictions uncalibrated"
                    )
                continue

            # Compute rolling conformity scores
            cal = ConformalCalibrator(self.quantiles)
            cal.calibrate(y_all[cal_mask], preds_all[cal_mask])

            # Apply to today's predictions
            for i, alpha in enumerate(self.quantiles):
                calibrated_preds[day_mask, i] += cal.corrections[alpha]

            days_using_rolling += 1

            # Log sample of days
            if day_idx < 5 or day_idx % 30 == 0:
                corrections_str = ", ".join([
                    f"q{alpha:.2f}={cal.corrections[alpha]:+.1f}"
                    for alpha in self.quantiles[::2]  # Every other quantile
                ])
                logger.info(
                    f"  {day.date()}: Rolling cal ({cal_samples:,} samples) — {corrections_str}"
                )

        # Enforce monotonicity across all predictions
        # This is critical after applying corrections
        logger.info("\nEnforcing monotonicity...")
        crossing_before = 0
        crossing_after = 0

        for j in range(calibrated_preds.shape[0]):
            # Check crossing before sort
            if not np.all(calibrated_preds[j, :-1] <= calibrated_preds[j, 1:]):
                crossing_before += 1

            # Sort to enforce monotonicity
            calibrated_preds[j] = np.sort(calibrated_preds[j])

        logger.info(f"  Crossings fixed: {crossing_before:,} → 0")

        logger.info("\n" + "=" * 70)
        logger.info("ROLLING CALIBRATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Total test days: {len(unique_days)}")
        logger.info(f"  Using rolling calibration: {days_using_rolling} ({days_using_rolling/len(unique_days)*100:.1f}%)")
        logger.info(f"  Using static fallback: {days_using_static} ({days_using_static/len(unique_days)*100:.1f}%)")

        if static_corrections:
            logger.info("\n  Static corrections (used for first ~30 days):")
            for alpha in self.quantiles:
                logger.info(f"    q{alpha:.2f}: {static_corrections[alpha]:+7.2f} EUR/MWh")

        return calibrated_preds
