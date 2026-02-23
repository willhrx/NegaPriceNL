"""
Settings for model v10 (Conformalized Quantile Regression).

Extends base settings.py with v10-specific configurations:
- New data splits (train extended to H1 2024, dedicated calibration period)
- Expanded quantile set (9 levels for better tail coverage)
- Conformal calibration parameters

Separate from settings.py to preserve reproducibility of v5–v9 models
which depend on the original split boundaries.
"""
import pandas as pd
from config.settings import *  # Import all base settings

# ==============================================================================
# Data Split Boundaries for v10
# ==============================================================================

# Training: 2019-01-01 → 2024-06-30 (includes first half of 2024)
TRAIN_END_DATE = pd.Timestamp('2024-06-30 23:45:00', tz='UTC')

# Validation: 2024-07-01 → 2024-08-31 (early stopping, sufficient with 2 months)
VAL_START_DATE = pd.Timestamp('2024-07-01 00:00:00', tz='UTC')
VAL_END_DATE = pd.Timestamp('2024-08-31 23:45:00', tz='UTC')

# Calibration: 2024-09-01 → 2024-12-31 (static conformal calibration)
# This is the most recent data before the test period, used to compute
# conformity scores that shift the distribution to match 2025 regime
CAL_START_DATE = pd.Timestamp('2024-09-01 00:00:00', tz='UTC')
CAL_END_DATE = pd.Timestamp('2024-12-31 23:45:00', tz='UTC')

# Test: 2025-01-01 → 2025-12-31 (final evaluation)
TEST_START_DATE = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
TEST_END_DATE = pd.Timestamp('2025-12-31 23:45:00', tz='UTC')

# ==============================================================================
# Conformal Calibration Parameters
# ==============================================================================

# Rolling window size (days) for production-realistic backtesting
# For each test day D, calibrate using [D-30, D-2] actuals
CONFORMAL_WINDOW_DAYS = 30

# Minimum samples required for rolling calibration (fallback to static if less)
CONFORMAL_MIN_SAMPLES = 96  # 1 day of quarter-hourly data

# ==============================================================================
# Quantile Configuration for v10
# ==============================================================================

# Expanded quantile set: 9 levels (vs 5 in v9)
# Benefits:
# - Better CRPS approximation (more quantiles → better integral estimate)
# - Improved tail resolution (q05/q95 capture extreme negative price events)
# - More calibration anchor points for conformity score computation
QUANTILES_V10 = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95]

# ==============================================================================
# Model Version
# ==============================================================================

MODEL_VERSION_V10 = "v10"

# ==============================================================================
# Notes on Reproducibility
# ==============================================================================

# v5-v9 models use the original settings.py splits:
#   Train: 2019-2023 (5 years)
#   Val:   2024 (1 year)
#   Test:  2025 (1 year)
#
# v10 uses these new splits to:
#   - Include H1 2024 in training (fresher patterns)
#   - Reserve Sep-Dec 2024 for calibration (closest to test regime)
#   - Reduce validation to 2 months (sufficient for early stopping)
#
# All v10 scripts should import from settings_v10, not settings.
