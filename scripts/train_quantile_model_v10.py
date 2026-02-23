"""
Train XGBoost Quantile Regression Model v10 with Conformalized Calibration

v10 extends v9 with Conformalized Quantile Regression (CQR) to address
the persistent calibration bias that features alone could not fix.

v9 Results (feature-based approach):
- CRPS improved 12.9% (17.19 → 14.93) ✓
- Crossing rate improved 4.8pp (33.6% → 28.9%) ✓
- Calibration bias UNCHANGED: q50 70.8% observed vs 50% expected ✗

v10 Approach (conformal calibration):
- Same feature matrix (v7 with 68 features)
- Expanded quantiles: 9 levels (was 5)
- New data splits: dedicated calibration period (Sep-Dec 2024)
- Static + rolling conformal calibration
- Monotonicity enforcement

Expected outcome: q50 calibration converges to 48-52% observed

Usage:
    python scripts/train_quantile_model_v10.py
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
import warnings
import pickle

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_v10 import (
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE,
    TRAIN_END_DATE, VAL_END_DATE, CAL_START_DATE, CAL_END_DATE,
    TEST_START_DATE, QUANTILES_V10, CONFORMAL_WINDOW_DAYS,
)
from src.models.quantile_regressor import NegativePriceQuantileRegressor
from src.models.conformal_calibrator import ConformalCalibrator
from src.models.rolling_conformal import RollingConformalCalibrator
from src.evaluation.regression_metrics import (
    evaluate_quantile_forecast,
    mean_quantile_loss,
    compare_calibration,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v10"

EXCLUDE_COLUMNS = [
    'price_eur_mwh',
    'is_negative_price',
    'price_is_15min',
    'solar_generation_mw',
    'wind_generation_mw',
    'load_mw',
    'flow_NL_DE_mw',
    # Raw NTC per-direction columns
    'ntc_da_nl_be_mw', 'ntc_da_be_nl_mw',
    'ntc_da_nl_delu_mw', 'ntc_da_delu_nl_mw',
    'ntc_da_nl_gb_mw', 'ntc_da_gb_nl_mw',
    'ntc_da_nl_no2_mw', 'ntc_da_no2_nl_mw',
]


def load_feature_matrix() -> pd.DataFrame:
    """Load v7 feature matrix (same as v9)."""
    input_path = PROCESSED_DATA_DIR / "feature_matrix_v7.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {input_path}. "
            "Run create_feature_matrix_v7.py first."
        )

    logger.info(f"Loading feature matrix from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    logger.info(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare train/val/cal/test splits for v10.

    New splits compared to v9:
    - Train: extended to Jun 2024 (was Dec 2023)
    - Val: Jul-Aug 2024 (was Jan-Dec 2024)
    - Cal: Sep-Dec 2024 (NEW: dedicated calibration period)
    - Test: 2025 (unchanged)
    """
    logger.info("Preparing train/val/cal/test splits...")

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    logger.info(f"  Using {len(feature_cols)} features")
    logger.info(f"  Target: price_eur_mwh")
    logger.info(f"  Quantiles: {QUANTILES_V10}")

    target_col = 'price_eur_mwh'

    # Define splits
    train_mask = df.index <= TRAIN_END_DATE
    val_mask = (df.index > TRAIN_END_DATE) & (df.index <= VAL_END_DATE)
    cal_mask = (df.index >= CAL_START_DATE) & (df.index <= CAL_END_DATE)
    test_mask = df.index >= TEST_START_DATE

    # Extract data
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]

    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, target_col]

    X_cal = df.loc[cal_mask, feature_cols]
    y_cal = df.loc[cal_mask, target_col]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    logger.info(f"\n  Train: {len(X_train):,} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
    logger.info(f"  Val:   {len(X_val):,} samples ({X_val.index.min().date()} to {X_val.index.max().date()})")
    logger.info(f"  Cal:   {len(X_cal):,} samples ({X_cal.index.min().date()} to {X_cal.index.max().date()})")
    logger.info(f"  Test:  {len(X_test):,} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")

    return X_train, y_train, X_val, y_val, X_cal, y_cal, X_test, y_test, feature_cols


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50
) -> dict:
    """Optimize hyperparameters using Optuna (median quantile)."""
    logger.info(f"\nOptimizing hyperparameters ({n_trials} trials)...")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'objective': 'reg:quantileerror',
            'quantile_alpha': 0.50,
            'early_stopping_rounds': 50,
            'random_state': RANDOM_STATE,
            'tree_method': 'hist',
            'n_jobs': -1,
        }

        model = xgb.XGBRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        # Quantile loss for α=0.5
        errors = y_val.values - preds
        loss = np.where(errors >= 0, 0.5 * errors, (0.5 - 1) * errors).mean()

        return loss

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\n  Best trial score: {study.best_trial.value:.4f}")
    logger.info(f"  Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    return study.best_params


def main():
    logger.info("=" * 70)
    logger.info("NEGAPRICENL QUANTILE MODEL v10 TRAINING (CONFORMALIZED)")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")

    try:
        # Load data
        df = load_feature_matrix()

        # Prepare splits
        X_train, y_train, X_val, y_val, X_cal, y_cal, X_test, y_test, feature_cols = prepare_data(df)

        # Optimize hyperparameters
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50)

        # Train final model with expanded quantile set
        logger.info(f"\nTraining final quantile model v10...")
        model = NegativePriceQuantileRegressor(quantiles=QUANTILES_V10, params=best_params)
        model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50, verbose=True)

        # ==================== RAW PREDICTIONS ====================

        logger.info("\nGenerating raw (uncalibrated) predictions...")

        # Predict with monotonicity enforcement
        raw_preds_cal = model.predict(X_cal, enforce_monotonicity=True)
        raw_preds_test = model.predict(X_test, enforce_monotonicity=True)

        logger.info(f"  Calibration set: {raw_preds_cal.shape}")
        logger.info(f"  Test set: {raw_preds_test.shape}")

        # ==================== STATIC CONFORMAL CALIBRATION ====================

        logger.info("\n" + "=" * 70)
        logger.info("STATIC CONFORMAL CALIBRATION")
        logger.info("=" * 70)

        static_calibrator = ConformalCalibrator(quantiles=QUANTILES_V10)
        static_calibrator.calibrate(y_cal.values, raw_preds_cal)

        corrections_summary = static_calibrator.get_corrections_summary()
        logger.info("\nStatic calibration corrections:")
        logger.info(corrections_summary.to_string(index=False))

        # Apply static calibration
        cal_preds_cal_static = static_calibrator.apply(raw_preds_cal)
        cal_preds_test_static = static_calibrator.apply(raw_preds_test)

        # ==================== ROLLING CONFORMAL CALIBRATION ====================

        logger.info("\n" + "=" * 70)
        logger.info("ROLLING CONFORMAL CALIBRATION (30-DAY WINDOW)")
        logger.info("=" * 70)

        # Concatenate all data for rolling calibration
        # (roller needs access to pre-test data for calibration windows)
        X_all = pd.concat([X_train, X_val, X_cal, X_test])
        y_all = pd.concat([y_train, y_val, y_cal, y_test])
        preds_all_raw = model.predict(X_all, enforce_monotonicity=True)

        # Convert static corrections to dict for fallback
        static_corrections_dict = static_calibrator.corrections

        # Apply rolling calibration
        roller = RollingConformalCalibrator(quantiles=QUANTILES_V10, window_days=CONFORMAL_WINDOW_DAYS)
        preds_all_cal_rolling = roller.calibrate_and_predict(
            y_all.values,
            preds_all_raw,
            X_all.index,
            TEST_START_DATE,
            static_corrections=static_corrections_dict
        )

        # Extract test period
        test_mask = X_all.index >= TEST_START_DATE
        cal_preds_test_rolling = preds_all_cal_rolling[test_mask]

        # ==================== EVALUATION ====================

        logger.info("\n" + "=" * 70)
        logger.info("TEST SET EVALUATION (v10 - Conformalized QR)")
        logger.info("=" * 70)

        # Raw predictions
        raw_metrics = evaluate_quantile_forecast(y_test.values, raw_preds_test, QUANTILES_V10)

        logger.info("\nRAW (Uncalibrated) Predictions:")
        logger.info(f"  Mean Quantile Loss: {raw_metrics['mean_quantile_loss']:.4f}")
        logger.info(f"  CRPS:               {raw_metrics['crps']:.4f}")
        logger.info(f"  Coverage (80%):     {raw_metrics['coverage_80pct']*100:.1f}%")
        logger.info(f"  Crossing Rate:      {raw_metrics['crossing_rate']*100:.2f}%")

        # Static calibration
        static_metrics = evaluate_quantile_forecast(y_test.values, cal_preds_test_static, QUANTILES_V10)

        logger.info("\nSTATIC Calibration:")
        logger.info(f"  Mean Quantile Loss: {static_metrics['mean_quantile_loss']:.4f}")
        logger.info(f"  CRPS:               {static_metrics['crps']:.4f}")
        logger.info(f"  Coverage (80%):     {static_metrics['coverage_80pct']*100:.1f}%")
        logger.info(f"  Crossing Rate:      {static_metrics['crossing_rate']*100:.2f}%")

        # Rolling calibration
        rolling_metrics = evaluate_quantile_forecast(y_test.values, cal_preds_test_rolling, QUANTILES_V10)

        logger.info("\nROLLING Calibration (30-day window):")
        logger.info(f"  Mean Quantile Loss: {rolling_metrics['mean_quantile_loss']:.4f}")
        logger.info(f"  CRPS:               {rolling_metrics['crps']:.4f}")
        logger.info(f"  Coverage (80%):     {rolling_metrics['coverage_80pct']*100:.1f}%")
        logger.info(f"  Crossing Rate:      {rolling_metrics['crossing_rate']*100:.2f}%")

        # ==================== CALIBRATION COMPARISON ====================

        logger.info("\n" + "=" * 70)
        logger.info("CALIBRATION COMPARISON (Raw vs Static vs Rolling)")
        logger.info("=" * 70)

        comp_static = compare_calibration(y_test.values, raw_preds_test, cal_preds_test_static, QUANTILES_V10)
        comp_rolling = compare_calibration(y_test.values, raw_preds_test, cal_preds_test_rolling, QUANTILES_V10)

        logger.info("\nRaw vs Static:")
        logger.info(comp_static.to_string(index=False))

        logger.info("\nRaw vs Rolling:")
        logger.info(comp_rolling.to_string(index=False))

        # ==================== SAVE ARTIFACTS ====================

        logger.info("\n" + "=" * 70)
        logger.info("SAVING MODEL ARTIFACTS")
        logger.info("=" * 70)

        # Save model
        model_path = MODELS_DIR / f"quantile_regressor_{MODEL_VERSION}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"  Model saved to: {model_path}")

        # Save static calibrator
        calibrator_path = MODELS_DIR / f"static_calibrator_{MODEL_VERSION}.pkl"
        static_calibrator.save(calibrator_path)
        logger.info(f"  Static calibrator saved to: {calibrator_path}")

        # Save test metrics
        metrics_path = MODELS_DIR / f"test_metrics_{MODEL_VERSION}.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump({
                'raw': raw_metrics,
                'static': static_metrics,
                'rolling': rolling_metrics,
                'calibration_comparison_static': comp_static,
                'calibration_comparison_rolling': comp_rolling,
                'quantiles': QUANTILES_V10,
            }, f)
        logger.info(f"  Test metrics saved to: {metrics_path}")

        # Save feature columns
        feature_cols_path = MODELS_DIR / f"feature_columns_{MODEL_VERSION}.pkl"
        joblib.dump(feature_cols, feature_cols_path)
        logger.info(f"  Feature columns saved to: {feature_cols_path}")

        # Save predictions
        preds_path = MODELS_DIR / f"test_predictions_{MODEL_VERSION}.pkl"
        with open(preds_path, 'wb') as f:
            pickle.dump({
                'y_test': y_test.values,
                'index_test': X_test.index,
                'raw_preds': raw_preds_test,
                'cal_preds_static': cal_preds_test_static,
                'cal_preds_rolling': cal_preds_test_rolling,
            }, f)
        logger.info(f"  Test predictions saved to: {preds_path}")

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING v10 COMPLETE")
        logger.info("=" * 70)
        logger.info(f"End: {datetime.now()}")

        logger.info(f"\nModel Summary:")
        logger.info(f"  Algorithm: XGBRegressor (quantile loss)")
        logger.info(f"  Quantiles: {QUANTILES_V10}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Raw CRPS: {raw_metrics['crps']:.4f}")
        logger.info(f"  Rolling Calibrated CRPS: {rolling_metrics['crps']:.4f}")

        return model, static_calibrator

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
