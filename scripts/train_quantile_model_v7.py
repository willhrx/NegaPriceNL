"""
Train XGBoost Quantile Regression Model v7

Predicts the 10th, 25th, 50th, 75th, and 90th percentiles of the
day-ahead price distribution for each quarter-hour.

Same feature pipeline as v5/v6 (D-1 12:00 CET safe, 15-min resolution)
but targets continuous price instead of binary is_negative_price.

Usage:
    python scripts/train_quantile_model_v7.py
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
import warnings

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE,
    TRAIN_END_DATE, TEST_START_DATE,
)
from src.models.quantile_regressor import NegativePriceQuantileRegressor, DEFAULT_QUANTILES
from src.evaluation.regression_metrics import (
    evaluate_quantile_forecast,
    mean_quantile_loss,
    quantile_loss_per_alpha,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v7"

EXCLUDE_COLUMNS = [
    'price_eur_mwh',
    'is_negative_price',
    'price_is_15min',
    'solar_generation_mw',
    'wind_generation_mw',
    'load_mw',
    'flow_NL_DE_mw',
]


def load_feature_matrix() -> pd.DataFrame:
    input_path = PROCESSED_DATA_DIR / "feature_matrix_v5.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {input_path}. "
            "Run create_feature_matrix_v5.py first."
        )

    logger.info(f"Loading feature matrix from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    logger.info(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    logger.info("Preparing train/val/test splits...")

    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    logger.info(f"  Using {len(feature_cols)} features")

    # Target: continuous price (not binary)
    target_col = 'price_eur_mwh'

    train_mask = df.index <= pd.Timestamp(TRAIN_END_DATE, tz='UTC')
    val_mask = (df.index > pd.Timestamp(TRAIN_END_DATE, tz='UTC')) & \
               (df.index < pd.Timestamp(TEST_START_DATE, tz='UTC'))
    test_mask = df.index >= pd.Timestamp(TEST_START_DATE, tz='UTC')

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]

    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, target_col]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    # Also keep binary target for comparison
    y_test_binary = df.loc[test_mask, 'is_negative_price']

    # Drop NaN rows
    train_valid = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train, y_train = X_train[train_valid], y_train[train_valid]

    val_valid = ~(X_val.isnull().any(axis=1) | y_val.isnull())
    X_val, y_val = X_val[val_valid], y_val[val_valid]

    test_valid = ~(X_test.isnull().any(axis=1) | y_test.isnull())
    X_test, y_test = X_test[test_valid], y_test[test_valid]
    y_test_binary = y_test_binary[test_valid]

    logger.info(f"  Train: {len(X_train):,} samples (price range: {y_train.min():.1f} to {y_train.max():.1f} EUR/MWh)")
    logger.info(f"  Val:   {len(X_val):,} samples (price range: {y_val.min():.1f} to {y_val.max():.1f} EUR/MWh)")
    logger.info(f"  Test:  {len(X_test):,} samples (price range: {y_test.min():.1f} to {y_test.max():.1f} EUR/MWh)")
    logger.info(f"  Test negative QHs: {y_test_binary.sum():,} ({y_test_binary.mean()*100:.2f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_binary, feature_cols


def create_optuna_objective(X_train, y_train, X_val, y_val, quantiles):
    """Optuna objective: minimize mean quantile loss across all quantiles."""

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        # Train one model per quantile and compute mean pinball loss
        all_preds = np.zeros((len(X_val), len(quantiles)))

        for i, alpha in enumerate(quantiles):
            model_params = params.copy()
            model_params['objective'] = 'reg:quantileerror'
            model_params['quantile_alpha'] = alpha
            model_params['tree_method'] = 'hist'
            model_params['random_state'] = RANDOM_STATE
            model_params['n_jobs'] = -1
            model_params['early_stopping_rounds'] = 20

            model = xgb.XGBRegressor(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            all_preds[:, i] = model.predict(X_val)

        return mean_quantile_loss(y_val.values, all_preds, quantiles)

    return objective


def run_optuna_optimization(
    X_train, y_train, X_val, y_val,
    quantiles: list,
    n_trials: int = 50,
) -> dict:
    logger.info(f"\nRunning Optuna optimization ({n_trials} trials)...")
    logger.info(f"Objective: minimize mean quantile loss across {len(quantiles)} quantiles")

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    objective = create_optuna_objective(X_train, y_train, X_val, y_val, quantiles)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\nBest trial:")
    logger.info(f"  Mean Quantile Loss: {study.best_value:.4f}")
    logger.info(f"  Best params:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    return study.best_params


def train_final_model(
    X_train, y_train, X_val, y_val,
    best_params: dict,
    quantiles: list,
) -> NegativePriceQuantileRegressor:
    logger.info(f"\nTraining final quantile model {MODEL_VERSION}...")

    final_params = best_params.copy()
    final_params['tree_method'] = 'hist'
    final_params['random_state'] = RANDOM_STATE
    final_params['n_jobs'] = -1

    model = NegativePriceQuantileRegressor(
        quantiles=quantiles,
        params=final_params,
    )
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50, verbose=True)

    return model


def evaluate_on_test(
    model: NegativePriceQuantileRegressor,
    X_test, y_test, y_test_binary,
) -> dict:
    logger.info("\n" + "=" * 70)
    logger.info(f"TEST SET EVALUATION ({MODEL_VERSION} - Quantile Regression, 15-min)")
    logger.info("=" * 70)

    # Quantile predictions
    preds = model.predict(X_test)

    # Full quantile evaluation
    metrics = evaluate_quantile_forecast(y_test.values, preds, model.quantiles)

    logger.info(f"\nQuantile Forecast Metrics:")
    logger.info(f"  Mean Quantile Loss: {metrics['mean_quantile_loss']:.4f}")
    logger.info(f"  CRPS:               {metrics['crps']:.4f}")
    logger.info(f"  Coverage (80%):     {metrics['coverage_80pct']*100:.1f}% (expected: {metrics['expected_coverage_80pct']*100:.0f}%)")
    logger.info(f"  Mean Interval Width: {metrics['mean_width']:.1f} EUR/MWh")
    logger.info(f"  Crossing Rate:      {metrics['crossing_rate']*100:.2f}%")

    logger.info(f"\n  Pinball Loss by Quantile:")
    for alpha, loss in metrics['quantile_loss_per_alpha'].items():
        logger.info(f"    q{alpha:.2f}: {loss:.4f}")

    logger.info(f"\n  Calibration (expected vs observed % below):")
    for _, row in metrics['calibration'].iterrows():
        status = "OK" if abs(row['deviation']) < 0.05 else "MISCALIBRATED"
        logger.info(f"    q{row['alpha']:.2f}: expected {row['expected_below']*100:.0f}%, observed {row['observed_below']*100:.1f}% ({status})")

    # Binary comparison: curtail if median < 0
    binary_preds = model.predict_binary(X_test, price_threshold=0.0)
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_test_binary, binary_preds, zero_division=0)
    r = recall_score(y_test_binary, binary_preds, zero_division=0)
    f1 = f1_score(y_test_binary, binary_preds, zero_division=0)

    logger.info(f"\n  Binary Comparison (curtail if median < 0):")
    logger.info(f"    Precision: {p*100:.1f}%")
    logger.info(f"    Recall:    {r*100:.1f}%")
    logger.info(f"    F1:        {f1:.3f}")

    metrics['binary_precision'] = p
    metrics['binary_recall'] = r
    metrics['binary_f1'] = f1

    return metrics


def save_artifacts(
    model: NegativePriceQuantileRegressor,
    metrics: dict,
    feature_cols: list,
    X_val, y_val,
):
    logger.info(f"\nSaving model {MODEL_VERSION} artifacts...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"quantile_regressor_{MODEL_VERSION}.pkl"
    model.save(model_path)
    logger.info(f"  Model saved to: {model_path}")

    # Feature importance (aggregated across quantiles)
    importance = model.get_feature_importance()
    importance_path = MODELS_DIR / f"feature_importance_{MODEL_VERSION}.csv"
    importance.to_csv(importance_path, index=False)
    logger.info(f"  Feature importance saved to: {importance_path}")

    # Permutation importance on median model
    logger.info("  Computing permutation importance (median model)...")
    perm_importance = model.compute_permutation_importance(X_val, y_val, n_repeats=5)
    perm_path = MODELS_DIR / f"feature_importance_perm_{MODEL_VERSION}.csv"
    perm_importance.to_csv(perm_path, index=False)
    logger.info(f"  Permutation importance saved to: {perm_path}")

    metrics_path = MODELS_DIR / f"test_metrics_{MODEL_VERSION}.pkl"
    joblib.dump(metrics, metrics_path)
    logger.info(f"  Metrics saved to: {metrics_path}")

    features_path = MODELS_DIR / f"feature_columns_{MODEL_VERSION}.pkl"
    joblib.dump(feature_cols, features_path)
    logger.info(f"  Feature columns saved to: {features_path}")

    return importance


def main():
    logger.info("=" * 70)
    logger.info(f"NEGAPRICENL QUANTILE MODEL TRAINING {MODEL_VERSION}")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")
    logger.info("Using: XGBRegressor with quantile loss (xgboost library)")
    logger.info(f"Quantiles: {DEFAULT_QUANTILES}")
    logger.info("Target: price_eur_mwh (continuous)")
    logger.info("Features: 55 D-1 12:00 CET safe features (same as v5/v6)")

    try:
        df = load_feature_matrix()

        X_train, y_train, X_val, y_val, X_test, y_test, y_test_binary, feature_cols = prepare_data(df)

        # Leakage check
        actual_cols = ['solar_generation_mw', 'wind_generation_mw', 'load_mw',
                       'price_eur_mwh', 'flow_NL_DE_mw']
        leaked = [c for c in actual_cols if c in feature_cols]
        if leaked:
            raise ValueError(f"LEAKAGE DETECTED: {leaked} in feature columns!")
        logger.info("\nLeakage check: PASSED")

        # Optuna
        best_params = run_optuna_optimization(
            X_train, y_train, X_val, y_val,
            DEFAULT_QUANTILES,
            n_trials=50,
        )

        # Train final model
        model = train_final_model(
            X_train, y_train, X_val, y_val,
            best_params, DEFAULT_QUANTILES,
        )

        # Evaluate
        metrics = evaluate_on_test(model, X_test, y_test, y_test_binary)

        # Save
        importance = save_artifacts(model, metrics, feature_cols, X_val, y_val)

        # Top features
        logger.info("\nTop 15 Most Important Features (aggregated):")
        for _, row in importance.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # v6 comparison
        v6_metrics_path = MODELS_DIR / "test_metrics_v6.pkl"
        if v6_metrics_path.exists():
            v6_metrics = joblib.load(v6_metrics_path)
            logger.info("\n" + "-" * 70)
            logger.info("v6 (XGBoost Classifier) vs v7 (Quantile Regressor) — Binary Comparison")
            logger.info("-" * 70)
            logger.info(f"  {'Metric':<20} {'v6':>10} {'v7':>10} {'Delta':>10}")
            logger.info(f"  {'Recall':<20} {v6_metrics['recall']*100:>9.1f}% {metrics['binary_recall']*100:>9.1f}% {(metrics['binary_recall']-v6_metrics['recall'])*100:>+9.1f}%")
            logger.info(f"  {'Precision':<20} {v6_metrics['precision']*100:>9.1f}% {metrics['binary_precision']*100:>9.1f}% {(metrics['binary_precision']-v6_metrics['precision'])*100:>+9.1f}%")
            logger.info(f"  {'F1':<20} {v6_metrics['f1']:>10.3f} {metrics['binary_f1']:>10.3f} {metrics['binary_f1']-v6_metrics['f1']:>+10.3f}")

        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING {MODEL_VERSION} COMPLETE")
        logger.info("=" * 70)
        logger.info(f"End: {datetime.now()}")

        logger.info(f"\n{MODEL_VERSION} Model Summary:")
        logger.info(f"  Algorithm: XGBRegressor (quantile loss)")
        logger.info(f"  Quantiles: {DEFAULT_QUANTILES}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Mean Quantile Loss: {metrics['mean_quantile_loss']:.4f}")
        logger.info(f"  CRPS: {metrics['crps']:.4f}")
        logger.info(f"  Coverage (80%): {metrics['coverage_80pct']*100:.1f}%")
        logger.info(f"  Binary F1 (median<0): {metrics['binary_f1']:.3f}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
