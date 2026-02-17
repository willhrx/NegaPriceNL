"""
Train XGBoost Model v6 for Negative Price Prediction

Same data pipeline as v5 (D-1 12:00 CET safe features, 15-min resolution)
but uses real XGBoost (XGBClassifier) instead of sklearn HistGradientBoosting.

Usage:
    py scripts/train_model_v6.py
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
from sklearn.metrics import f1_score
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE,
    TRAIN_END_DATE, TEST_START_DATE,
)
from src.models.xgb_classifier import NegativePriceXGBClassifier
from src.models.threshold_optimizer import optimize_threshold, find_best_threshold

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v6"

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

    target_col = 'is_negative_price'

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

    # Drop NaN rows
    train_valid = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train, y_train = X_train[train_valid], y_train[train_valid]

    val_valid = ~(X_val.isnull().any(axis=1) | y_val.isnull())
    X_val, y_val = X_val[val_valid], y_val[val_valid]

    test_valid = ~(X_test.isnull().any(axis=1) | y_test.isnull())
    X_test, y_test = X_test[test_valid], y_test[test_valid]

    logger.info(f"  Train: {len(X_train):,} samples ({y_train.mean()*100:.2f}% positive)")
    logger.info(f"  Val:   {len(X_val):,} samples ({y_val.mean()*100:.2f}% positive)")
    logger.info(f"  Test:  {len(X_test):,} samples ({y_test.mean()*100:.2f}% positive)")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def create_optuna_objective(X_train, y_train, X_val, y_val, scale_pos_weight):
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
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'early_stopping_rounds': 20,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        best_threshold, best_f1 = optimize_threshold(y_val.values, y_proba)

        return best_f1

    return objective


def run_optuna_optimization(
    X_train, y_train, X_val, y_val,
    scale_pos_weight: float,
    n_trials: int = 50,
) -> dict:
    logger.info(f"\nRunning Optuna optimization ({n_trials} trials)...")

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    objective = create_optuna_objective(
        X_train, y_train, X_val, y_val, scale_pos_weight
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\nBest trial:")
    logger.info(f"  F1 Score: {study.best_value:.4f}")
    logger.info(f"  Best params:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    return study.best_params


def train_final_model(
    X_train, y_train, X_val, y_val,
    best_params: dict,
    scale_pos_weight: float,
) -> NegativePriceXGBClassifier:
    logger.info(f"\nTraining final model {MODEL_VERSION} with best parameters...")

    final_params = best_params.copy()
    final_params['objective'] = 'binary:logistic'
    final_params['eval_metric'] = 'logloss'
    final_params['tree_method'] = 'hist'
    final_params['random_state'] = RANDOM_STATE
    final_params['n_jobs'] = -1

    model = NegativePriceXGBClassifier(
        params=final_params,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=50, verbose=True)

    # Optimize threshold on validation set
    y_val_proba = model.predict_proba(X_val)
    best_threshold, best_metrics = find_best_threshold(
        y_val.values, y_val_proba,
        min_precision=0.5, min_recall=0.5,
    )

    logger.info(f"\nOptimal threshold: {best_threshold:.3f}")
    logger.info(f"  Validation F1: {best_metrics['f1']:.4f}")
    logger.info(f"  Validation Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Validation Recall: {best_metrics['recall']:.4f}")

    model.set_threshold(best_threshold)
    return model


def evaluate_on_test(model: NegativePriceXGBClassifier, X_test, y_test) -> dict:
    logger.info("\n" + "=" * 70)
    logger.info(f"TEST SET EVALUATION ({MODEL_VERSION} - XGBoost, D-1 Safe, 15-min)")
    logger.info("=" * 70)

    metrics = model.evaluate(X_test, y_test, verbose=True)

    logger.info("\nTarget Achievement:")
    logger.info(f"  Recall target (>75%):    {'PASS' if metrics['recall'] > 0.75 else 'FAIL'} ({metrics['recall']*100:.1f}%)")
    logger.info(f"  Precision target (>60%): {'PASS' if metrics['precision'] > 0.60 else 'FAIL'} ({metrics['precision']*100:.1f}%)")
    logger.info(f"  F1 target (>0.67):       {'PASS' if metrics['f1'] > 0.67 else 'FAIL'} ({metrics['f1']:.3f})")

    return metrics


def save_artifacts(
    model: NegativePriceXGBClassifier,
    metrics: dict,
    feature_cols: list,
    X_val,
    y_val,
):
    logger.info(f"\nSaving model {MODEL_VERSION} artifacts...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"gradient_boost_negative_price_{MODEL_VERSION}.pkl"
    model.save(model_path)
    logger.info(f"  Model saved to: {model_path}")

    threshold_path = MODELS_DIR / f"optimal_threshold_{MODEL_VERSION}.pkl"
    joblib.dump(model.threshold, threshold_path)
    logger.info(f"  Threshold saved to: {threshold_path}")

    logger.info("  Computing permutation importance...")
    importance = model.compute_permutation_importance(X_val, y_val, n_repeats=5)
    importance_path = MODELS_DIR / f"feature_importance_{MODEL_VERSION}.csv"
    importance.to_csv(importance_path, index=False)
    logger.info(f"  Feature importance saved to: {importance_path}")

    # Also save native XGBoost feature importance
    native_importance = model.get_feature_importance()
    native_path = MODELS_DIR / f"feature_importance_native_{MODEL_VERSION}.csv"
    native_importance.to_csv(native_path, index=False)
    logger.info(f"  Native importance saved to: {native_path}")

    metrics_path = MODELS_DIR / f"test_metrics_{MODEL_VERSION}.pkl"
    joblib.dump(metrics, metrics_path)
    logger.info(f"  Metrics saved to: {metrics_path}")

    features_path = MODELS_DIR / f"feature_columns_{MODEL_VERSION}.pkl"
    joblib.dump(feature_cols, features_path)
    logger.info(f"  Feature columns saved to: {features_path}")

    return importance


def main():
    logger.info("=" * 70)
    logger.info(f"NEGAPRICENL MODEL TRAINING {MODEL_VERSION} (XGBoost)")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")
    logger.info("Using: XGBClassifier (xgboost library)")
    logger.info("Features: 55 D-1 12:00 CET safe features (same as v5)")

    try:
        df = load_feature_matrix()

        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(df)

        logger.info(f"\nFeature columns ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            logger.info(f"  {i:2d}. {col}")

        # Leakage check
        actual_cols = ['solar_generation_mw', 'wind_generation_mw', 'load_mw',
                       'price_eur_mwh', 'flow_NL_DE_mw']
        leaked = [c for c in actual_cols if c in feature_cols]
        if leaked:
            raise ValueError(f"LEAKAGE DETECTED: {leaked} in feature columns!")
        logger.info("\nLeakage check: PASSED (no delivery-day actuals in features)")

        # Class weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos
        logger.info(f"\nClass imbalance — scale_pos_weight: {scale_pos_weight:.2f}")

        # Optuna
        best_params = run_optuna_optimization(
            X_train, y_train, X_val, y_val,
            scale_pos_weight,
            n_trials=50,
        )

        # Train final model
        model = train_final_model(
            X_train, y_train, X_val, y_val,
            best_params, scale_pos_weight,
        )

        # Evaluate
        metrics = evaluate_on_test(model, X_test, y_test)

        # Save
        importance = save_artifacts(model, metrics, feature_cols, X_val, y_val)

        # Top features
        logger.info("\nTop 15 Most Important Features (permutation):")
        for i, row in importance.head(15).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

        # v5 comparison
        v5_metrics_path = MODELS_DIR / "test_metrics_v5.pkl"
        if v5_metrics_path.exists():
            v5_metrics = joblib.load(v5_metrics_path)
            logger.info("\n" + "-" * 70)
            logger.info("v5 (HistGradientBoosting) vs v6 (XGBoost) COMPARISON")
            logger.info("-" * 70)
            logger.info(f"  {'Metric':<20} {'v5':>10} {'v6':>10} {'Delta':>10}")
            logger.info(f"  {'Recall':<20} {v5_metrics['recall']*100:>9.1f}% {metrics['recall']*100:>9.1f}% {(metrics['recall']-v5_metrics['recall'])*100:>+9.1f}%")
            logger.info(f"  {'Precision':<20} {v5_metrics['precision']*100:>9.1f}% {metrics['precision']*100:>9.1f}% {(metrics['precision']-v5_metrics['precision'])*100:>+9.1f}%")
            logger.info(f"  {'F1':<20} {v5_metrics['f1']:>10.3f} {metrics['f1']:>10.3f} {metrics['f1']-v5_metrics['f1']:>+10.3f}")

        # v4 comparison
        v4_metrics_path = MODELS_DIR / "test_metrics_v4.pkl"
        if v4_metrics_path.exists():
            v4_metrics = joblib.load(v4_metrics_path)
            logger.info("\n" + "-" * 70)
            logger.info("v4 (HistGradientBoosting, hourly) vs v6 (XGBoost, 15-min) COMPARISON")
            logger.info("-" * 70)
            logger.info(f"  {'Metric':<20} {'v4':>10} {'v6':>10} {'Delta':>10}")
            logger.info(f"  {'Recall':<20} {v4_metrics['recall']*100:>9.1f}% {metrics['recall']*100:>9.1f}% {(metrics['recall']-v4_metrics['recall'])*100:>+9.1f}%")
            logger.info(f"  {'Precision':<20} {v4_metrics['precision']*100:>9.1f}% {metrics['precision']*100:>9.1f}% {(metrics['precision']-v4_metrics['precision'])*100:>+9.1f}%")
            logger.info(f"  {'F1':<20} {v4_metrics['f1']:>10.3f} {metrics['f1']:>10.3f} {metrics['f1']-v4_metrics['f1']:>+10.3f}")

        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING {MODEL_VERSION} COMPLETE")
        logger.info("=" * 70)
        logger.info(f"End: {datetime.now()}")

        logger.info(f"\n{MODEL_VERSION} Model Summary:")
        logger.info(f"  Algorithm: XGBClassifier")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Threshold: {model.threshold:.3f}")
        logger.info(f"  Recall: {metrics['recall']*100:.1f}%")
        logger.info(f"  Precision: {metrics['precision']*100:.1f}%")
        logger.info(f"  F1: {metrics['f1']:.3f}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
