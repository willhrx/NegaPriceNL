"""
Train Gradient Boosting Model v2 for Negative Price Prediction

This version optimizes for RECALL instead of F1, using:
1. F2 score (weights recall 2x higher than precision)
2. Lower classification threshold
3. Relaxed precision requirements

Goal: Recall >70% with Precision >50%
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report
)
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE,
    TRAIN_END_DATE, VALIDATION_START_DATE, TEST_START_DATE
)
from src.models.xgboost_model import NegativePriceXGBoost
from src.models.threshold_optimizer import optimize_threshold, find_best_threshold, fbeta_score

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Features to exclude from training (target, metadata, raw price)
EXCLUDE_COLUMNS = [
    'price_eur_mwh',      # Raw price (would leak target)
    'is_negative_price',  # Target variable
    'Actual Aggregated',  # Unclear column
]


def load_feature_matrix() -> pd.DataFrame:
    """Load the feature matrix."""
    input_path = PROCESSED_DATA_DIR / "feature_matrix.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {input_path}. "
            "Run create_feature_matrix.py first."
        )

    logger.info(f"Loading feature matrix from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    logger.info(f"  Loaded {len(df):,} records with {len(df.columns)} columns")

    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare train/val/test splits based on time.

    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    """
    logger.info("Preparing train/val/test splits...")

    # Define feature columns (exclude target and metadata)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_COLUMNS]
    logger.info(f"  Using {len(feature_cols)} features")

    # Target variable
    target_col = 'is_negative_price'

    # Time-based splits
    train_mask = df.index <= pd.Timestamp(TRAIN_END_DATE, tz='UTC')
    val_mask = (df.index > pd.Timestamp(TRAIN_END_DATE, tz='UTC')) & \
               (df.index < pd.Timestamp(TEST_START_DATE, tz='UTC'))
    test_mask = df.index >= pd.Timestamp(TEST_START_DATE, tz='UTC')

    # Split data
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]

    X_val = df.loc[val_mask, feature_cols]
    y_val = df.loc[val_mask, target_col]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    # Drop any remaining NaN rows
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


def create_optuna_objective_v2(X_train, y_train, X_val, y_val, class_weight):
    """Create Optuna objective function optimizing for F2 score (recall-weighted)."""

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
            'max_bins': trial.suggest_categorical('max_bins', [63, 127, 255]),
            'class_weight': class_weight,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'random_state': RANDOM_STATE,
        }

        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        # Get predictions and optimize threshold for F2 (recall-weighted)
        y_proba = model.predict_proba(X_val)[:, 1]
        best_threshold, best_f2 = optimize_threshold(
            y_val.values, y_proba,
            metric='f2',  # F2 weights recall 2x higher than precision
            min_threshold=0.1,  # Start lower to catch more positives
            max_threshold=0.7   # Don't go too high
        )

        return best_f2

    return objective


def run_optuna_optimization_v2(
    X_train, y_train, X_val, y_val,
    class_weight: dict,
    n_trials: int = 50
) -> dict:
    """Run Optuna hyperparameter optimization for v2 (recall-focused)."""
    logger.info(f"\nRunning Optuna optimization v2 ({n_trials} trials)...")
    logger.info("  Optimizing for F2 score (recall-weighted)")

    # Create study
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Create objective
    objective = create_optuna_objective_v2(X_train, y_train, X_val, y_val, class_weight)

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\nBest trial:")
    logger.info(f"  F2 Score: {study.best_value:.4f}")
    logger.info(f"  Best params:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    return study.best_params


def train_final_model_v2(
    X_train, y_train, X_val, y_val,
    best_params: dict,
    class_weight: dict
) -> NegativePriceXGBoost:
    """Train final model v2 with recall-focused threshold."""
    logger.info("\nTraining final model v2 with best parameters...")

    # Add fixed params
    final_params = best_params.copy()
    final_params['class_weight'] = class_weight
    final_params['random_state'] = RANDOM_STATE
    final_params['early_stopping'] = True
    final_params['validation_fraction'] = 0.1
    final_params['n_iter_no_change'] = 20

    # Create and train model
    model = NegativePriceXGBoost(params=final_params, class_weight=class_weight)
    model.fit(X_train, y_train, X_val, y_val, verbose=True)

    # Optimize threshold on validation set - PRIORITIZE RECALL
    y_val_proba = model.predict_proba(X_val)
    best_threshold, best_metrics = find_best_threshold(
        y_val.values, y_val_proba,
        min_precision=0.50,  # Lower precision requirement than v1
        min_recall=0.70      # Higher recall requirement
    )

    logger.info(f"\nOptimal threshold (recall-focused): {best_threshold:.3f}")
    logger.info(f"  Validation F1: {best_metrics['f1']:.4f}")
    logger.info(f"  Validation Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Validation Recall: {best_metrics['recall']:.4f}")

    if 'constraints_met' in best_metrics:
        logger.info(f"  Constraints met: {best_metrics['constraints_met']}")

    model.set_threshold(best_threshold)

    return model


def evaluate_on_test_v2(model: NegativePriceXGBoost, X_test, y_test) -> dict:
    """Evaluate model v2 on test set."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION (v2 - Recall Focused)")
    logger.info("=" * 70)

    metrics = model.evaluate(X_test, y_test, verbose=True)

    # Check against v2 targets (relaxed precision, higher recall)
    logger.info("\nv2 Target Achievement:")
    logger.info(f"  Recall target (>70%):    {'PASS' if metrics['recall'] > 0.70 else 'FAIL'} ({metrics['recall']*100:.1f}%)")
    logger.info(f"  Precision target (>50%): {'PASS' if metrics['precision'] > 0.50 else 'FAIL'} ({metrics['precision']*100:.1f}%)")
    logger.info(f"  F1 (informational):      {metrics['f1']:.3f}")

    return metrics


def save_artifacts_v2(model: NegativePriceXGBoost, metrics: dict, feature_cols: list, X_val, y_val):
    """Save model v2 and related artifacts."""
    logger.info("\nSaving model v2 artifacts...")

    # Ensure output directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model with _v2 suffix
    model_path = MODELS_DIR / "gradient_boost_negative_price_v2.pkl"
    model.save(model_path)
    logger.info(f"  Model saved to: {model_path}")

    # Save threshold separately for easy access
    threshold_path = MODELS_DIR / "optimal_threshold_v2.pkl"
    joblib.dump(model.threshold, threshold_path)
    logger.info(f"  Threshold saved to: {threshold_path}")

    # Compute and save feature importance (permutation-based)
    logger.info("  Computing permutation importance (this may take a moment)...")
    importance = model.compute_permutation_importance(X_val, y_val, n_repeats=5)
    importance_path = MODELS_DIR / "feature_importance_v2.csv"
    importance.to_csv(importance_path, index=False)
    logger.info(f"  Feature importance saved to: {importance_path}")

    # Save metrics
    metrics_path = MODELS_DIR / "test_metrics_v2.pkl"
    joblib.dump(metrics, metrics_path)
    logger.info(f"  Metrics saved to: {metrics_path}")

    # Save feature list
    features_path = MODELS_DIR / "feature_columns_v2.pkl"
    joblib.dump(feature_cols, features_path)
    logger.info(f"  Feature columns saved to: {features_path}")

    return importance


def main():
    """Main training function for v2 (recall-focused) model."""
    logger.info("=" * 70)
    logger.info("NEGAPRICENL MODEL TRAINING v2 (RECALL-FOCUSED)")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")
    logger.info("Using: HistGradientBoostingClassifier (scikit-learn)")
    logger.info("Optimization: F2 score (weights recall 2x higher than precision)")

    try:
        # Load data
        df = load_feature_matrix()

        # Prepare splits
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_data(df)

        # Calculate class weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos
        class_weight = {0: 1.0, 1: scale_pos_weight}
        logger.info(f"\nClass imbalance weight: {scale_pos_weight:.2f}")

        # Run Optuna optimization (F2-focused)
        best_params = run_optuna_optimization_v2(
            X_train, y_train, X_val, y_val,
            class_weight,
            n_trials=50
        )

        # Train final model
        model = train_final_model_v2(
            X_train, y_train, X_val, y_val,
            best_params, class_weight
        )

        # Evaluate on test set
        metrics = evaluate_on_test_v2(model, X_test, y_test)

        # Save artifacts
        importance = save_artifacts_v2(model, metrics, feature_cols, X_val, y_val)

        # Print top features
        logger.info("\nTop 10 Most Important Features:")
        for i, row in importance.head(10).iterrows():
            logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING v2 COMPLETE")
        logger.info("=" * 70)
        logger.info(f"End: {datetime.now()}")

        # Summary comparison
        logger.info("\nv2 Model Summary:")
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
