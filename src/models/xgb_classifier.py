"""
XGBoost Classifier for Negative Price Prediction

Uses the actual xgboost library (XGBClassifier) for binary classification.
Same interface as NegativePriceXGBoost for compatibility with MLStrategy
and the economic backtesting pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import RANDOM_STATE


class NegativePriceXGBClassifier:
    """
    XGBoost classifier wrapper for negative electricity price prediction.

    Handles:
    - Class imbalance via scale_pos_weight
    - Custom classification threshold
    - Native feature importance + permutation importance
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5,
        scale_pos_weight: Optional[float] = None,
    ):
        self.params = params or self._default_params()
        self.threshold = threshold
        self.scale_pos_weight = scale_pos_weight
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None

    def _default_params(self) -> Dict[str, Any]:
        return {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'min_child_weight': 20,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
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
    ) -> 'NegativePriceXGBClassifier':
        self.feature_names = list(X_train.columns)

        if self.scale_pos_weight is None:
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()
            self.scale_pos_weight = n_negative / n_positive
            if verbose:
                print(f"Calculated scale_pos_weight: {self.scale_pos_weight:.2f}")

        params = self.params.copy()
        params['scale_pos_weight'] = self.scale_pos_weight

        if X_val is not None and y_val is not None:
            params['early_stopping_rounds'] = early_stopping_rounds

        self.model = xgb.XGBClassifier(**params)

        if verbose:
            print(f"Training XGBClassifier with {len(X_train)} samples...")

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['verbose'] = False

        self.model.fit(X_train, y_train, **fit_kwargs)

        if verbose:
            n_trees = self.model.get_booster().num_boosted_rounds()
            print(f"Training complete. Trees: {n_trees}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True,
    ) -> Dict[str, float]:
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'average_precision': average_precision_score(y, y_proba),
            'threshold': self.threshold,
        }

        if verbose:
            print(f"\nModel Evaluation (threshold={self.threshold:.3f}):")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  Avg Precision (PR-AUC): {metrics['average_precision']:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y, y_pred, target_names=['Normal', 'Negative']))

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)

        return importance_df

    def compute_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': result.importances_mean,
            'importance_std': result.importances_std,
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        model_data = {
            'model': self.model,
            'params': self.params,
            'threshold': self.threshold,
            'scale_pos_weight': self.scale_pos_weight,
            'feature_names': self.feature_names,
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Path) -> 'NegativePriceXGBClassifier':
        model_data = joblib.load(path)

        instance = cls(
            params=model_data['params'],
            threshold=model_data['threshold'],
            scale_pos_weight=model_data.get('scale_pos_weight'),
        )
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']

        return instance
