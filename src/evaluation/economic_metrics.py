"""
Economic metrics for evaluating negative price predictions.

This module calculates the business value of predictions for a solar asset
operator who can curtail production during predicted negative price hours.

Business Logic:
- Without prediction: Revenue = Σ(generation × price), including negative terms
- With prediction: Curtail when predicted negative → avoid paying to generate
- Perfect foresight: Curtail only during actual negatives → maximum possible savings
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def calculate_economic_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    generation: np.ndarray,
    capacity_mw: float = 10.0,
) -> Dict:
    """
    Calculate economic value of negative price predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Actual negative price indicators (1 if price < 0, else 0)
    y_pred : np.ndarray
        Predicted negative price indicators (1 if predicted negative, else 0)
    prices : np.ndarray
        Actual electricity prices (EUR/MWh)
    generation : np.ndarray
        Solar generation (MWh) - actual or scaled to capacity
    capacity_mw : float
        Asset capacity in MW (used for normalization)

    Returns
    -------
    dict
        Economic metrics including revenues, savings, and capture rate
    """
    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    prices = np.asarray(prices).flatten()
    generation = np.asarray(generation).flatten()

    # Scale generation to asset capacity if needed
    # (assumes input generation is in MW, convert to MWh for hourly data)
    gen_scaled = generation * (capacity_mw / generation.max()) if generation.max() > 0 else generation

    # 1. Baseline: No curtailment (produce always)
    # Revenue includes negative terms when price < 0
    revenue_no_curtail = np.sum(gen_scaled * prices)

    # 2. Perfect foresight: Curtail only during actual negative hours
    # Revenue = generation × max(price, 0)
    revenue_perfect = np.sum(gen_scaled * np.maximum(prices, 0))

    # 3. With ML prediction: Curtail when predicted negative
    # When y_pred=1, we curtail → no revenue/loss
    # When y_pred=0, we produce → get price (positive or negative)
    revenue_with_pred = np.sum(gen_scaled * prices * (1 - y_pred))

    # Calculate savings
    max_savings = revenue_perfect - revenue_no_curtail  # Theoretical maximum
    actual_savings = revenue_with_pred - revenue_no_curtail  # What we achieved

    # Capture rate: what % of possible savings did we capture?
    capture_rate = (actual_savings / max_savings * 100) if max_savings != 0 else 0

    # Breakdown of costs by confusion matrix quadrant
    costs = calculate_confusion_costs(y_true, y_pred, prices, gen_scaled)

    return {
        'revenue_no_curtailment_eur': revenue_no_curtail,
        'revenue_perfect_foresight_eur': revenue_perfect,
        'revenue_with_prediction_eur': revenue_with_pred,
        'savings_achieved_eur': actual_savings,
        'max_possible_savings_eur': max_savings,
        'capture_rate_pct': capture_rate,
        'total_generation_mwh': np.sum(gen_scaled),
        'hours_analyzed': len(prices),
        'negative_price_hours': int(np.sum(y_true)),
        'predicted_negative_hours': int(np.sum(y_pred)),
        **costs
    }


def calculate_confusion_costs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prices: np.ndarray,
    generation: np.ndarray
) -> Dict:
    """
    Calculate the economic cost/benefit of each confusion matrix quadrant.

    Parameters
    ----------
    y_true : np.ndarray
        Actual negative price indicators
    y_pred : np.ndarray
        Predicted negative price indicators
    prices : np.ndarray
        Actual electricity prices
    generation : np.ndarray
        Generation (scaled)

    Returns
    -------
    dict
        Costs and benefits by prediction outcome
    """
    # True Positives: Correctly predicted negative → curtailed → avoided loss
    tp_mask = (y_true == 1) & (y_pred == 1)
    tp_value = np.sum(generation[tp_mask] * np.abs(prices[tp_mask]))  # Avoided paying
    tp_count = int(np.sum(tp_mask))

    # True Negatives: Correctly predicted positive → produced → captured revenue
    tn_mask = (y_true == 0) & (y_pred == 0)
    tn_value = np.sum(generation[tn_mask] * prices[tn_mask])  # Revenue earned
    tn_count = int(np.sum(tn_mask))

    # False Positives: Predicted negative but was positive → curtailed unnecessarily
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_value = np.sum(generation[fp_mask] * prices[fp_mask])  # Missed revenue
    fp_count = int(np.sum(fp_mask))

    # False Negatives: Predicted positive but was negative → produced → paid to generate
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_value = np.sum(generation[fn_mask] * np.abs(prices[fn_mask]))  # Loss incurred
    fn_count = int(np.sum(fn_mask))

    return {
        'true_positives': tp_count,
        'true_negatives': tn_count,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'value_avoided_losses_eur': tp_value,      # TP: Good - avoided paying
        'value_captured_revenue_eur': tn_value,    # TN: Good - earned revenue
        'cost_missed_revenue_eur': fp_value,       # FP: Bad - missed positive revenue
        'cost_paid_to_generate_eur': fn_value,     # FN: Bad - paid to generate
    }


def calculate_capture_rate(
    revenue_baseline: float,
    revenue_with_strategy: float,
    revenue_perfect: float
) -> float:
    """
    Calculate capture rate: % of possible savings achieved.

    Parameters
    ----------
    revenue_baseline : float
        Revenue without any curtailment
    revenue_with_strategy : float
        Revenue with the prediction strategy
    revenue_perfect : float
        Revenue with perfect foresight

    Returns
    -------
    float
        Capture rate as percentage (0-100)
    """
    max_savings = revenue_perfect - revenue_baseline
    actual_savings = revenue_with_strategy - revenue_baseline

    if max_savings == 0:
        return 0.0

    return (actual_savings / max_savings) * 100


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Actual labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    dict
        Precision, recall, F1 score
    """
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }


def summarize_by_period(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    period: str = 'M'
) -> pd.DataFrame:
    """
    Summarize economic results by time period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index, prices, generation
    y_pred : np.ndarray
        Predicted negative indicators
    period : str
        Resampling period ('D' for daily, 'W' for weekly, 'M' for monthly)

    Returns
    -------
    pd.DataFrame
        Summary statistics by period
    """
    temp = df.copy()
    temp['y_pred'] = y_pred
    temp['revenue_no_curtail'] = temp['generation'] * temp['price']
    temp['revenue_with_pred'] = temp['generation'] * temp['price'] * (1 - temp['y_pred'])

    summary = temp.resample(period).agg({
        'revenue_no_curtail': 'sum',
        'revenue_with_pred': 'sum',
        'generation': 'sum',
        'is_negative_price': 'sum',
        'y_pred': 'sum',
    }).rename(columns={
        'is_negative_price': 'actual_negative_hours',
        'y_pred': 'predicted_negative_hours',
        'generation': 'total_generation_mwh'
    })

    summary['savings'] = summary['revenue_with_pred'] - summary['revenue_no_curtail']

    return summary
