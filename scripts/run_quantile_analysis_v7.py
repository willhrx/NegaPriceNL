"""
Run Quantile Forecast Analysis and Economic Comparison (v7)

This script:
1. Loads the trained quantile regressor (v7)
2. Generates quantile price forecasts for the 2025 test period
3. Derives P(price<0), P(price<-10), interval widths
4. Bridges to binary decisions for economic comparison with v5/v6
5. Creates quantile-specific visualizations

Usage:
    python scripts/run_quantile_analysis_v7.py
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    TEST_START_DATE,
)
from src.models.quantile_regressor import NegativePriceQuantileRegressor
from src.evaluation.regression_metrics import (
    evaluate_quantile_forecast,
    calibration_table,
)
from src.evaluation.economic_metrics import calculate_economic_value
from src.evaluation.benchmark_strategies import (
    BaseStrategy,
    HeuristicStrategy,
    NeverCurtailStrategy,
    MLStrategy,
)
from src.evaluation.backtester import EconomicBacktester

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_VERSION = "v7"
ASSET_CAPACITY_MW = 10.0
HOURS_PER_PERIOD = 0.25


# === Benchmark strategies (same as v5/v6) ===

class NaiveStrategyV5(BaseStrategy):
    def __init__(self):
        super().__init__(name="Naive (D-2 Same QH)")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df['is_negative_d2_same_qh'].fillna(0).astype(int).values


class SolarThresholdStrategyV5(BaseStrategy):
    def __init__(self, res_threshold: float = 0.5):
        super().__init__(name=f"Solar Threshold (Forecast RES>{res_threshold*100:.0f}% + Low Load)")
        self.res_threshold = res_threshold

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        median_load = df['load_forecast_mw'].median()
        is_risky = (
            (df['forecast_res_penetration'] > self.res_threshold) &
            (df['load_forecast_mw'] < median_load)
        )
        return is_risky.astype(int).values


class QuantileMLStrategy(BaseStrategy):
    """Bridge: quantile regressor → binary decisions for EconomicBacktester."""

    def __init__(self, model: NegativePriceQuantileRegressor, feature_columns: list):
        super().__init__(name=f"Quantile ML ({MODEL_VERSION}, median<0)")
        self.model = model
        self.feature_columns = feature_columns

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_columns]
        return self.model.predict_binary(X, price_threshold=0.0)


def load_test_data() -> pd.DataFrame:
    input_path = PROCESSED_DATA_DIR / "feature_matrix_v5.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {input_path}.")

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    test_start = pd.Timestamp(TEST_START_DATE, tz='UTC')
    test_df = df[df.index >= test_start].copy()

    for col in ['solar_generation_mw', 'wind_generation_mw', 'load_mw']:
        if col in test_df.columns:
            n_nan = test_df[col].isna().sum()
            if n_nan > 0:
                test_df[col] = test_df[col].fillna(0)

    logger.info(f"  Loaded {len(test_df):,} test records")
    logger.info(f"  Negative price QHs: {test_df['is_negative_price'].sum():,} ({test_df['is_negative_price'].mean()*100:.2f}%)")

    return test_df


def create_visualizations(
    test_df: pd.DataFrame,
    quantile_preds: np.ndarray,
    prob_data: dict,
    model: NegativePriceQuantileRegressor,
    metrics: dict,
):
    """Create quantile-specific visualizations."""
    logger.info("Creating visualizations...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Fan Chart: Actual prices vs prediction intervals (2-week sample)
    fig, ax = plt.subplots(figsize=(16, 6))

    # Pick a 2-week window with negative prices
    neg_dates = test_df[test_df['is_negative_price'] == 1].index
    if len(neg_dates) > 0:
        mid_date = neg_dates[len(neg_dates) // 2]
        start = mid_date - pd.Timedelta(days=7)
        end = mid_date + pd.Timedelta(days=7)
    else:
        start = test_df.index[0]
        end = start + pd.Timedelta(days=14)

    mask = (test_df.index >= start) & (test_df.index <= end)
    idx = test_df.index[mask]
    actual = test_df.loc[mask, 'price_eur_mwh'].values

    # Get quantile indices
    q10_i = model.quantiles.index(0.10)
    q25_i = model.quantiles.index(0.25)
    q50_i = model.quantiles.index(0.50)
    q75_i = model.quantiles.index(0.75)
    q90_i = model.quantiles.index(0.90)

    # Convert mask to positional indices for quantile_preds
    pos_mask = np.where(mask)[0] if isinstance(mask, np.ndarray) else np.where(mask.values)[0]

    ax.fill_between(idx, quantile_preds[pos_mask, q10_i], quantile_preds[pos_mask, q90_i],
                    alpha=0.2, color='steelblue', label='10-90% interval')
    ax.fill_between(idx, quantile_preds[pos_mask, q25_i], quantile_preds[pos_mask, q75_i],
                    alpha=0.3, color='steelblue', label='25-75% interval')
    ax.plot(idx, quantile_preds[pos_mask, q50_i], color='blue', linewidth=1, label='Median forecast')
    ax.plot(idx, actual, color='black', linewidth=0.8, alpha=0.7, label='Actual price')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.set_title(f'Quantile Price Forecast vs Actual ({MODEL_VERSION})\n2-week sample around negative price events')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f'fan_chart_{MODEL_VERSION}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: fan_chart_{MODEL_VERSION}.png")

    # 2. Calibration Plot
    cal = metrics['calibration']
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.scatter(cal['expected_below'], cal['observed_below'], s=100, zorder=5, color='steelblue')
    for _, row in cal.iterrows():
        ax.annotate(f"q{row['alpha']:.2f}", (row['expected_below'], row['observed_below']),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)
    ax.set_xlabel('Expected fraction below quantile')
    ax.set_ylabel('Observed fraction below quantile')
    ax.set_title(f'Quantile Calibration Plot ({MODEL_VERSION})')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f'calibration_plot_{MODEL_VERSION}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: calibration_plot_{MODEL_VERSION}.png")

    # 3. P(negative) heatmap by hour x month
    test_df_copy = test_df.copy()
    test_df_copy['p_negative'] = prob_data['p_negative']
    test_df_copy['hour'] = test_df_copy.index.hour
    test_df_copy['month'] = test_df_copy.index.month

    pivot = test_df_copy.pivot_table(values='p_negative', index='hour', columns='month', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, vmin=0, vmax=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Hour of Day')
    ax.set_title(f'Mean P(price < 0) by Hour and Month ({MODEL_VERSION})\n2025 Test Period')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f'p_negative_heatmap_{MODEL_VERSION}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: p_negative_heatmap_{MODEL_VERSION}.png")

    # 4. Interval width by hour of day
    test_df_copy['interval_width'] = prob_data['interval_width']
    hourly_iw = test_df_copy.groupby('hour')['interval_width'].agg(['mean', 'std'])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hourly_iw.index, hourly_iw['mean'], yerr=hourly_iw['std'], capsize=3,
           color='steelblue', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Prediction Interval Width (EUR/MWh)')
    ax.set_title(f'Prediction Interval Width [q10, q90] by Hour ({MODEL_VERSION})')
    ax.set_xticks(range(24))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f'interval_width_by_hour_{MODEL_VERSION}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: interval_width_by_hour_{MODEL_VERSION}.png")


def main():
    logger.info("=" * 70)
    logger.info(f"NEGAPRICENL QUANTILE ANALYSIS {MODEL_VERSION}")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")

    try:
        # Load model
        model_path = MODELS_DIR / f"quantile_regressor_{MODEL_VERSION}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run train_quantile_model_v7.py first.")

        model = NegativePriceQuantileRegressor.load(model_path)
        feature_cols = joblib.load(MODELS_DIR / f"feature_columns_{MODEL_VERSION}.pkl")
        logger.info(f"Loaded quantile model with {len(model.quantiles)} quantiles: {model.quantiles}")

        # Load test data
        test_df = load_test_data()

        # Generate quantile forecasts
        logger.info("\nGenerating quantile forecasts...")
        X_test = test_df[feature_cols]
        quantile_preds = model.predict(X_test)
        logger.info(f"  Predictions shape: {quantile_preds.shape}")

        # Derive probabilities
        prob_data = model.derive_probabilities(X_test)
        logger.info(f"\n  P(price < 0):  mean={prob_data['p_negative'].mean():.3f}, max={prob_data['p_negative'].max():.3f}")
        logger.info(f"  P(price < -10): mean={prob_data['p_below_minus10'].mean():.3f}")
        logger.info(f"  Median forecast: mean={prob_data['median_forecast'].mean():.1f} EUR/MWh")
        logger.info(f"  Interval width:  mean={prob_data['interval_width'].mean():.1f} EUR/MWh")

        neg_mask = ~np.isnan(prob_data['expected_negative_price'])
        if neg_mask.any():
            logger.info(f"  E[price | price<0]: {prob_data['expected_negative_price'][neg_mask].mean():.1f} EUR/MWh")

        # Full quantile evaluation
        y_test = test_df['price_eur_mwh'].values
        metrics = evaluate_quantile_forecast(y_test, quantile_preds, model.quantiles)

        logger.info(f"\n  CRPS:           {metrics['crps']:.4f}")
        logger.info(f"  Mean QL:        {metrics['mean_quantile_loss']:.4f}")
        logger.info(f"  Coverage (80%): {metrics['coverage_80pct']*100:.1f}%")
        logger.info(f"  Crossing Rate:  {metrics['crossing_rate']*100:.2f}%")

        # === Economic comparison via binary bridge ===
        logger.info("\n" + "=" * 70)
        logger.info("ECONOMIC COMPARISON (Binary Bridge)")
        logger.info("=" * 70)

        strategies = [
            NeverCurtailStrategy(),
            NaiveStrategyV5(),
            HeuristicStrategy(),
            SolarThresholdStrategyV5(res_threshold=0.5),
            QuantileMLStrategy(model, feature_cols),
        ]

        # Also add v6 classifier if available
        v6_model_path = MODELS_DIR / "gradient_boost_negative_price_v6.pkl"
        v6_threshold_path = MODELS_DIR / "optimal_threshold_v6.pkl"
        v6_features_path = MODELS_DIR / "feature_columns_v6.pkl"
        if v6_model_path.exists():
            strategies.append(MLStrategy(
                model_path=v6_model_path,
                threshold_path=v6_threshold_path,
                feature_columns_path=v6_features_path,
                model_version="v6",
            ))

        backtester = EconomicBacktester(
            capacity_mw=ASSET_CAPACITY_MW,
            hours_per_period=HOURS_PER_PERIOD,
        )
        backtester.set_strategies(strategies)

        results = backtester.run(
            test_df,
            price_col='price_eur_mwh',
            generation_col='solar_generation_mw',
            target_col='is_negative_price',
        )

        report = backtester.generate_summary_report(results)
        print("\n" + report)

        # Save reports
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = REPORTS_DIR / f"economic_analysis_{MODEL_VERSION}.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"\nResults saved to: {csv_path}")

        report_path = REPORTS_DIR / f"quantile_report_{MODEL_VERSION}.txt"
        with open(report_path, 'w') as f:
            f.write(f"QUANTILE FORECAST ANALYSIS {MODEL_VERSION}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Quantiles: {model.quantiles}\n\n")

            f.write("QUANTILE METRICS\n")
            f.write(f"  CRPS:           {metrics['crps']:.4f}\n")
            f.write(f"  Mean QL:        {metrics['mean_quantile_loss']:.4f}\n")
            f.write(f"  Coverage (80%): {metrics['coverage_80pct']*100:.1f}%\n")
            f.write(f"  Interval Width: {metrics['mean_width']:.1f} EUR/MWh\n")
            f.write(f"  Crossing Rate:  {metrics['crossing_rate']*100:.2f}%\n\n")

            f.write("PINBALL LOSS BY QUANTILE\n")
            for alpha, loss in metrics['quantile_loss_per_alpha'].items():
                f.write(f"  q{alpha:.2f}: {loss:.4f}\n")

            f.write("\nCALIBRATION\n")
            for _, row in metrics['calibration'].iterrows():
                f.write(f"  q{row['alpha']:.2f}: expected {row['expected_below']*100:.0f}%, observed {row['observed_below']*100:.1f}%\n")

            f.write(f"\nDERIVED PROBABILITIES (test set averages)\n")
            f.write(f"  P(price < 0):    {prob_data['p_negative'].mean():.3f}\n")
            f.write(f"  P(price < -10):  {prob_data['p_below_minus10'].mean():.3f}\n")
            f.write(f"  E[price|neg]:    {prob_data['expected_negative_price'][neg_mask].mean():.1f} EUR/MWh\n")
            f.write(f"  Mean interval:   {prob_data['interval_width'].mean():.1f} EUR/MWh\n")

            f.write("\n\n" + report)

        logger.info(f"Report saved to: {report_path}")

        # Create visualizations
        create_visualizations(test_df, quantile_preds, prob_data, model, metrics)

        logger.info("\n" + "=" * 70)
        logger.info(f"QUANTILE ANALYSIS {MODEL_VERSION} COMPLETE")
        logger.info("=" * 70)

        # Summary
        q_ml = results[results['strategy'].str.contains('Quantile')]
        if len(q_ml) > 0:
            row = q_ml.iloc[0]
            logger.info(f"\nQuantile ML (median<0) Economic Results:")
            logger.info(f"  Savings:      {row['savings_achieved_eur']:>12,.2f} EUR")
            logger.info(f"  Capture Rate: {row['capture_rate_pct']:>11.1f}%")

        return results

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
