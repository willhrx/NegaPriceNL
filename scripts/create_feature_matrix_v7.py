"""
Create Feature Matrix v7 for NegaPriceNL Model Training

Extends v6 with price-anchoring features:
- price_d1_same_qh (D-1 same quarter-hour price)
- is_negative_d1_same_qh (D-1 negative indicator)
- price_same_hour_7d_mean (7-day rolling same-hour mean)

Pipeline:
1. Load the unified v6 dataset (15-min resolution)
2. Apply v7 feature engineering (v6 + price-anchoring features)
3. Handle missing values
4. Validate features
5. Save feature matrix for model training

Usage:
    python scripts/create_feature_matrix_v7.py
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR, PERIODS_PER_HOUR
from src.features.feature_engine_v7 import NegativePriceFeatureEngineV7

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_unified_dataset() -> pd.DataFrame:
    """Load the v6 unified dataset (Phase 1 uses v6 data sources)."""
    # For Phase 1, we use the existing v6 unified dataset
    # Future phases will use v7 unified dataset with German prices, TTF, EUA
    input_path = PROCESSED_DATA_DIR / "unified_dataset_v6.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Unified v6 dataset not found at {input_path}. "
            "Run data_preparation_v6.py first."
        )

    logger.info(f"Loading unified v6 dataset from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    logger.info(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def handle_missing_values(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Handle missing values in the feature matrix.

    Strategy:
    - Drop rows where target is missing
    - Drop first 672 rows (7-day warmup for D-7 lag)
    - Forward-fill forecast and NTC gaps (max 6 QH = 1.5h)
    - Forward-fill snapshot features (max 96 QH = 1 day)
    - Drop rows with NaN in critical features
    """
    logger.info("Handling missing values...")
    initial_rows = len(df)

    # Drop rows without target
    df = df.dropna(subset=['is_negative_price'])

    # Drop warmup rows (7 days = 672 QH periods for D-7 lag)
    # Increased from 672 to account for 8-day lookback in price_same_hour_7d_mean
    warmup = 192 * PERIODS_PER_HOUR  # 8 days = 768 QH periods
    if len(df) > warmup:
        df = df.iloc[warmup:]
        logger.info(f"  Dropped {warmup} warmup rows for 8-day lookback")

    # Forward-fill weather forecast gaps (max 6 QH = 1.5 hours)
    weather_cols = [c for c in df.columns if c.startswith('forecast_') and c in feature_cols]
    if weather_cols:
        df[weather_cols] = df[weather_cols].ffill(limit=6)

    # Forward-fill NTC features (max 6 QH = 1.5 hours, same as forecast)
    ntc_cols = [c for c in feature_cols if c.startswith('ntc_') and c in df.columns]
    if ntc_cols:
        df[ntc_cols] = df[ntc_cols].ffill(limit=6)

    # Forward-fill snapshot features (1 day max = 96 periods)
    snapshot_cols = [c for c in feature_cols if c.startswith('price_d1') or
                     c.startswith('price_d2') or c.startswith('neg_count') or
                     c.endswith('_d2_mean') or c.endswith('_vs_d2') or
                     c == 'price_same_hour_7d_mean']
    existing_snap = [c for c in snapshot_cols if c in df.columns]
    if existing_snap:
        df[existing_snap] = df[existing_snap].ffill(limit=96)

    # Log remaining NaN in feature columns
    logger.info("  NaN in feature columns after fill:")
    for col in feature_cols:
        if col in df.columns:
            nan_ct = df[col].isna().sum()
            if nan_ct > 0:
                logger.info(f"    {col}: {nan_ct:,} ({nan_ct / len(df) * 100:.1f}%)")

    # Drop rows with NaN in critical columns
    critical = ['price_eur_mwh', 'is_negative_price']
    existing_critical = [c for c in critical if c in df.columns]
    df = df.dropna(subset=existing_critical)

    final_rows = len(df)
    dropped = initial_rows - final_rows
    logger.info(f"  Total dropped: {dropped:,} ({dropped / initial_rows * 100:.1f}%)")
    logger.info(f"  Final: {final_rows:,} rows")

    return df


def validate_features(df: pd.DataFrame, engine: NegativePriceFeatureEngineV7):
    """Validate the feature matrix."""
    logger.info("Validating feature matrix...")

    expected = engine.get_feature_columns()
    present = [f for f in expected if f in df.columns]
    missing = [f for f in expected if f not in df.columns]

    logger.info(f"  Expected features: {len(expected)}")
    logger.info(f"  Present features: {len(present)}")
    if missing:
        logger.warning(f"  Missing features: {missing}")

    # Range checks for new v7 features
    if 'price_d1_same_qh' in df.columns:
        d1_price = df['price_d1_same_qh']
        logger.info(f"  price_d1_same_qh: [{d1_price.min():.1f}, {d1_price.max():.1f}] EUR/MWh")
        logger.info(f"    NaN count: {d1_price.isna().sum():,}")

    if 'price_same_hour_7d_mean' in df.columns:
        hour_mean = df['price_same_hour_7d_mean']
        logger.info(f"  price_same_hour_7d_mean: [{hour_mean.min():.1f}, {hour_mean.max():.1f}] EUR/MWh")
        logger.info(f"    NaN count: {hour_mean.isna().sum():,}")

    # Existing v6 validation checks
    if 'forecast_solar_cf' in df.columns:
        scf = df['forecast_solar_cf']
        logger.info(f"  forecast_solar_cf: [{scf.min():.3f}, {scf.max():.3f}]")

    if 'ntc_nl_total_export_mw' in df.columns:
        ntc = df['ntc_nl_total_export_mw']
        logger.info(f"  ntc_nl_total_export_mw: [{ntc.min():.0f}, {ntc.max():.0f}] MW")

    if 'ntc_export_constrained' in df.columns:
        constrained = df['ntc_export_constrained'].mean() * 100
        logger.info(f"  ntc_export_constrained: {constrained:.1f}% of QHs")

    # Target distribution
    neg_count = df['is_negative_price'].sum()
    neg_pct = neg_count / len(df) * 100
    logger.info(f"  Negative price QHs: {neg_count:,} ({neg_pct:.2f}%)")

    # Remaining NaN
    nan_counts = df[present].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        logger.warning("  Features still with NaN:")
        for col, count in cols_with_nan.items():
            logger.warning(f"    {col}: {count:,}")
    else:
        logger.info("  No NaN in feature columns ✓")


def main():
    logger.info("=" * 70)
    logger.info("NEGAPRICENL FEATURE MATRIX v7 CREATION")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")

    try:
        df = load_unified_dataset()

        engine = NegativePriceFeatureEngineV7()

        logger.info("\nApplying v7 feature engineering...")
        df = engine.transform(df)

        feature_cols = engine.get_feature_columns()
        logger.info(f"  Created {len(feature_cols)} features (v6: 65 + Phase 1: 3):")
        for feat in feature_cols:
            logger.info(f"    - {feat}")

        df = handle_missing_values(df, feature_cols)

        validate_features(df, engine)

        output_path = PROCESSED_DATA_DIR / "feature_matrix_v7.csv"
        df.to_csv(output_path)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
        logger.info(f"Shape: {df.shape}")

        logger.info("\n" + "=" * 70)
        logger.info("FEATURE MATRIX v7 SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Records: {len(df):,}")
        logger.info(f"Columns: {len(df.columns)}")
        logger.info(f"Feature columns: {len(feature_cols)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        logger.info("\nAll columns:")
        for i, col in enumerate(df.columns, 1):
            logger.info(f"  {i:2d}. {col}")

        logger.info("\n" + "=" * 70)
        logger.info("FEATURE MATRIX v7 CREATION COMPLETE")
        logger.info("=" * 70)

        return df

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
