"""
Create Feature Matrix for NegaPriceNL Model Training

This script:
1. Loads the unified dataset
2. Applies feature engineering transformations
3. Handles missing values appropriately
4. Saves the feature matrix for model training
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from src.features.feature_engine import NegativePriceFeatureEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_unified_dataset() -> pd.DataFrame:
    """Load the unified dataset from processed data directory."""
    input_path = PROCESSED_DATA_DIR / "unified_dataset.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Unified dataset not found at {input_path}. "
            "Run data_preparation.py first."
        )

    logger.info(f"Loading unified dataset from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    logger.info(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the feature matrix.

    Strategy:
    - Drop rows where target variable is missing
    - Drop rows where lag features would be invalid (first 168 hours)
    - Forward-fill weather features (short gaps only)
    """
    logger.info("Handling missing values...")

    initial_rows = len(df)

    # Check missing values before
    missing_before = df.isnull().sum()
    cols_with_missing = missing_before[missing_before > 0]
    if len(cols_with_missing) > 0:
        logger.info("  Missing values before handling:")
        for col, count in cols_with_missing.items():
            pct = (count / len(df)) * 100
            logger.info(f"    {col}: {count} ({pct:.2f}%)")

    # Drop rows where target is missing
    df = df.dropna(subset=['is_negative_price'])

    # Drop rows where critical lag features are missing (first 168 hours for weekly lags)
    # This is expected behavior - we need history for lag features
    lag_cols = [col for col in df.columns if 'lag_168h' in col]
    if lag_cols:
        df = df.dropna(subset=lag_cols)

    # Forward-fill weather features (for small gaps only)
    weather_cols = ['ghi_wm2', 'wind_speed_100m_ms', 'temperature_2m_c',
                    'cloud_cover_pct', 'pressure_hpa', 'humidity_pct']
    existing_weather = [col for col in weather_cols if col in df.columns]
    if existing_weather:
        df[existing_weather] = df[existing_weather].ffill(limit=6)  # Max 6 hour gap

    # Drop any remaining rows with NaN in critical features
    critical_cols = ['price_eur_mwh', 'load_mw', 'solar_generation_mw',
                     'wind_generation_mw', 'is_negative_price']
    existing_critical = [col for col in critical_cols if col in df.columns]
    df = df.dropna(subset=existing_critical)

    final_rows = len(df)
    dropped = initial_rows - final_rows
    logger.info(f"  Dropped {dropped:,} rows ({(dropped/initial_rows)*100:.2f}%)")
    logger.info(f"  Final dataset: {final_rows:,} rows")

    return df


def validate_features(df: pd.DataFrame, feature_engine: NegativePriceFeatureEngine):
    """Validate the feature matrix meets quality standards."""
    logger.info("Validating feature matrix...")

    # Check all expected features exist
    expected_features = feature_engine.get_feature_columns()
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        logger.warning(f"  Missing expected features: {missing_features}")

    # Check feature value ranges
    if 'solar_capacity_factor' in df.columns:
        scf = df['solar_capacity_factor']
        if scf.min() < 0 or scf.max() > 1:
            logger.warning(f"  solar_capacity_factor out of range: [{scf.min():.3f}, {scf.max():.3f}]")
        else:
            logger.info(f"  solar_capacity_factor range: [{scf.min():.3f}, {scf.max():.3f}] OK")

    if 'res_penetration' in df.columns:
        rp = df['res_penetration']
        logger.info(f"  res_penetration range: [{rp.min():.3f}, {rp.max():.3f}]")

    # Check target variable distribution
    if 'is_negative_price' in df.columns:
        neg_count = df['is_negative_price'].sum()
        neg_pct = (neg_count / len(df)) * 100
        logger.info(f"  Negative price events: {neg_count:,} ({neg_pct:.2f}%)")

    # Check for remaining NaN values
    nan_counts = df.isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        logger.warning("  Remaining NaN values:")
        for col, count in cols_with_nan.items():
            logger.warning(f"    {col}: {count}")
    else:
        logger.info("  No NaN values remaining")


def save_feature_matrix(df: pd.DataFrame, output_path: Path):
    """Save the feature matrix to disk."""
    logger.info(f"Saving feature matrix to {output_path}")

    # Save as CSV
    df.to_csv(output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved successfully ({file_size_mb:.2f} MB)")
    logger.info(f"  Shape: {df.shape}")


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("NEGAPRICENL FEATURE MATRIX CREATION")
    logger.info("=" * 70)
    logger.info(f"Start: {datetime.now()}")

    try:
        # Load unified dataset
        df = load_unified_dataset()

        # Initialize feature engine
        feature_engine = NegativePriceFeatureEngine()

        # Apply feature engineering
        logger.info("\nApplying feature engineering...")
        df = feature_engine.transform(df)

        # Get feature columns for reporting
        engineered_features = feature_engine.get_feature_columns()
        logger.info(f"  Created {len(engineered_features)} engineered features:")
        for feat in engineered_features:
            logger.info(f"    - {feat}")

        # Handle missing values
        df = handle_missing_values(df)

        # Validate features
        validate_features(df, feature_engine)

        # Save feature matrix
        output_path = PROCESSED_DATA_DIR / "feature_matrix.csv"
        save_feature_matrix(df, output_path)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE MATRIX SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        # List all columns
        logger.info("\nAll columns:")
        for i, col in enumerate(df.columns, 1):
            logger.info(f"  {i:2d}. {col}")

        logger.info("\n" + "=" * 70)
        logger.info("FEATURE MATRIX CREATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"End: {datetime.now()}")

        return df

    except Exception as e:
        logger.error(f"Error creating feature matrix: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
