"""
Download day-ahead solar and wind generation forecasts from ENTSO-E.

Uses the Wind and Solar Generation Forecasts endpoint (document type A69)
which provides separate forecasts for:
- Solar generation
- Wind onshore generation
- Wind offshore generation

Data is saved to data/raw/entsoe/ directory.

Usage:
    python scripts/download_generation_forecasts.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collectors.entsoe_collector import EntsoeDataCollector
from config.settings import (
    DATA_START_DATE,
    DATA_END_DATE,
    ENTSOE_DATA_DIR,
    COUNTRY_CODE_NL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'generation_forecast_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

YEAR_START = DATA_START_DATE.year
YEAR_END = DATA_END_DATE.year


def main():
    logger.info("=" * 60)
    logger.info("DOWNLOADING SOLAR & WIND GENERATION FORECASTS")
    logger.info("=" * 60)
    logger.info(f"Date range: {DATA_START_DATE.date()} to {DATA_END_DATE.date()}")

    collector = EntsoeDataCollector()

    # Download wind and solar forecasts
    logger.info("\nFetching wind & solar DA generation forecasts for NL...")
    forecast_df = collector.get_wind_and_solar_forecast(
        DATA_START_DATE, DATA_END_DATE, COUNTRY_CODE_NL
    )

    if forecast_df.empty:
        logger.error("No forecast data returned. Exiting.")
        sys.exit(1)

    # Inspect raw columns from the API
    logger.info(f"\nRaw columns from API: {list(forecast_df.columns)}")
    logger.info(f"Raw shape: {forecast_df.shape}")
    logger.info(f"Index range: {forecast_df.index.min()} to {forecast_df.index.max()}")

    # Print first few rows to understand structure
    logger.info(f"\nSample data:\n{forecast_df.head()}")

    # Flatten multi-level columns if present
    if isinstance(forecast_df.columns, pd.MultiIndex):
        logger.info("Flattening multi-level column index...")
        forecast_df.columns = [
            '_'.join(str(c) for c in col).strip() for col in forecast_df.columns
        ]
        logger.info(f"Flattened columns: {list(forecast_df.columns)}")

    # Map columns to clean names
    # The entsoe-py API typically returns columns like:
    #   'Solar', 'Wind Onshore', 'Wind Offshore'
    # or multi-level with 'Actual Aggregated' sub-columns
    column_mapping = {}
    for col in forecast_df.columns:
        col_lower = str(col).lower()
        if 'solar' in col_lower:
            column_mapping[col] = 'solar_forecast_mw'
        elif 'offshore' in col_lower:
            column_mapping[col] = 'wind_offshore_forecast_mw'
        elif 'onshore' in col_lower:
            column_mapping[col] = 'wind_onshore_forecast_mw'
        elif 'wind' in col_lower:
            # Generic wind column (if no onshore/offshore split)
            column_mapping[col] = 'wind_forecast_mw'

    if not column_mapping:
        logger.warning("Could not auto-map columns. Saving raw data as-is.")
    else:
        logger.info(f"\nColumn mapping: {column_mapping}")
        forecast_df = forecast_df.rename(columns=column_mapping)

    # Compute combined wind forecast if onshore + offshore exist separately
    if ('wind_onshore_forecast_mw' in forecast_df.columns
            and 'wind_offshore_forecast_mw' in forecast_df.columns):
        forecast_df['wind_forecast_mw'] = (
            forecast_df['wind_onshore_forecast_mw'].fillna(0)
            + forecast_df['wind_offshore_forecast_mw'].fillna(0)
        )
        logger.info("Computed combined wind_forecast_mw = onshore + offshore")

    # Save
    output_path = ENTSOE_DATA_DIR / f"nl_solar_wind_forecast_{YEAR_START}_{YEAR_END}.csv"
    forecast_df.to_csv(output_path)
    logger.info(f"\nSaved to: {output_path}")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Records: {len(forecast_df):,}")
    logger.info(f"Columns: {list(forecast_df.columns)}")
    logger.info(f"Date range: {forecast_df.index.min()} to {forecast_df.index.max()}")

    # Check resolution
    diffs = forecast_df.index.to_series().diff().dropna()
    modal_freq = diffs.mode().iloc[0] if len(diffs) > 0 else "unknown"
    logger.info(f"Modal frequency: {modal_freq}")

    # NaN summary
    logger.info("\nNaN counts per column:")
    for col in forecast_df.columns:
        nan_count = forecast_df[col].isna().sum()
        nan_pct = nan_count / len(forecast_df) * 100
        logger.info(f"  {col}: {nan_count:,} ({nan_pct:.1f}%)")

    # Basic stats
    logger.info("\nBasic statistics:")
    for col in forecast_df.select_dtypes(include='number').columns:
        logger.info(f"  {col}: mean={forecast_df[col].mean():.1f}, "
                     f"max={forecast_df[col].max():.1f}, "
                     f"min={forecast_df[col].min():.1f}")

    logger.info(f"\nTotal API requests made: {collector.requests_made}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
