"""
Download D-1 weather forecast data from Open-Meteo.

Uses the Historical Forecast API (2022+) for archived NWP model outputs
and the Archive API (2019-2021) for observed weather as a proxy.

Data is saved to data/raw/weather_forecast/ directory.

Usage:
    python scripts/download_weather_forecasts.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collectors.weather_forecast_collector import OpenMeteoForecastCollector
from config.settings import (
    DATA_START_DATE,
    DATA_END_DATE,
    WEATHER_FORECAST_DATA_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'weather_forecast_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("DOWNLOADING D-1 WEATHER FORECAST DATA")
    logger.info("=" * 60)
    logger.info(f"Date range: {DATA_START_DATE.date()} to {DATA_END_DATE.date()}")
    logger.info(f"Output dir: {WEATHER_FORECAST_DATA_DIR}")

    collector = OpenMeteoForecastCollector()

    start_date = DATA_START_DATE.strftime('%Y-%m-%d')
    end_date = DATA_END_DATE.strftime('%Y-%m-%d')

    combined_df = collector.collect_netherlands_forecasts(
        start_date=start_date,
        end_date=end_date,
        output_dir=WEATHER_FORECAST_DATA_DIR
    )

    if not combined_df.empty:
        logger.info("\nDataset Summary:")
        logger.info(f"  Shape: {combined_df.shape}")
        logger.info(f"  Columns: {list(combined_df.columns)}")

        # NaN summary
        logger.info("\nNaN counts per column:")
        for col in combined_df.columns:
            nan_count = combined_df[col].isna().sum()
            if nan_count > 0:
                nan_pct = nan_count / len(combined_df) * 100
                logger.info(f"  {col}: {nan_count:,} ({nan_pct:.1f}%)")

        # Basic stats for numeric columns
        logger.info("\nBasic statistics (Amsterdam only):")
        ams = combined_df[combined_df['location'] == 'Amsterdam']
        for col in ams.select_dtypes(include='number').columns:
            logger.info(f"  {col}: mean={ams[col].mean():.1f}, "
                         f"max={ams[col].max():.1f}, "
                         f"min={ams[col].min():.1f}")
    else:
        logger.error("Failed to collect weather forecast data")
        sys.exit(1)


if __name__ == "__main__":
    main()
