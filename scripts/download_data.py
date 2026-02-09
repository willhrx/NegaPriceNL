"""
Master script to download all required ENTSO-E data for the NegaPriceNL project.

This script downloads:
1. Dutch day-ahead prices (2019-2025)
2. Dutch solar generation actual (2019-2025)
3. Dutch wind generation actual (2019-2025)
4. Dutch solar/wind generation forecasts (2019-2025)
5. Dutch load actual and forecast (2019-2025)
6. Cross-border flows NL-DE and NL-BE (2019-2025)
7. German and Belgian day-ahead prices (2019-2025)

Data is saved to data/raw/entsoe/ directory.

Usage:
    python scripts/download_data.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging

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
    COUNTRY_CODE_DE,
    COUNTRY_CODE_BE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def download_prices(collector: EntsoeDataCollector, start: datetime, end: datetime):
    """Download day-ahead prices for NL, DE, and BE."""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Downloading Day-Ahead Prices")
    logger.info("="*60)

    # Netherlands
    logger.info("\n--- Netherlands Day-Ahead Prices ---")
    nl_prices = collector.get_day_ahead_prices(start, end, COUNTRY_CODE_NL)
    if not nl_prices.empty:
        output_path = ENTSOE_DATA_DIR / f"nl_day_ahead_prices_{start.year}_{end.year}.csv"
        collector.save_data(nl_prices, output_path)
        logger.info(f"✓ Netherlands prices saved: {len(nl_prices)} records")
    else:
        logger.warning("✗ Failed to collect Netherlands prices")

    # Germany
    logger.info("\n--- Germany Day-Ahead Prices ---")
    de_prices = collector.get_day_ahead_prices(start, end, COUNTRY_CODE_DE)
    if not de_prices.empty:
        output_path = ENTSOE_DATA_DIR / f"de_day_ahead_prices_{start.year}_{end.year}.csv"
        collector.save_data(de_prices, output_path)
        logger.info(f"✓ Germany prices saved: {len(de_prices)} records")
    else:
        logger.warning("✗ Failed to collect Germany prices")

    # Belgium
    logger.info("\n--- Belgium Day-Ahead Prices ---")
    be_prices = collector.get_day_ahead_prices(start, end, COUNTRY_CODE_BE)
    if not be_prices.empty:
        output_path = ENTSOE_DATA_DIR / f"be_day_ahead_prices_{start.year}_{end.year}.csv"
        collector.save_data(be_prices, output_path)
        logger.info(f"✓ Belgium prices saved: {len(be_prices)} records")
    else:
        logger.warning("✗ Failed to collect Belgium prices")


def download_generation_actual(collector: EntsoeDataCollector, start: datetime, end: datetime):
    """Download actual generation data for Netherlands."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Downloading Actual Generation Data")
    logger.info("="*60)

    logger.info("\n--- Netherlands Actual Generation ---")
    nl_generation = collector.get_generation_actual(start, end, COUNTRY_CODE_NL)

    if not nl_generation.empty:
        # Extract solar and wind columns if they exist
        if 'Solar' in nl_generation.columns:
            # Convert to numeric and extract (use base column, not .1 suffix which is consumption)
            nl_generation['Solar'] = pd.to_numeric(nl_generation['Solar'], errors='coerce')
            solar = nl_generation[['Solar']].rename(columns={'Solar': 'solar_generation_mw'})
            output_path = ENTSOE_DATA_DIR / f"nl_solar_generation_{start.year}_{end.year}.csv"
            collector.save_data(solar, output_path)
            logger.info(f"✓ Solar generation saved: {len(solar)} records")

        if 'Wind Onshore' in nl_generation.columns or 'Wind Offshore' in nl_generation.columns:
            # Combine onshore and offshore wind (use only base columns, not .1 suffix which are consumption)
            wind_cols = [col for col in nl_generation.columns
                        if 'Wind' in col and not col.endswith('.1')]

            # Convert to numeric and sum
            for col in wind_cols:
                nl_generation[col] = pd.to_numeric(nl_generation[col], errors='coerce')

            nl_generation['total_wind_mw'] = nl_generation[wind_cols].sum(axis=1)
            wind = nl_generation[['total_wind_mw']].rename(columns={'total_wind_mw': 'wind_generation_mw'})
            output_path = ENTSOE_DATA_DIR / f"nl_wind_generation_{start.year}_{end.year}.csv"
            collector.save_data(wind, output_path)
            logger.info(f"✓ Wind generation saved: {len(wind)} records (offshore + onshore)")

        # Save full generation mix for reference
        output_path = ENTSOE_DATA_DIR / f"nl_generation_all_{start.year}_{end.year}.csv"
        collector.save_data(nl_generation, output_path)
        logger.info(f"✓ Full generation mix saved: {len(nl_generation)} records")
    else:
        logger.warning("✗ Failed to collect Netherlands generation data")


def download_generation_forecast(collector: EntsoeDataCollector, start: datetime, end: datetime):
    """Download generation forecast data for Netherlands."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Downloading Generation Forecast Data")
    logger.info("="*60)

    logger.info("\n--- Netherlands Generation Forecast ---")
    nl_forecast = collector.get_generation_forecast(start, end, COUNTRY_CODE_NL)

    if not nl_forecast.empty:
        # Convert Series to DataFrame if needed
        if isinstance(nl_forecast, pd.Series):
            nl_forecast = nl_forecast.to_frame(name='generation_forecast_mw')
            logger.info("Note: Forecast returned as Series, converted to DataFrame")

        # Save full forecast
        output_path = ENTSOE_DATA_DIR / f"nl_generation_forecast_{start.year}_{end.year}.csv"
        collector.save_data(nl_forecast, output_path)
        logger.info(f"✓ Generation forecast saved: {len(nl_forecast)} records")

        # Extract solar and wind if available (only if we have columns)
        if hasattr(nl_forecast, 'columns') and 'Solar' in nl_forecast.columns:
            solar_forecast = nl_forecast[['Solar']].rename(columns={'Solar': 'solar_forecast_mw'})
            output_path = ENTSOE_DATA_DIR / f"nl_solar_forecast_{start.year}_{end.year}.csv"
            collector.save_data(solar_forecast, output_path)
            logger.info(f"✓ Solar forecast saved")

        if hasattr(nl_forecast, 'columns'):
            wind_cols = [col for col in nl_forecast.columns if 'Wind' in col]
            if wind_cols:
                nl_forecast['total_wind_forecast'] = nl_forecast[wind_cols].sum(axis=1)
                wind_forecast = nl_forecast[['total_wind_forecast']].rename(columns={'total_wind_forecast': 'wind_forecast_mw'})
                output_path = ENTSOE_DATA_DIR / f"nl_wind_forecast_{start.year}_{end.year}.csv"
                collector.save_data(wind_forecast, output_path)
                logger.info(f"✓ Wind forecast saved")
    else:
        logger.warning("✗ Failed to collect Netherlands generation forecast")


def download_load(collector: EntsoeDataCollector, start: datetime, end: datetime):
    """Download load data (actual and forecast) for Netherlands."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Downloading Load Data")
    logger.info("="*60)

    # Actual load
    logger.info("\n--- Netherlands Actual Load ---")
    nl_load = collector.get_load_actual(start, end, COUNTRY_CODE_NL)
    if not nl_load.empty:
        output_path = ENTSOE_DATA_DIR / f"nl_load_{start.year}_{end.year}.csv"
        collector.save_data(nl_load, output_path)
        logger.info(f"✓ Actual load saved: {len(nl_load)} records")
    else:
        logger.warning("✗ Failed to collect Netherlands load")

    # Load forecast
    logger.info("\n--- Netherlands Load Forecast ---")
    nl_load_forecast = collector.get_load_forecast(start, end, COUNTRY_CODE_NL)
    if not nl_load_forecast.empty:
        output_path = ENTSOE_DATA_DIR / f"nl_load_forecast_{start.year}_{end.year}.csv"
        collector.save_data(nl_load_forecast, output_path)
        logger.info(f"✓ Load forecast saved: {len(nl_load_forecast)} records")
    else:
        logger.warning("✗ Failed to collect Netherlands load forecast")


def download_cross_border_flows(collector: EntsoeDataCollector, start: datetime, end: datetime):
    """Download cross-border flow data."""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Downloading Cross-Border Flows")
    logger.info("="*60)

    # NL -> DE
    logger.info("\n--- Netherlands -> Germany ---")
    nl_de_flow = collector.get_cross_border_flows(start, end, COUNTRY_CODE_NL, COUNTRY_CODE_DE)
    if not nl_de_flow.empty:
        output_path = ENTSOE_DATA_DIR / f"cross_border_nl_de_{start.year}_{end.year}.csv"
        collector.save_data(nl_de_flow, output_path)
        logger.info(f"✓ NL->DE flows saved: {len(nl_de_flow)} records")
    else:
        logger.warning("✗ Failed to collect NL->DE flows")

    # DE -> NL
    logger.info("\n--- Germany -> Netherlands ---")
    de_nl_flow = collector.get_cross_border_flows(start, end, COUNTRY_CODE_DE, COUNTRY_CODE_NL)
    if not de_nl_flow.empty:
        output_path = ENTSOE_DATA_DIR / f"cross_border_de_nl_{start.year}_{end.year}.csv"
        collector.save_data(de_nl_flow, output_path)
        logger.info(f"✓ DE->NL flows saved: {len(de_nl_flow)} records")
    else:
        logger.warning("✗ Failed to collect DE->NL flows")

    # NL -> BE
    logger.info("\n--- Netherlands -> Belgium ---")
    nl_be_flow = collector.get_cross_border_flows(start, end, COUNTRY_CODE_NL, COUNTRY_CODE_BE)
    if not nl_be_flow.empty:
        output_path = ENTSOE_DATA_DIR / f"cross_border_nl_be_{start.year}_{end.year}.csv"
        collector.save_data(nl_be_flow, output_path)
        logger.info(f"✓ NL->BE flows saved: {len(nl_be_flow)} records")
    else:
        logger.warning("✗ Failed to collect NL->BE flows")

    # BE -> NL
    logger.info("\n--- Belgium -> Netherlands ---")
    be_nl_flow = collector.get_cross_border_flows(start, end, COUNTRY_CODE_BE, COUNTRY_CODE_NL)
    if not be_nl_flow.empty:
        output_path = ENTSOE_DATA_DIR / f"cross_border_be_nl_{start.year}_{end.year}.csv"
        collector.save_data(be_nl_flow, output_path)
        logger.info(f"✓ BE->NL flows saved: {len(be_nl_flow)} records")
    else:
        logger.warning("✗ Failed to collect BE->NL flows")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download ENTSO-E data for NegaPriceNL project"
    )
    parser.add_argument(
        '--start',
        type=str,
        default=DATA_START_DATE.strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=DATA_END_DATE.strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--skip-prices',
        action='store_true',
        help='Skip downloading price data'
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip downloading generation data'
    )
    parser.add_argument(
        '--skip-forecast',
        action='store_true',
        help='Skip downloading forecast data'
    )
    parser.add_argument(
        '--skip-load',
        action='store_true',
        help='Skip downloading load data'
    )
    parser.add_argument(
        '--skip-flows',
        action='store_true',
        help='Skip downloading cross-border flows'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    logger.info("\n" + "="*60)
    logger.info("NegaPriceNL Data Download Script")
    logger.info("="*60)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Output directory: {ENTSOE_DATA_DIR}")
    logger.info("="*60)

    try:
        # Initialize collector
        logger.info("\nInitializing ENTSO-E data collector...")
        collector = EntsoeDataCollector()
        logger.info("✓ Collector initialized successfully")

        # Download data based on arguments
        if not args.skip_prices:
            download_prices(collector, start_date, end_date)

        if not args.skip_generation:
            download_generation_actual(collector, start_date, end_date)

        if not args.skip_forecast:
            download_generation_forecast(collector, start_date, end_date)

        if not args.skip_load:
            download_load(collector, start_date, end_date)

        if not args.skip_flows:
            download_cross_border_flows(collector, start_date, end_date)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("DATA DOWNLOAD COMPLETE")
        logger.info("="*60)
        logger.info(f"Total API requests made: {collector.requests_made}")
        logger.info(f"Data saved to: {ENTSOE_DATA_DIR}")
        logger.info("\nNext steps:")
        logger.info("1. Check the downloaded files in data/raw/entsoe/")
        logger.info("2. Run data quality checks")
        logger.info("3. Proceed with data processing and feature engineering")
        logger.info("="*60)

    except ValueError as e:
        logger.error(f"\n✗ Configuration error: {e}")
        logger.error("Please ensure ENTSOE_API_KEY is set in your .env file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
