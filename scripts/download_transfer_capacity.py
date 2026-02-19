"""
Download Day-Ahead Net Transfer Capacity (NTC) for all NL interconnectors.

Downloads NTC from ENTSO-E Transparency Platform for 4 borders (8 directions):
  NL <-> BE, NL <-> DE_LU, NL <-> GB (BritNed), NL <-> NO_2 (NorNed)

NTC is published before the day-ahead auction and represents the maximum
allowed commercial exchange. It is D-1 safe and suitable as a model feature.

Usage:
    python scripts/download_transfer_capacity.py
    python scripts/download_transfer_capacity.py --start 2024-01-01 --end 2025-12-31
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collectors.entsoe_collector import EntsoeDataCollector
from config.settings import (
    DATA_START_DATE,
    DATA_END_DATE,
    ENTSOE_DATA_DIR,
    COUNTRY_CODE_NL,
    COUNTRY_CODE_BE,
    COUNTRY_CODE_DE_LU,
    COUNTRY_CODE_GB,
    COUNTRY_CODE_NO2,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'ntc_download.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# All NL borders to download (from, to, label)
NL_BORDERS = [
    (COUNTRY_CODE_NL, COUNTRY_CODE_BE, 'nl_be'),
    (COUNTRY_CODE_BE, COUNTRY_CODE_NL, 'be_nl'),
    (COUNTRY_CODE_NL, COUNTRY_CODE_DE_LU, 'nl_delu'),
    (COUNTRY_CODE_DE_LU, COUNTRY_CODE_NL, 'delu_nl'),
    (COUNTRY_CODE_NL, COUNTRY_CODE_GB, 'nl_gb'),
    (COUNTRY_CODE_GB, COUNTRY_CODE_NL, 'gb_nl'),
    (COUNTRY_CODE_NL, COUNTRY_CODE_NO2, 'nl_no2'),
    (COUNTRY_CODE_NO2, COUNTRY_CODE_NL, 'no2_nl'),
]


def download_all_ntc(
    collector: EntsoeDataCollector,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    """Download day-ahead NTC for all NL borders."""
    results = {}

    for from_code, to_code, label in NL_BORDERS:
        logger.info(f"\n--- NTC {from_code} -> {to_code} ---")

        df = collector.get_net_transfer_capacity_dayahead(
            start, end, from_code, to_code,
        )

        if not df.empty:
            filename = f"ntc_da_{label}_{start.year}_{end.year}.csv"
            output_path = ENTSOE_DATA_DIR / filename
            collector.save_data(df, output_path)
            logger.info(f"  Saved {len(df)} records to {filename}")
            results[label] = df
        else:
            logger.warning(f"  No data for {from_code} -> {to_code}")

    return results


def build_combined_file(
    results: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
):
    """Merge all NTC directions into a single combined file."""
    if not results:
        logger.warning("No NTC data to combine")
        return

    logger.info("\n--- Building combined NTC file ---")

    # Merge all directions on datetime index
    combined = None
    for label, df in results.items():
        if combined is None:
            combined = df.copy()
        else:
            combined = combined.join(df, how='outer')

    # Compute totals
    export_cols = [c for c in combined.columns if c.startswith('ntc_da_nl_')]
    import_cols = [c for c in combined.columns if not c.startswith('ntc_da_nl_')]

    if export_cols:
        combined['ntc_nl_total_export_mw'] = combined[export_cols].sum(axis=1)
    if import_cols:
        combined['ntc_nl_total_import_mw'] = combined[import_cols].sum(axis=1)
    if export_cols and import_cols:
        combined['ntc_nl_net_mw'] = (
            combined['ntc_nl_total_import_mw'] - combined['ntc_nl_total_export_mw']
        )

    output_path = ENTSOE_DATA_DIR / f"ntc_nl_combined_{start.year}_{end.year}.csv"
    combined.to_csv(output_path)
    logger.info(f"  Combined NTC saved: {len(combined)} records, {len(combined.columns)} columns")
    logger.info(f"  Columns: {list(combined.columns)}")
    logger.info(f"  Saved to: {output_path}")

    # Summary statistics
    logger.info("\n--- NTC Summary Statistics ---")
    for col in combined.columns:
        s = combined[col]
        logger.info(
            f"  {col}: mean={s.mean():.0f} MW, "
            f"min={s.min():.0f}, max={s.max():.0f}, "
            f"NaN={s.isna().sum()} ({s.isna().mean()*100:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download day-ahead NTC for all NL interconnectors"
    )
    parser.add_argument(
        '--start', type=str,
        default=DATA_START_DATE.strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--end', type=str,
        default=DATA_END_DATE.strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD)',
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    logger.info("=" * 60)
    logger.info("NegaPriceNL — Day-Ahead NTC Download")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Borders: {len(NL_BORDERS)} directions across 4 interconnectors")
    logger.info(f"Output directory: {ENTSOE_DATA_DIR}")
    logger.info("=" * 60)

    try:
        collector = EntsoeDataCollector()

        results = download_all_ntc(collector, start_date, end_date)

        build_combined_file(results, start_date, end_date)

        logger.info("\n" + "=" * 60)
        logger.info("NTC DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total API requests: {collector.requests_made}")
        logger.info(f"Directions downloaded: {len(results)}/{len(NL_BORDERS)}")
        logger.info(f"Data saved to: {ENTSOE_DATA_DIR}")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Ensure ENTSOE_API_KEY is set in your .env file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
