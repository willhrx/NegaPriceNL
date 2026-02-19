"""
Data Preparation Module v6 for NegaPriceNL Project

Extends v5 by adding day-ahead Net Transfer Capacity (NTC) data for all
4 NL interconnectors: NL-BE, NL-DE_LU, NL-GB (BritNed), NL-NO2 (NorNed).

NTC is published before the DA auction and is D-1 safe.

Data sources merged (v5 sources + 1 new):
1–7. All v5 sources (prices, generation, load, weather, cross-border flows)
8.   Day-ahead NTC for all NL borders (hourly → 15-min via ffill)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ENTSOE_DATA_DIR, PROCESSED_DATA_DIR
from src.data.processing.data_preparation_v5 import DataPreparationV5

logger = logging.getLogger(__name__)


class DataPreparationV6(DataPreparationV5):
    """
    Unified data preparation pipeline v6 — adds NTC data to v5 pipeline.
    """

    def load_ntc_data(self) -> Optional[pd.DataFrame]:
        """Load day-ahead NTC for all NL borders, upsample hourly → 15-min."""
        logger.info("Loading day-ahead NTC data...")

        ntc_file = ENTSOE_DATA_DIR / "ntc_nl_combined_2019_2025.csv"
        if not ntc_file.exists():
            logger.warning(f"  NTC data not found: {ntc_file}")
            return None

        df = pd.read_csv(ntc_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"  Raw NTC: {len(df):,} records (hourly), {len(df.columns)} columns")

        # Upsample to 15-min via ffill (same pattern as cross-border flows)
        df = df.reindex(self.full_index, method='ffill')
        df.index.name = 'datetime'

        logger.info(f"  After 15-min reindex: {len(df):,} records")

        for col in df.columns:
            self._check_missing(df, col, col)

        return df

    def merge_all_data(self) -> pd.DataFrame:
        """Merge all v5 data sources + NTC onto 15-min UTC index."""
        # Run the full v5 merge first
        df = super().merge_all_data()

        # Add NTC data
        ntc = self.load_ntc_data()
        if ntc is not None:
            df = pd.merge(df, ntc, left_index=True, right_index=True, how='left')
            logger.info(f"+ NTC data: {len(df):,}")

        logger.info(f"\nFinal unified dataset v6: {len(df):,} records, {len(df.columns)} columns")

        return df

    def run(self) -> pd.DataFrame:
        """Execute the complete v6 data preparation pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info("NEGAPRICENL DATA PREPARATION PIPELINE v6")
        logger.info("=" * 70)
        logger.info(f"Start: {datetime.now()}")

        try:
            df = self.merge_all_data()
            self.unified_df = df

            report = self.generate_quality_report()
            print(report)

            output_path = PROCESSED_DATA_DIR / "unified_dataset_v6.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            logger.info("\nNaN summary:")
            for col in df.columns:
                nan_ct = df[col].isna().sum()
                if nan_ct > 0:
                    logger.info(f"  {col}: {nan_ct:,} ({nan_ct / len(df) * 100:.1f}%)")

            logger.info("\n" + "=" * 70)
            logger.info("DATA PREPARATION v6 COMPLETE")
            logger.info("=" * 70)

            return df

        except Exception as e:
            logger.error(f"ERROR in data preparation v6: {e}", exc_info=True)
            raise


def main():
    pipeline = DataPreparationV6()
    df = pipeline.run()

    print("\n" + "=" * 70)
    print("DATASET v6 SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nMissing values:")
    missing = df.isna().sum()
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing[col] / len(df) * 100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
