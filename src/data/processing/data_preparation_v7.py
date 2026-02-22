"""
Data Preparation Module v7 for NegaPriceNL Project

Phase 1: Extends v6 with minimal changes (uses same data sources).
Phase 2+: Will add German DA prices, TTF, and EUA data.

Current data sources (Phase 1 - same as v6):
1. NL day-ahead prices
2. Solar + wind generation (actual + forecasts)
3. Load data (actual + forecasts)
4. Weather forecasts
5. Cross-border flows
6. NTC (Net Transfer Capacity)
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from src.data.processing.data_preparation_v6 import DataPreparationV6

logger = logging.getLogger(__name__)


class DataPreparationV7(DataPreparationV6):
    """
    Unified data preparation pipeline v7.

    Phase 1: Uses v6 data sources (NL prices, generation, load, weather, NTC).
    Future phases will add: German prices (Phase 2), TTF/EUA (Phase 3).
    """

    def run(self) -> pd.DataFrame:
        """Execute the complete v7 data preparation pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info("NEGAPRICENL DATA PREPARATION PIPELINE v7")
        logger.info("=" * 70)
        logger.info(f"Start: {datetime.now()}")

        try:
            # Merge all data (uses v6 logic for Phase 1)
            df = self.merge_all_data()
            self.unified_df = df

            # Generate quality report
            report = self.generate_quality_report()
            print(report)

            # Save to v7 output path
            output_path = PROCESSED_DATA_DIR / "unified_dataset_v7.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)

            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            # NaN summary
            logger.info("\nNaN summary:")
            for col in df.columns:
                nan_ct = df[col].isna().sum()
                if nan_ct > 0:
                    logger.info(f"  {col}: {nan_ct:,} ({nan_ct / len(df) * 100:.1f}%)")

            logger.info("\n" + "=" * 70)
            logger.info("DATA PREPARATION v7 COMPLETE")
            logger.info("=" * 70)

            return df

        except Exception as e:
            logger.error(f"ERROR in data preparation v7: {e}", exc_info=True)
            raise


def main():
    pipeline = DataPreparationV7()
    df = pipeline.run()

    print("\n" + "=" * 70)
    print("DATASET v7 SUMMARY")
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
