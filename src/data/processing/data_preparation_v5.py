"""
Data Preparation Module v5 for NegaPriceNL Project

Phase 1 auction-aligned pipeline:
- 15-minute resolution throughout (2019-2025)
- Includes D-1 forecast data (weather, generation, load) as first-class columns
- Includes actual data (generation, load) for backtesting only
- Enforces proper timezone handling (CET/CEST → UTC)

Data sources merged:
1. Day-ahead prices (hourly → 15-min via ffill pre-Oct 2025, native 15-min after)
2. Solar + wind generation actuals (15-min)
3. Load actual + DA load forecast (15-min)
4. ENTSO-E solar/wind DA generation forecasts (15-min)
5. ENTSO-E total generation forecast (mixed → 15-min via ffill)
6. Open-Meteo weather forecasts (hourly → 15-min via ffill)
7. Cross-border NL-DE flows (hourly → 15-min via ffill)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ENTSOE_DATA_DIR,
    WEATHER_FORECAST_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATA_START_DATE,
    DATA_END_DATE,
    QH_PRICE_START_DATE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityIssue:
    """Container for data quality issues."""

    def __init__(self, severity: str, category: str, description: str, details: Dict = None):
        self.severity = severity
        self.category = category
        self.description = description
        self.details = details or {}

    def __str__(self):
        return f"[{self.severity.upper()}] {self.category}: {self.description}"


class DataPreparationV5:
    """
    Unified data preparation pipeline v5 — 15-minute, auction-aligned.

    Creates a 15-min UTC-indexed dataset with both actuals (for backtesting)
    and DA forecasts (for features).
    """

    def __init__(self):
        self.issues: List[DataQualityIssue] = []
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self.unified_df: Optional[pd.DataFrame] = None

        # 15-min UTC index spanning the full data range
        self.full_index = pd.date_range(
            start=DATA_START_DATE,
            end=DATA_END_DATE + pd.Timedelta(hours=23, minutes=45),
            freq='15min',
            tz='UTC',
        )
        logger.info(f"DataPreparationV5 initialized — {len(self.full_index):,} target QH slots")

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_price_data(self) -> pd.DataFrame:
        """Load day-ahead prices, upsample hourly → 15-min via ffill."""
        logger.info("Loading price data...")

        price_file = ENTSOE_DATA_DIR / "nl_day_ahead_prices_2019_2025.csv"
        if not price_file.exists():
            raise FileNotFoundError(f"Price data not found: {price_file}")

        df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'

        if 'price' in df.columns:
            df = df.rename(columns={'price': 'price_eur_mwh'})

        logger.info(f"  Raw: {len(df):,} records, {df.index.min()} → {df.index.max()}")

        # Mark which rows are native 15-min prices
        qh_cutoff = pd.Timestamp(QH_PRICE_START_DATE, tz='UTC')

        # Reindex to 15-min — ffill replicates hourly prices into 4 QH slots
        df = df.reindex(self.full_index, method='ffill')
        df.index.name = 'datetime'

        # Flag native 15-min price rows
        df['price_is_15min'] = df.index >= qh_cutoff

        logger.info(f"  After 15-min reindex: {len(df):,} records")
        self._check_missing(df, 'price_eur_mwh', 'prices')

        return df[['price_eur_mwh', 'price_is_15min']]

    def load_generation_actual(self) -> pd.DataFrame:
        """Load actual solar + wind generation (15-min native)."""
        logger.info("Loading actual generation data...")

        # Solar
        solar_file = ENTSOE_DATA_DIR / "nl_solar_generation_2019_2025.csv"
        solar = pd.read_csv(solar_file, index_col=0, parse_dates=True)
        solar.index = pd.to_datetime(solar.index, utc=True)
        solar = solar.dropna(how='all')  # blank row after header
        solar['solar_generation_mw'] = pd.to_numeric(solar['solar_generation_mw'], errors='coerce')

        # Wind
        wind_file = ENTSOE_DATA_DIR / "nl_wind_generation_2019_2025.csv"
        wind = pd.read_csv(wind_file, index_col=0, parse_dates=True)
        wind.index = pd.to_datetime(wind.index, utc=True)
        wind['wind_generation_mw'] = pd.to_numeric(wind['wind_generation_mw'], errors='coerce')

        generation = pd.merge(solar, wind, left_index=True, right_index=True, how='outer')
        generation.index.name = 'datetime'

        logger.info(f"  Solar+Wind: {len(generation):,} records")
        self._check_missing(generation, 'solar_generation_mw', 'solar generation')
        self._check_missing(generation, 'wind_generation_mw', 'wind generation')

        return generation[['solar_generation_mw', 'wind_generation_mw']]

    def load_generation_forecasts(self) -> pd.DataFrame:
        """Load ENTSO-E DA solar/wind generation forecasts (15-min native)."""
        logger.info("Loading solar/wind generation forecasts...")

        forecast_file = ENTSOE_DATA_DIR / "nl_solar_wind_forecast_2019_2025.csv"
        if not forecast_file.exists():
            raise FileNotFoundError(f"Solar/wind forecast not found: {forecast_file}")

        df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'

        cols = ['solar_forecast_mw', 'wind_onshore_forecast_mw',
                'wind_offshore_forecast_mw', 'wind_forecast_mw']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"  Solar/wind forecasts: {len(df):,} records")
        for col in cols:
            self._check_missing(df, col, col)

        return df[cols]

    def load_total_generation_forecast(self) -> Optional[pd.DataFrame]:
        """Load ENTSO-E total generation forecast, upsample to 15-min."""
        logger.info("Loading total generation forecast...")

        forecast_file = ENTSOE_DATA_DIR / "nl_generation_forecast_2019_2025.csv"
        if not forecast_file.exists():
            logger.warning(f"  Total gen forecast not found: {forecast_file}")
            return None

        df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'

        # The column is named 'Actual Aggregated' from the ENTSO-E API
        col_name = df.columns[0]
        df = df.rename(columns={col_name: 'total_generation_forecast_mw'})
        df['total_generation_forecast_mw'] = pd.to_numeric(
            df['total_generation_forecast_mw'], errors='coerce'
        )

        logger.info(f"  Raw total gen forecast: {len(df):,} records")

        # Reindex to 15-min (mixed resolution source)
        df = df.reindex(self.full_index, method='ffill')
        df.index.name = 'datetime'

        logger.info(f"  After 15-min reindex: {len(df):,} records")
        self._check_missing(df, 'total_generation_forecast_mw', 'total gen forecast')

        return df[['total_generation_forecast_mw']]

    def load_load_data(self) -> pd.DataFrame:
        """Load actual load + DA load forecast (15-min native)."""
        logger.info("Loading load data...")

        # Actual load
        load_file = ENTSOE_DATA_DIR / "nl_load_2019_2025.csv"
        load_actual = pd.read_csv(load_file, index_col=0, parse_dates=True)
        load_actual.index = pd.to_datetime(load_actual.index, utc=True)
        load_actual['load_mw'] = pd.to_numeric(load_actual['load_mw'], errors='coerce')

        # Load forecast (DA forecast — directly usable, no shift needed)
        forecast_file = ENTSOE_DATA_DIR / "nl_load_forecast_2019_2025.csv"
        load_forecast = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
        load_forecast.index = pd.to_datetime(load_forecast.index, utc=True)
        load_forecast['load_forecast_mw'] = pd.to_numeric(
            load_forecast['load_forecast_mw'], errors='coerce'
        )

        load_data = pd.merge(
            load_actual[['load_mw']],
            load_forecast[['load_forecast_mw']],
            left_index=True, right_index=True, how='outer'
        )
        load_data.index.name = 'datetime'

        logger.info(f"  Load actual+forecast: {len(load_data):,} records")
        self._check_missing(load_data, 'load_mw', 'load actual')
        self._check_missing(load_data, 'load_forecast_mw', 'load forecast')

        return load_data

    def load_weather_forecast(self) -> pd.DataFrame:
        """Load Amsterdam weather forecast, upsample hourly → 15-min."""
        logger.info("Loading weather forecast data (Amsterdam)...")

        weather_file = WEATHER_FORECAST_DATA_DIR / "nl_weather_forecast_amsterdam_20190101_20251231.csv"
        if not weather_file.exists():
            raise FileNotFoundError(f"Weather forecast not found: {weather_file}")

        df = pd.read_csv(weather_file)

        # Naive datetimes in CET/CEST → convert to UTC
        # During fall-back DST transitions, 02:00 is ambiguous (occurs twice).
        # Data is chronological: first occurrence is CEST (dst=True), second is CET.
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        dt_index = df.index

        # Build ambiguous mask: True = DST for ambiguous times
        is_dup = dt_index.duplicated(keep='first')
        ambiguous = ~is_dup  # first occurrence: DST=True; second: DST=False

        df.index = dt_index.tz_localize(
            'Europe/Amsterdam', ambiguous=ambiguous, nonexistent='shift_forward'
        ).tz_convert('UTC')
        # Drop any duplicate UTC timestamps
        df = df[~df.index.duplicated(keep='first')]

        forecast_cols = [
            'forecast_ghi_wm2', 'forecast_wind_speed_100m_ms',
            'forecast_temperature_2m_c', 'forecast_cloud_cover_pct',
            'forecast_pressure_hpa', 'forecast_humidity_pct',
        ]
        available = [c for c in forecast_cols if c in df.columns]
        df = df[available]

        logger.info(f"  Raw weather forecast: {len(df):,} records (hourly)")

        # Upsample to 15-min via ffill
        df = df.reindex(self.full_index, method='ffill')
        df.index.name = 'datetime'

        logger.info(f"  After 15-min reindex: {len(df):,} records")
        for col in available:
            self._check_missing(df, col, col)

        return df

    def load_cross_border(self) -> Optional[pd.DataFrame]:
        """Load NL-DE cross-border flows, upsample hourly → 15-min."""
        logger.info("Loading cross-border flow data...")

        flow_file = ENTSOE_DATA_DIR / "cross_border_nl_de_2019_2025.csv"
        if not flow_file.exists():
            logger.warning(f"  Cross-border data not found: {flow_file}")
            return None

        df = pd.read_csv(flow_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'
        df['flow_NL_DE_mw'] = pd.to_numeric(df['flow_NL_DE_mw'], errors='coerce')

        logger.info(f"  Raw cross-border: {len(df):,} records (hourly)")

        # Upsample to 15-min via ffill
        df = df.reindex(self.full_index, method='ffill')
        df.index.name = 'datetime'

        logger.info(f"  After 15-min reindex: {len(df):,} records")
        self._check_missing(df, 'flow_NL_DE_mw', 'cross-border NL-DE')

        return df[['flow_NL_DE_mw']]

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_all_data(self) -> pd.DataFrame:
        """Merge all data sources onto 15-min UTC index."""
        logger.info("\n" + "=" * 70)
        logger.info("MERGING ALL DATA SOURCES (15-min)")
        logger.info("=" * 70)

        prices = self.load_price_data()
        gen_actual = self.load_generation_actual()
        gen_forecast = self.load_generation_forecasts()
        total_gen_forecast = self.load_total_generation_forecast()
        load_data = self.load_load_data()
        weather = self.load_weather_forecast()
        cross_border = self.load_cross_border()

        # Base: price data (already on full 15-min index)
        df = prices.copy()
        logger.info(f"\nBase (prices): {len(df):,} records")

        # Merge generation actuals
        df = pd.merge(df, gen_actual, left_index=True, right_index=True, how='left')
        logger.info(f"+ generation actuals: {len(df):,}")

        # Merge solar/wind forecasts
        df = pd.merge(df, gen_forecast, left_index=True, right_index=True, how='left')
        logger.info(f"+ solar/wind forecasts: {len(df):,}")

        # Merge total generation forecast
        if total_gen_forecast is not None:
            df = pd.merge(df, total_gen_forecast, left_index=True, right_index=True, how='left')
            logger.info(f"+ total gen forecast: {len(df):,}")

        # Merge load
        df = pd.merge(df, load_data, left_index=True, right_index=True, how='left')
        logger.info(f"+ load actual+forecast: {len(df):,}")

        # Merge weather forecast
        df = pd.merge(df, weather, left_index=True, right_index=True, how='left')
        logger.info(f"+ weather forecast: {len(df):,}")

        # Merge cross-border
        if cross_border is not None:
            df = pd.merge(df, cross_border, left_index=True, right_index=True, how='left')
            logger.info(f"+ cross-border flows: {len(df):,}")

        # Target variable
        df['is_negative_price'] = (df['price_eur_mwh'] < 0).astype(int)

        logger.info(f"\nFinal unified dataset: {len(df):,} records, {len(df.columns)} columns")
        logger.info(f"Date range: {df.index.min()} → {df.index.max()}")
        logger.info(f"Negative price QHs: {df['is_negative_price'].sum():,} "
                     f"({df['is_negative_price'].mean() * 100:.2f}%)")

        return df

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def _check_missing(self, df: pd.DataFrame, column: str, name: str):
        """Log missing value rate."""
        if column not in df.columns:
            return
        missing = df[column].isna().sum()
        pct = (missing / len(df)) * 100
        if pct > 10:
            self.issues.append(DataQualityIssue(
                'warning', 'missing_data',
                f"High missing rate in {name}: {missing:,} ({pct:.1f}%)",
                {'column': column, 'missing': int(missing), 'pct': float(pct)}
            ))
            logger.warning(f"  WARNING: {pct:.1f}% missing in {column}")
        elif pct > 0:
            logger.info(f"  {pct:.2f}% missing in {column}")

    def generate_quality_report(self) -> str:
        """Generate quality report."""
        lines = ["\n" + "=" * 70, "DATA QUALITY REPORT (v5)", "=" * 70]
        if not self.issues:
            lines.append("\nNo significant data quality issues detected.")
        else:
            for issue in self.issues:
                lines.append(f"  {issue}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the complete v5 data preparation pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info("NEGAPRICENL DATA PREPARATION PIPELINE v5")
        logger.info("=" * 70)
        logger.info(f"Start: {datetime.now()}")

        try:
            df = self.merge_all_data()
            self.unified_df = df

            report = self.generate_quality_report()
            print(report)

            # Save
            output_path = PROCESSED_DATA_DIR / "unified_dataset_v5.csv"
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
            logger.info("DATA PREPARATION v5 COMPLETE")
            logger.info("=" * 70)

            return df

        except Exception as e:
            logger.error(f"ERROR in data preparation v5: {e}", exc_info=True)
            raise


def main():
    pipeline = DataPreparationV5()
    df = pipeline.run()

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
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
