"""
Data Preparation Module v2 for NegaPriceNL Project

Identical to data_preparation.py, but adds total_generation_mw column
from nl_generation_all_2019_2025.csv (sum of all fuel types' actual generation).

This script:
1. Loads all available data sources
2. Aligns them on a common hourly time index
3. Adds total generation from the full generation mix
4. Performs quality checks and reports anomalies
5. Creates target variable (negative price indicator)
6. Saves the unified dataset for feature engineering
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    ENTSOE_DATA_DIR,
    WEATHER_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATA_START_DATE,
    DATA_END_DATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityIssue:
    """Container for data quality issues that need user review."""

    def __init__(self, severity: str, category: str, description: str, details: Dict = None):
        self.severity = severity  # 'critical', 'warning', 'info'
        self.category = category  # 'missing_data', 'outlier', 'suspicious_pattern', etc.
        self.description = description
        self.details = details or {}
        self.timestamp = datetime.now()

    def __str__(self):
        return f"[{self.severity.upper()}] {self.category}: {self.description}"


class DataPreparation:
    """
    Unified data preparation pipeline for NegaPriceNL project (v2).

    Adds total_generation_mw from the full generation mix CSV.
    """

    def __init__(self):
        """Initialize the data preparation pipeline."""
        self.issues: List[DataQualityIssue] = []
        self.data_sources: Dict[str, pd.DataFrame] = {}
        self.unified_df: Optional[pd.DataFrame] = None

        logger.info("DataPreparation v2 pipeline initialized")

    def load_price_data(self) -> pd.DataFrame:
        """Load day-ahead price data for Netherlands."""
        logger.info("Loading price data...")

        price_file = ENTSOE_DATA_DIR / "nl_day_ahead_prices_2019_2025.csv"
        if not price_file.exists():
            raise FileNotFoundError(f"Price data not found: {price_file}")

        df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'

        # Rename column for clarity
        if 'price' in df.columns:
            df = df.rename(columns={'price': 'price_eur_mwh'})

        logger.info(f"  Loaded {len(df):,} price records from {df.index.min()} to {df.index.max()}")

        # Quality checks
        self._check_missing_values(df, 'price_eur_mwh', 'prices')
        self._check_outliers(df, 'price_eur_mwh', 'prices')

        return df

    def load_generation_actual(self) -> pd.DataFrame:
        """Load actual generation data (solar and wind)."""
        logger.info("Loading actual generation data...")

        # Solar
        solar_file = ENTSOE_DATA_DIR / "nl_solar_generation_2019_2025.csv"
        if not solar_file.exists():
            raise FileNotFoundError(f"Solar generation data not found: {solar_file}")

        solar = pd.read_csv(solar_file, index_col=0, parse_dates=True)
        solar.index = pd.to_datetime(solar.index, utc=True)
        solar['solar_generation_mw'] = pd.to_numeric(solar['solar_generation_mw'], errors='coerce')

        # Wind
        wind_file = ENTSOE_DATA_DIR / "nl_wind_generation_2019_2025.csv"
        if not wind_file.exists():
            raise FileNotFoundError(f"Wind generation data not found: {wind_file}")

        wind = pd.read_csv(wind_file, index_col=0, parse_dates=True)
        wind.index = pd.to_datetime(wind.index, utc=True)
        wind['wind_generation_mw'] = pd.to_numeric(wind['wind_generation_mw'], errors='coerce')

        # Merge
        generation = pd.merge(solar, wind, left_index=True, right_index=True, how='outer')
        generation.index.name = 'datetime'

        logger.info(f"  Loaded {len(generation):,} generation records")
        logger.info(f"    Solar: mean={generation['solar_generation_mw'].mean():.2f} MW, "
                   f"max={generation['solar_generation_mw'].max():.2f} MW")
        logger.info(f"    Wind: mean={generation['wind_generation_mw'].mean():.2f} MW, "
                   f"max={generation['wind_generation_mw'].max():.2f} MW")

        # Quality checks
        self._check_missing_values(generation, 'solar_generation_mw', 'solar generation')
        self._check_missing_values(generation, 'wind_generation_mw', 'wind generation')
        self._check_suspicious_low_values(generation, 'wind_generation_mw', 'wind generation')

        return generation

    def load_total_generation(self) -> Optional[pd.DataFrame]:
        """
        Load total actual generation from the full generation mix CSV.

        Reads nl_generation_all_2019_2025.csv (15-min resolution, multi-level header),
        selects only 'Actual Aggregated' columns (not 'Actual Consumption'),
        sums across all fuel types, converts CET/CEST to UTC, and resamples
        to hourly using .first() to avoid data leakage.

        Returns:
            DataFrame with single 'total_generation_mw' column at hourly UTC index,
            or None if the file is not found.
        """
        logger.info("Loading total generation data...")

        gen_all_file = ENTSOE_DATA_DIR / "nl_generation_all_2019_2025.csv"
        if not gen_all_file.exists():
            logger.warning(f"  Total generation data not found: {gen_all_file}")
            return None

        # Read with multi-level header (row 0 = fuel type, row 1 = Actual Aggregated/Consumption)
        df = pd.read_csv(gen_all_file, header=[0, 1], index_col=0, parse_dates=True)

        # Convert index to UTC (source is CET/CEST)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'

        logger.info(f"  Raw records: {len(df):,} (15-min resolution)")
        logger.info(f"  Multi-level columns: {len(df.columns)}")

        # Select only 'Actual Aggregated' sub-columns (generation output, not self-consumption)
        actual_agg_cols = [col for col in df.columns if col[1] == 'Actual Aggregated']
        df_actual = df[actual_agg_cols].copy()

        # Flatten column names to just the fuel type
        df_actual.columns = [col[0] for col in df_actual.columns]

        logger.info(f"  Fuel types (Actual Aggregated): {list(df_actual.columns)}")

        # Convert all to numeric
        for col in df_actual.columns:
            df_actual[col] = pd.to_numeric(df_actual[col], errors='coerce')

        # Sum across all fuel types for total generation
        df_actual['total_generation_mw'] = df_actual.sum(axis=1)

        logger.info(f"  Total generation (15-min): mean={df_actual['total_generation_mw'].mean():.1f} MW, "
                   f"max={df_actual['total_generation_mw'].max():.1f} MW")

        # Leak-free resample to hourly: take only the :00 value of each hour
        # This ensures we only use the observation at the start of the hour,
        # not future 15-min values within the same hour
        result = df_actual[['total_generation_mw']].resample('h').first()

        logger.info(f"  After hourly resample (.first()): {len(result):,} records")

        # Quality checks
        self._check_missing_values(result, 'total_generation_mw', 'total generation')

        return result

    def load_generation_forecast(self) -> Optional[pd.DataFrame]:
        """Load generation forecast data."""
        logger.info("Loading generation forecast data...")

        forecast_file = ENTSOE_DATA_DIR / "nl_generation_forecast_2019_2025.csv"
        if not forecast_file.exists():
            logger.warning(f"  Generation forecast data not found: {forecast_file}")
            return None

        df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'datetime'

        # Rename columns for clarity
        if 'generation_forecast_mw' in df.columns:
            df = df.rename(columns={'generation_forecast_mw': 'total_generation_forecast_mw'})

        logger.info(f"  Loaded {len(df):,} forecast records")
        logger.info(f"    Columns: {list(df.columns)}")

        # Quality checks
        for col in df.columns:
            self._check_missing_values(df, col, f'generation forecast ({col})')

        return df

    def load_weather_data(self) -> pd.DataFrame:
        """Load weather data (Amsterdam location)."""
        logger.info("Loading weather data...")

        weather_file = WEATHER_DATA_DIR / "nl_weather_amsterdam_20190101_20251231.csv"
        if not weather_file.exists():
            raise FileNotFoundError(f"Weather data not found: {weather_file}")

        df = pd.read_csv(weather_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df.index = pd.to_datetime(df.index, utc=True)

        # Select key weather variables
        weather_cols = [
            'ghi_wm2',                    # Global Horizontal Irradiance
            'wind_speed_100m_ms',         # Wind speed at hub height
            'temperature_2m_c',           # Temperature
            'cloud_cover_pct',            # Cloud cover
            'pressure_hpa',               # Pressure
            'humidity_pct'                # Humidity
        ]

        available_cols = [col for col in weather_cols if col in df.columns]
        df = df[available_cols]

        logger.info(f"  Loaded {len(df):,} weather records")
        logger.info(f"    Variables: {available_cols}")

        # Quality checks
        for col in available_cols:
            self._check_missing_values(df, col, f'weather ({col})')

        return df

    def load_load_data(self) -> Optional[pd.DataFrame]:
        """Load load data (actual and forecast) if available."""
        logger.info("Loading load data...")

        # Actual load
        load_actual_file = ENTSOE_DATA_DIR / "nl_load_2019_2025.csv"
        load_forecast_file = ENTSOE_DATA_DIR / "nl_load_forecast_2019_2025.csv"

        load_actual = None
        load_forecast = None

        if load_actual_file.exists():
            load_actual = pd.read_csv(load_actual_file, index_col=0, parse_dates=True)
            load_actual.index = pd.to_datetime(load_actual.index, utc=True)
            logger.info(f"  Loaded {len(load_actual):,} actual load records")
        else:
            logger.warning(f"  Actual load data not found (still downloading): {load_actual_file}")

        if load_forecast_file.exists():
            load_forecast = pd.read_csv(load_forecast_file, index_col=0, parse_dates=True)
            load_forecast.index = pd.to_datetime(load_forecast.index, utc=True)

            # Create 24-hour backward-shifted forecast for ML training
            # Shift backwards so the last 24h are NaN (no future forecast available)
            # 15-min data: 96 periods = 24 hours
            load_forecast['load_forecast_mw_lag_24h'] = load_forecast['load_forecast_mw'].shift(-96)

            logger.info(f"  Loaded {len(load_forecast):,} load forecast records")
            logger.info(f"  Created load_forecast_mw_lag_24h (24h backward shift for ML training)")
        else:
            logger.warning(f"  Load forecast data not found (still downloading): {load_forecast_file}")

        if load_actual is None and load_forecast is None:
            return None

        # Merge if both exist
        if load_actual is not None and load_forecast is not None:
            load_data = pd.merge(load_actual, load_forecast, left_index=True, right_index=True,
                               how='outer', suffixes=('_actual', '_forecast'))
        elif load_actual is not None:
            load_data = load_actual
        else:
            load_data = load_forecast

        load_data.index.name = 'datetime'

        # Quality checks
        for col in load_data.columns:
            self._check_missing_values(load_data, col, f'load ({col})')

        return load_data

    def merge_all_data(self) -> pd.DataFrame:
        """Merge all data sources on common time index."""
        logger.info("\n" + "="*70)
        logger.info("MERGING ALL DATA SOURCES")
        logger.info("="*70)

        # Load all data sources
        prices = self.load_price_data()
        generation = self.load_generation_actual()
        gen_forecast = self.load_generation_forecast()
        weather = self.load_weather_data()
        load_data = self.load_load_data()
        total_gen = self.load_total_generation()

        # Start with prices as base (this is our target variable)
        df = prices.copy()
        logger.info(f"\nStarting with price data: {len(df)} records")

        # Merge generation actual
        df = pd.merge(df, generation, left_index=True, right_index=True, how='left')
        logger.info(f"After merging generation: {len(df)} records")

        # Merge total generation (from full generation mix)
        if total_gen is not None:
            df = pd.merge(df, total_gen, left_index=True, right_index=True, how='left')
            logger.info(f"After merging total generation: {len(df)} records, "
                       f"total_generation_mw non-null: {df['total_generation_mw'].notna().sum()}")
        else:
            logger.warning("Total generation data not available")

        # Merge generation forecast if available
        if gen_forecast is not None:
            df = pd.merge(df, gen_forecast, left_index=True, right_index=True, how='left')
            logger.info(f"After merging generation forecast: {len(df)} records")

        # Merge weather
        df = pd.merge(df, weather, left_index=True, right_index=True, how='left')
        logger.info(f"After merging weather: {len(df)} records")

        # Merge load data if available
        if load_data is not None:
            df = pd.merge(df, load_data, left_index=True, right_index=True, how='left')
            logger.info(f"After merging load: {len(df)} records")
        else:
            logger.warning("Load data not available - will be added when download completes")

        # Create target variable
        df['is_negative_price'] = (df['price_eur_mwh'] < 0).astype(int)

        logger.info(f"\nFinal unified dataset: {len(df)} records, {len(df.columns)} columns")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Negative price events: {df['is_negative_price'].sum()} ({df['is_negative_price'].mean()*100:.2f}%)")

        return df

    def _check_missing_values(self, df: pd.DataFrame, column: str, name: str):
        """Check for missing values in a column."""
        missing_count = df[column].isna().sum()
        missing_pct = (missing_count / len(df)) * 100

        if missing_pct > 10:
            issue = DataQualityIssue(
                severity='warning',
                category='missing_data',
                description=f"High missing value rate in {name}",
                details={
                    'column': column,
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_pct),
                    'total_records': len(df)
                }
            )
            self.issues.append(issue)
            logger.warning(f"  ISSUE: {missing_pct:.2f}% missing values in {column}")
        elif missing_pct > 0:
            logger.info(f"  {missing_pct:.2f}% missing values in {column}")

    def _check_outliers(self, df: pd.DataFrame, column: str, name: str):
        """Check for outliers using IQR method."""
        data = df[column].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_pct = (len(outliers) / len(data)) * 100

        if outlier_pct > 5:
            issue = DataQualityIssue(
                severity='info',
                category='outlier',
                description=f"High outlier rate in {name}",
                details={
                    'column': column,
                    'outlier_count': int(len(outliers)),
                    'outlier_percentage': float(outlier_pct),
                    'min_value': float(data.min()),
                    'max_value': float(data.max()),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            )
            self.issues.append(issue)
            logger.info(f"  {outlier_pct:.2f}% outliers in {column} (min={data.min():.2f}, max={data.max():.2f})")

    def _check_suspicious_low_values(self, df: pd.DataFrame, column: str, name: str):
        """Check for suspiciously low values (e.g., wind generation near zero)."""
        data = df[column].dropna()
        near_zero_count = (data < 1).sum()  # Values less than 1 MW
        near_zero_pct = (near_zero_count / len(data)) * 100

        if near_zero_pct > 30:
            issue = DataQualityIssue(
                severity='warning',
                category='suspicious_pattern',
                description=f"Suspiciously high rate of near-zero values in {name}",
                details={
                    'column': column,
                    'near_zero_count': int(near_zero_count),
                    'near_zero_percentage': float(near_zero_pct),
                    'mean_value': float(data.mean()),
                    'median_value': float(data.median())
                }
            )
            self.issues.append(issue)
            logger.warning(f"  ISSUE: {near_zero_pct:.2f}% near-zero values in {column} "
                         f"(mean={data.mean():.2f}, median={data.median():.2f})")

    def generate_quality_report(self) -> str:
        """Generate a comprehensive data quality report."""
        report = []
        report.append("\n" + "="*70)
        report.append("DATA QUALITY REPORT")
        report.append("="*70)

        if not self.issues:
            report.append("\nNo significant data quality issues detected.")
        else:
            # Group by severity
            critical = [i for i in self.issues if i.severity == 'critical']
            warnings = [i for i in self.issues if i.severity == 'warning']
            info = [i for i in self.issues if i.severity == 'info']

            if critical:
                report.append(f"\nCRITICAL ISSUES ({len(critical)}):")
                for issue in critical:
                    report.append(f"  - {issue}")

            if warnings:
                report.append(f"\nWARNINGS ({len(warnings)}):")
                for issue in warnings:
                    report.append(f"  - {issue}")

            if info:
                report.append(f"\nINFORMATIONAL ({len(info)}):")
                for issue in info:
                    report.append(f"  - {issue}")

        report.append("\n" + "="*70)

        return "\n".join(report)

    def save_unified_dataset(self, df: pd.DataFrame, output_path: Path):
        """Save the unified dataset."""
        logger.info(f"\nSaving unified dataset to: {output_path}")

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        df.to_csv(output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        logger.info(f"  Saved successfully ({file_size_mb:.2f} MB)")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")

    def run(self) -> pd.DataFrame:
        """Execute the complete data preparation pipeline."""
        logger.info("\n" + "="*70)
        logger.info("NEGAPRICENL DATA PREPARATION PIPELINE v2")
        logger.info("="*70)
        logger.info(f"Start: {datetime.now()}")

        try:
            # Merge all data
            df = self.merge_all_data()
            self.unified_df = df

            # Generate quality report
            quality_report = self.generate_quality_report()
            print(quality_report)

            # Save unified dataset
            output_path = PROCESSED_DATA_DIR / "unified_dataset2.csv"
            self.save_unified_dataset(df, output_path)

            # Save quality report
            report_path = PROCESSED_DATA_DIR / "data_quality_report2.txt"
            with open(report_path, 'w') as f:
                f.write(quality_report)
            logger.info(f"\nQuality report saved to: {report_path}")

            logger.info("\n" + "="*70)
            logger.info("DATA PREPARATION v2 COMPLETE")
            logger.info("="*70)
            logger.info(f"End: {datetime.now()}")

            return df

        except Exception as e:
            logger.error(f"\nERROR in data preparation: {e}", exc_info=True)
            raise


def main():
    """Main execution function."""
    pipeline = DataPreparation()
    df = pipeline.run()

    # Print summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nMissing values:")
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
