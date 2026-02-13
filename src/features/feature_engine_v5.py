"""
Feature Engineering Module v5 for NegaPriceNL Project

D-1 12:00 CET auction-aligned features at 15-minute resolution.

ALL features are guaranteed available at the DA auction deadline (D-1 noon):
- Temporal: deterministic properties of the delivery quarter-hour
- Forecast: DA weather/generation/load forecasts for the delivery QH
- Forecast-derived: capacity factors, penetration ratios from forecasts
- Lagged: D-2 and D-7 same-QH prices (shift >= 192 periods)
- Snapshot: D-1 morning and D-2 daily aggregates (groupby + broadcast)

NO delivery-day actuals are used as features.
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import holidays

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    INSTALLED_SOLAR_CAPACITY_MW,
    INSTALLED_WIND_CAPACITY_MW,
    PERIODS_PER_HOUR,
)


class NegativePriceFeatureEngineV5:
    """
    D-1 auction-safe feature engine at 15-minute resolution.

    Every feature is available at D-1 12:00 CET for all delivery QHs on day D.
    """

    def __init__(
        self,
        installed_solar_mw: float = INSTALLED_SOLAR_CAPACITY_MW,
        installed_wind_mw: float = INSTALLED_WIND_CAPACITY_MW,
        periods_per_hour: int = PERIODS_PER_HOUR,
    ):
        self.installed_solar_mw = installed_solar_mw
        self.installed_wind_mw = installed_wind_mw
        self.pph = periods_per_hour  # 4 for 15-min

        self.nl_holidays = holidays.Netherlands()

        # Feature name tracking
        self._temporal_features: List[str] = []
        self._forecast_raw_features: List[str] = []
        self._forecast_derived_features: List[str] = []
        self._lagged_features: List[str] = []
        self._snapshot_features: List[str] = []

    # ------------------------------------------------------------------
    # A. Temporal features (16) — deterministic
    # ------------------------------------------------------------------

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['hour'] = df.index.hour
        df['qh_of_hour'] = df.index.minute // 15  # 0-3
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday'] = df.index.map(
            lambda x: 1 if x.date() in self.nl_holidays else 0
        )

        # Cyclical encodings
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Time-of-day buckets
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['is_morning_ramp'] = ((df['hour'] >= 6) & (df['hour'] < 10)).astype(int)
        df['is_solar_peak'] = ((df['hour'] >= 10) & (df['hour'] < 15)).astype(int)
        df['is_evening_ramp'] = ((df['hour'] >= 17) & (df['hour'] < 21)).astype(int)

        df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
        df['day_of_month'] = df.index.day

        self._temporal_features = [
            'hour', 'qh_of_hour', 'day_of_week', 'month',
            'is_weekend', 'is_holiday',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'is_night', 'is_morning_ramp', 'is_solar_peak', 'is_evening_ramp',
            'quarter', 'day_of_month',
        ]

        return df

    # ------------------------------------------------------------------
    # B. Forecast raw features (12) — pass-through from unified dataset
    # ------------------------------------------------------------------

    def register_forecast_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Register forecast columns already in the unified dataset as features."""
        forecast_cols = [
            'solar_forecast_mw', 'wind_onshore_forecast_mw',
            'wind_offshore_forecast_mw', 'wind_forecast_mw',
            'load_forecast_mw', 'total_generation_forecast_mw',
            'forecast_ghi_wm2', 'forecast_wind_speed_100m_ms',
            'forecast_temperature_2m_c', 'forecast_cloud_cover_pct',
            'forecast_pressure_hpa', 'forecast_humidity_pct',
        ]
        self._forecast_raw_features = [c for c in forecast_cols if c in df.columns]
        return df

    # ------------------------------------------------------------------
    # C. Forecast-derived features (10) — computed from forecast columns
    # ------------------------------------------------------------------

    def create_forecast_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Capacity factors
        df['forecast_solar_cf'] = (
            df['solar_forecast_mw'] / self.installed_solar_mw
        ).clip(0, 1)

        df['forecast_wind_cf'] = (
            df['wind_forecast_mw'] / self.installed_wind_mw
        ).clip(0, 1)

        # Total RES forecast
        df['forecast_res_total_mw'] = (
            df['solar_forecast_mw'].fillna(0) + df['wind_forecast_mw'].fillna(0)
        )

        # RES penetration and surplus (forecast-based)
        df['forecast_res_penetration'] = np.where(
            df['load_forecast_mw'] > 0,
            df['forecast_res_total_mw'] / df['load_forecast_mw'],
            0
        )

        df['forecast_res_surplus_mw'] = (
            df['forecast_res_total_mw'] - df['load_forecast_mw'].fillna(0)
        )

        # Interaction features (all forecast-based, D-1 safe)
        df['forecast_solar_x_weekend'] = df['forecast_solar_cf'] * df['is_weekend']
        df['forecast_duck_curve_hour'] = np.abs(df['hour'] - 13)

        df['forecast_high_res_solar_peak'] = (
            (df['forecast_res_penetration'] > 0.7) & (df['is_solar_peak'] == 1)
        ).astype(int)

        df['forecast_wind_x_night'] = df['forecast_wind_cf'] * df['is_night']

        df['forecast_weekend_solar_peak'] = (
            df['is_weekend'] * df['is_solar_peak'] * df['forecast_solar_cf']
        )

        self._forecast_derived_features = [
            'forecast_solar_cf', 'forecast_wind_cf',
            'forecast_res_total_mw', 'forecast_res_penetration',
            'forecast_res_surplus_mw',
            'forecast_solar_x_weekend', 'forecast_duck_curve_hour',
            'forecast_high_res_solar_peak', 'forecast_wind_x_night',
            'forecast_weekend_solar_peak',
        ]

        return df

    # ------------------------------------------------------------------
    # D. Same-QH lagged features (4) — shift >= 48h
    # ------------------------------------------------------------------

    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        d2 = 48 * self.pph   # 192 periods = 48 hours
        d7 = 168 * self.pph  # 672 periods = 7 days

        df['price_d2_same_qh'] = df['price_eur_mwh'].shift(d2)
        df['price_d7_same_qh'] = df['price_eur_mwh'].shift(d7)
        df['is_negative_d2_same_qh'] = df['is_negative_price'].shift(d2)
        df['is_negative_d7_same_qh'] = df['is_negative_price'].shift(d7)

        self._lagged_features = [
            'price_d2_same_qh', 'price_d7_same_qh',
            'is_negative_d2_same_qh', 'is_negative_d7_same_qh',
        ]

        return df

    # ------------------------------------------------------------------
    # E. D-1 morning snapshot features (13) — daily groupby + broadcast
    # ------------------------------------------------------------------

    def create_snapshot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['delivery_date'] = df.index.normalize()

        freq = pd.Timedelta(minutes=15)
        unique_dates = sorted(df['delivery_date'].unique())

        snapshot_rows = []

        for d in unique_dates:
            # D-1 morning: D-1 00:00 UTC to D-1 11:45 UTC (48 QH periods)
            d1_start = d - pd.Timedelta(hours=24)
            d1_cutoff = d - pd.Timedelta(hours=12) - freq  # D-1 11:45

            # D-2 full day: D-2 00:00 to D-2 23:45 UTC
            d2_start = d - pd.Timedelta(hours=48)
            d2_end = d - pd.Timedelta(hours=24) - freq  # D-2 23:45

            d1_morning = df.loc[d1_start:d1_cutoff]
            d2_full = df.loc[d2_start:d2_end]

            row = {'delivery_date': d}

            # D-1 morning price stats
            if len(d1_morning) > 0 and 'price_eur_mwh' in d1_morning.columns:
                prices_d1 = d1_morning['price_eur_mwh'].dropna()
                if len(prices_d1) > 0:
                    row['price_d1_latest'] = prices_d1.iloc[-1]
                    row['price_d1_morning_mean'] = prices_d1.mean()
                    row['price_d1_morning_min'] = prices_d1.min()
                    row['neg_count_d1_morning'] = (prices_d1 < 0).sum()

            # D-2 full day price stats
            if len(d2_full) > 0 and 'price_eur_mwh' in d2_full.columns:
                prices_d2 = d2_full['price_eur_mwh'].dropna()
                if len(prices_d2) > 0:
                    row['price_d2_mean'] = prices_d2.mean()
                    row['price_d2_std'] = prices_d2.std()
                    row['price_d2_min'] = prices_d2.min()
                    row['price_d2_max'] = prices_d2.max()
                    row['neg_count_d2'] = (prices_d2 < 0).sum()

            # D-2 generation actuals (known at D-1 noon)
            if len(d2_full) > 0:
                if 'solar_generation_mw' in d2_full.columns:
                    solar = d2_full['solar_generation_mw'].dropna()
                    if len(solar) > 0:
                        row['solar_gen_d2_mean'] = solar.mean()
                if 'wind_generation_mw' in d2_full.columns:
                    wind = d2_full['wind_generation_mw'].dropna()
                    if len(wind) > 0:
                        row['wind_gen_d2_mean'] = wind.mean()
                if 'flow_NL_DE_mw' in d2_full.columns:
                    flow = d2_full['flow_NL_DE_mw'].dropna()
                    if len(flow) > 0:
                        row['flow_nl_de_d2_mean'] = flow.mean()

            snapshot_rows.append(row)

        snapshot_df = pd.DataFrame(snapshot_rows)

        # Rolling 7-day negative count (D-8 through D-2)
        if 'neg_count_d2' in snapshot_df.columns:
            snapshot_df['neg_count_last_7d'] = (
                snapshot_df['neg_count_d2']
                .rolling(window=7, min_periods=1)
                .sum()
            )

        # Merge back to main df
        snapshot_df = snapshot_df.set_index('delivery_date')
        df = df.join(snapshot_df, on='delivery_date')
        df = df.drop(columns=['delivery_date'])

        self._snapshot_features = [
            'price_d1_latest', 'price_d1_morning_mean', 'price_d1_morning_min',
            'neg_count_d1_morning',
            'price_d2_mean', 'price_d2_std', 'price_d2_min', 'price_d2_max',
            'neg_count_d2', 'neg_count_last_7d',
            'solar_gen_d2_mean', 'wind_gen_d2_mean', 'flow_nl_de_d2_mean',
        ]

        return df

    # ------------------------------------------------------------------
    # Transform pipeline
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full v5 feature engineering pipeline."""
        required = ['price_eur_mwh', 'is_negative_price']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = self.create_temporal_features(df)
        df = self.register_forecast_raw_features(df)
        df = self.create_forecast_derived_features(df)
        df = self.create_lagged_features(df)
        df = self.create_snapshot_features(df)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of all feature column names (excludes target/metadata)."""
        return (
            self._temporal_features
            + self._forecast_raw_features
            + self._forecast_derived_features
            + self._lagged_features
            + self._snapshot_features
        )
