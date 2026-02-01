"""
Feature Engineering Module for NegaPriceNL Project

Creates features for negative electricity price prediction including:
- Temporal features (hour, day, month, holidays)
- Renewable generation features (capacity factors, penetration)
- Lag features (historical price and generation patterns)
- Interaction features (combined indicators)
"""

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import holidays

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    INSTALLED_SOLAR_CAPACITY_MW,
    INSTALLED_WIND_CAPACITY_MW
)


class NegativePriceFeatureEngine:
    """
    Creates features for negative electricity price prediction.

    Features are designed to capture:
    1. Temporal patterns (time of day, day of week, seasonality)
    2. Renewable generation dynamics (solar/wind capacity factors, penetration)
    3. Historical patterns (lagged prices and generation)
    4. Interaction effects (combined indicators)
    """

    def __init__(
        self,
        installed_solar_mw: float = INSTALLED_SOLAR_CAPACITY_MW,
        installed_wind_mw: float = INSTALLED_WIND_CAPACITY_MW
    ):
        """
        Initialize the feature engine.

        Parameters
        ----------
        installed_solar_mw : float
            Installed solar capacity in MW (default from settings)
        installed_wind_mw : float
            Installed wind capacity in MW (default from settings)
        """
        self.installed_solar_mw = installed_solar_mw
        self.installed_wind_mw = installed_wind_mw
        self.nl_holidays = holidays.Netherlands()

        # Track which features were created
        self._temporal_features: List[str] = []
        self._renewable_features: List[str] = []
        self._lag_features: List[str] = []
        self._interaction_features: List[str] = []

        # v3 enhanced features
        self._enhanced_lag_features: List[str] = []
        self._momentum_features: List[str] = []
        self._enhanced_rolling_features: List[str] = []
        self._enhanced_temporal_features: List[str] = []
        self._enhanced_interaction_features: List[str] = []

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime index.

        Features created:
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - month: Month (1-12)
        - is_weekend: Binary weekend indicator
        - is_holiday: Binary Dutch holiday indicator
        - hour_sin, hour_cos: Cyclical hour encoding
        - month_sin, month_cos: Cyclical month encoding
        """
        df = df.copy()

        # Basic temporal features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Dutch holiday indicator
        df['is_holiday'] = df.index.map(
            lambda x: 1 if x.date() in self.nl_holidays else 0
        )

        # Cyclical encoding for hour (captures continuity: 23:00 is close to 00:00)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for month (captures seasonality)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        self._temporal_features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]

        return df

    def create_renewable_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create renewable generation related features.

        Features created:
        - total_res_mw: Total renewable generation (solar + wind)
        - solar_capacity_factor: Solar generation / installed capacity
        - wind_capacity_factor: Wind generation / installed capacity
        - res_penetration: Renewable share of load (KEY DRIVER)
        - res_surplus_mw: Renewable generation minus load
        """
        df = df.copy()

        # Total renewable generation
        df['total_res_mw'] = (
            df['solar_generation_mw'].fillna(0) +
            df['wind_generation_mw'].fillna(0)
        )

        # Capacity factors (normalize by installed capacity)
        df['solar_capacity_factor'] = (
            df['solar_generation_mw'] / self.installed_solar_mw
        ).clip(0, 1)  # Clip to valid range

        df['wind_capacity_factor'] = (
            df['wind_generation_mw'] / self.installed_wind_mw
        ).clip(0, 1)

        # RES penetration - key driver of negative prices
        # Use actual load to calculate penetration
        df['res_penetration'] = np.where(
            df['load_mw'] > 0,
            df['total_res_mw'] / df['load_mw'],
            0
        )

        # RES surplus (positive means more generation than load)
        df['res_surplus_mw'] = df['total_res_mw'] - df['load_mw']

        self._renewable_features = [
            'total_res_mw', 'solar_capacity_factor', 'wind_capacity_factor',
            'res_penetration', 'res_surplus_mw'
        ]

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        price_lags: List[int] = [24, 168],
        generation_lags: List[int] = [24]
    ) -> pd.DataFrame:
        """
        Create lagged features for historical patterns.

        Features created:
        - price_lag_Xh: Price X hours ago
        - solar_lag_Xh: Solar generation X hours ago
        - wind_lag_Xh: Wind generation X hours ago
        - is_negative_lag_Xh: Negative price indicator X hours ago
        - price_rolling_mean_24h: 24-hour rolling mean price
        - price_rolling_std_24h: 24-hour rolling std (volatility)
        - negative_hours_last_7d: Count of negative hours in past week
        """
        df = df.copy()
        lag_features = []

        # Price lags
        for lag in price_lags:
            col_name = f'price_lag_{lag}h'
            df[col_name] = df['price_eur_mwh'].shift(lag)
            lag_features.append(col_name)

        # Generation lags
        for lag in generation_lags:
            solar_col = f'solar_lag_{lag}h'
            wind_col = f'wind_lag_{lag}h'
            df[solar_col] = df['solar_generation_mw'].shift(lag)
            df[wind_col] = df['wind_generation_mw'].shift(lag)
            lag_features.extend([solar_col, wind_col])

        # Negative price indicator lags
        for lag in price_lags:
            col_name = f'is_negative_lag_{lag}h'
            df[col_name] = df['is_negative_price'].shift(lag)
            lag_features.append(col_name)

        # Rolling statistics - SHIFTED to exclude current observation (prevent leakage)
        df['price_rolling_mean_24h'] = (
            df['price_eur_mwh'].shift(1).rolling(window=24, min_periods=1).mean()
        )
        df['price_rolling_std_24h'] = (
            df['price_eur_mwh'].shift(1).rolling(window=24, min_periods=1).std()
        )
        lag_features.extend(['price_rolling_mean_24h', 'price_rolling_std_24h'])

        # Count of negative hours in last 7 days (168 hours) - SHIFTED to exclude current target
        df['negative_hours_last_7d'] = (
            df['is_negative_price'].shift(1).rolling(window=168, min_periods=1).sum()
        )
        lag_features.append('negative_hours_last_7d')

        self._lag_features = lag_features

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction and derived features.

        Features created:
        - solar_x_weekend: High solar on weekend (low demand) indicator
        - res_load_ratio: RES / load forecast ratio (using lagged forecast)
        - duck_curve_hour: Distance from solar peak hour (13:00)
        """
        df = df.copy()

        # Solar on weekend interaction (high solar + low demand = negative prices)
        df['solar_x_weekend'] = df['solar_capacity_factor'] * df['is_weekend']

        # RES to load forecast ratio (using lagged forecast for proper alignment)
        # Handle division by zero
        df['res_load_ratio'] = np.where(
            df['load_forecast_mw_lag_24h'] > 0,
            df['total_res_mw'] / df['load_forecast_mw_lag_24h'],
            0
        )

        # Duck curve position (distance from typical solar peak at 13:00)
        df['duck_curve_hour'] = np.abs(df['hour'] - 13)

        self._interaction_features = [
            'solar_x_weekend', 'res_load_ratio', 'duck_curve_hour'
        ]

        return df

    def create_enhanced_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fine-grained lag features for short-term patterns.

        Features created:
        - price_lag_1h, 2h, 4h, 6h, 12h: Recent price history
        - res_penetration_lag_1h, 4h, 24h: RES penetration history
        - solar_lag_1h, wind_lag_1h: Recent generation
        """
        df = df.copy()
        enhanced_lag_features = []

        # Fine-grained price lags
        for lag in [1, 2, 4, 6, 12]:
            col_name = f'price_lag_{lag}h'
            df[col_name] = df['price_eur_mwh'].shift(lag)
            enhanced_lag_features.append(col_name)

        # RES penetration lags (key driver history)
        for lag in [1, 4, 24]:
            col_name = f'res_penetration_lag_{lag}h'
            df[col_name] = df['res_penetration'].shift(lag)
            enhanced_lag_features.append(col_name)

        # Short-term generation lags
        df['solar_lag_1h'] = df['solar_generation_mw'].shift(1)
        df['wind_lag_1h'] = df['wind_generation_mw'].shift(1)
        enhanced_lag_features.extend(['solar_lag_1h', 'wind_lag_1h'])

        self._enhanced_lag_features = enhanced_lag_features
        return df

    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rate-of-change and momentum features.

        LEAK-FREE VERSION: All features use only lagged values (no current price).

        Features created:
        - price_change_lag1h: Change from 2h ago to 1h ago (lagged momentum)
        - price_change_lag4h: Change from 5h ago to 1h ago (lagged momentum)
        - price_pct_change_lag1h: Percentage change (lagged)
        - solar_change_1h, wind_change_1h: Generation ramping (current vs 1h ago - OK for nowcast)
        - res_change_1h: Total RES change
        - load_change_1h: Load ramping
        """
        df = df.copy()
        momentum_features = []

        # Price momentum - LAGGED to prevent leakage
        # price_change_lag1h = price_1h_ago - price_2h_ago (no current price!)
        df['price_change_lag1h'] = df['price_eur_mwh'].shift(1) - df['price_eur_mwh'].shift(2)
        df['price_change_lag4h'] = df['price_eur_mwh'].shift(1) - df['price_eur_mwh'].shift(5)

        # Percentage change - LAGGED (handle division by zero)
        df['price_pct_change_lag1h'] = np.where(
            df['price_eur_mwh'].shift(2).abs() > 0.01,
            df['price_change_lag1h'] / df['price_eur_mwh'].shift(2).abs(),
            0
        )
        momentum_features.extend(['price_change_lag1h', 'price_change_lag4h', 'price_pct_change_lag1h'])

        # Generation ramping (current vs 1h ago - OK for nowcast, RES data is observable)
        df['solar_change_1h'] = df['solar_generation_mw'] - df['solar_generation_mw'].shift(1)
        df['wind_change_1h'] = df['wind_generation_mw'] - df['wind_generation_mw'].shift(1)
        df['res_change_1h'] = df['total_res_mw'] - df['total_res_mw'].shift(1)
        momentum_features.extend(['solar_change_1h', 'wind_change_1h', 'res_change_1h'])

        # Load ramping (current vs 1h ago - OK, load is observable)
        df['load_change_1h'] = df['load_mw'] - df['load_mw'].shift(1)
        momentum_features.append('load_change_1h')

        self._momentum_features = momentum_features
        return df

    def create_enhanced_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced rolling statistics.

        LEAK-FREE VERSION: All rolling windows exclude current observation via .shift(1).

        Features created:
        - price_rolling_min/max_24h: Price range (previous 24h, excluding current)
        - price_range_24h: Price volatility indicator
        - negative_hours_last_24h, 48h: Multi-window negative counts (excluding current target!)
        - res_penetration_rolling_max_24h: Peak RES penetration (previous 24h)
        """
        df = df.copy()
        enhanced_rolling_features = []

        # Price range indicators - SHIFTED to exclude current price
        df['price_rolling_min_24h'] = df['price_eur_mwh'].shift(1).rolling(window=24, min_periods=1).min()
        df['price_rolling_max_24h'] = df['price_eur_mwh'].shift(1).rolling(window=24, min_periods=1).max()
        df['price_range_24h'] = df['price_rolling_max_24h'] - df['price_rolling_min_24h']
        enhanced_rolling_features.extend(['price_rolling_min_24h', 'price_rolling_max_24h', 'price_range_24h'])

        # Multi-window negative hour counts - SHIFTED to exclude current target (CRITICAL!)
        df['negative_hours_last_24h'] = df['is_negative_price'].shift(1).rolling(window=24, min_periods=1).sum()
        df['negative_hours_last_48h'] = df['is_negative_price'].shift(1).rolling(window=48, min_periods=1).sum()
        enhanced_rolling_features.extend(['negative_hours_last_24h', 'negative_hours_last_48h'])

        # RES penetration statistics - SHIFTED (RES penetration itself is OK, but rolling should exclude current)
        df['res_penetration_rolling_max_24h'] = df['res_penetration'].shift(1).rolling(window=24, min_periods=1).max()
        enhanced_rolling_features.append('res_penetration_rolling_max_24h')

        self._enhanced_rolling_features = enhanced_rolling_features
        return df

    def create_enhanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-of-day bucket features.

        Features created:
        - is_night: 00:00-06:00 (low demand, wind-dominated)
        - is_morning_ramp: 06:00-10:00 (demand ramp-up)
        - is_solar_peak: 10:00-15:00 (solar peak, negative price risk)
        - is_evening_ramp: 17:00-21:00 (evening peak demand)
        - quarter: 1-4
        - day_of_month: 1-31
        - is_end_of_month: last 3 days of month
        """
        df = df.copy()
        enhanced_temporal_features = []

        # Time-of-day buckets
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['is_morning_ramp'] = ((df['hour'] >= 6) & (df['hour'] < 10)).astype(int)
        df['is_solar_peak'] = ((df['hour'] >= 10) & (df['hour'] < 15)).astype(int)
        df['is_evening_ramp'] = ((df['hour'] >= 17) & (df['hour'] < 21)).astype(int)
        enhanced_temporal_features.extend(['is_night', 'is_morning_ramp', 'is_solar_peak', 'is_evening_ramp'])

        # Additional temporal features
        df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
        df['day_of_month'] = df.index.day
        df['is_end_of_month'] = (df['day_of_month'] >= 28).astype(int)
        enhanced_temporal_features.extend(['quarter', 'day_of_month', 'is_end_of_month'])

        self._enhanced_temporal_features = enhanced_temporal_features
        return df

    def create_enhanced_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create improved interaction terms.

        Features created:
        - high_res_solar_peak: High RES during solar peak hours
        - wind_x_night: Wind generation at night
        - weekend_solar_peak: Weekend + solar peak + high solar
        - is_res_surplus: Binary indicator for RES > load
        - is_high_wind, is_high_solar: Extreme condition indicators
        """
        df = df.copy()
        enhanced_interaction_features = []

        # High RES during solar peak (highest negative price risk)
        df['high_res_solar_peak'] = ((df['res_penetration'] > 0.7) & (df['is_solar_peak'] == 1)).astype(int)
        enhanced_interaction_features.append('high_res_solar_peak')

        # Wind at night
        df['wind_x_night'] = df['wind_capacity_factor'] * df['is_night']
        enhanced_interaction_features.append('wind_x_night')

        # Weekend solar peak (very high risk)
        df['weekend_solar_peak'] = df['is_weekend'] * df['is_solar_peak'] * df['solar_capacity_factor']
        enhanced_interaction_features.append('weekend_solar_peak')

        # Load deficit indicator
        df['is_res_surplus'] = (df['res_surplus_mw'] > 0).astype(int)
        enhanced_interaction_features.append('is_res_surplus')

        # Extreme conditions
        df['is_high_wind'] = (df['wind_capacity_factor'] > 0.6).astype(int)
        df['is_high_solar'] = (df['solar_capacity_factor'] > 0.5).astype(int)
        enhanced_interaction_features.extend(['is_high_wind', 'is_high_solar'])

        self._enhanced_interaction_features = enhanced_interaction_features
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with datetime index and required columns:
            - price_eur_mwh
            - solar_generation_mw
            - wind_generation_mw
            - load_mw
            - load_forecast_mw_lag_24h
            - is_negative_price

        Returns
        -------
        pd.DataFrame
            Dataframe with all engineered features added
        """
        # Validate required columns
        required_cols = [
            'price_eur_mwh', 'solar_generation_mw', 'wind_generation_mw',
            'load_mw', 'load_forecast_mw_lag_24h', 'is_negative_price'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Apply all feature creation methods (order matters for dependencies)
        df = self.create_temporal_features(df)
        df = self.create_renewable_features(df)
        df = self.create_lag_features(df)
        df = self.create_interaction_features(df)

        # v3 enhanced features
        df = self.create_enhanced_lag_features(df)
        df = self.create_momentum_features(df)
        df = self.create_enhanced_rolling_features(df)
        df = self.create_enhanced_temporal_features(df)
        df = self.create_enhanced_interaction_features(df)

        return df

    def get_feature_columns(self) -> List[str]:
        """
        Get list of all engineered feature column names.

        Returns
        -------
        List[str]
            List of feature column names (excludes target and metadata)
        """
        features = (
            self._temporal_features +
            self._renewable_features +
            self._lag_features +
            self._interaction_features
        )

        # Add v3 enhanced features if they exist
        if hasattr(self, '_enhanced_lag_features'):
            features += self._enhanced_lag_features
        if hasattr(self, '_momentum_features'):
            features += self._momentum_features
        if hasattr(self, '_enhanced_rolling_features'):
            features += self._enhanced_rolling_features
        if hasattr(self, '_enhanced_temporal_features'):
            features += self._enhanced_temporal_features
        if hasattr(self, '_enhanced_interaction_features'):
            features += self._enhanced_interaction_features

        return features

    def get_all_feature_columns(self, include_raw: bool = True) -> List[str]:
        """
        Get all feature columns including raw data columns.

        Parameters
        ----------
        include_raw : bool
            Whether to include raw data columns (weather, load, etc.)

        Returns
        -------
        List[str]
            List of all feature column names
        """
        engineered = self.get_feature_columns()

        if include_raw:
            raw_features = [
                'ghi_wm2', 'wind_speed_100m_ms', 'temperature_2m_c',
                'cloud_cover_pct', 'pressure_hpa', 'humidity_pct',
                'load_mw', 'load_forecast_mw', 'load_forecast_mw_lag_24h',
                'solar_generation_mw', 'wind_generation_mw'
            ]
            return raw_features + engineered

        return engineered
