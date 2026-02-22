"""
Feature Engineering Module v7 for NegaPriceNL Project

Extends v6 with price-anchoring features to address v8 systematic upward bias.

New features (Phase 1):
- price_d1_same_qh: D-1 same quarter-hour price (24h lag)
- is_negative_d1_same_qh: D-1 same QH negative indicator
- price_same_hour_7d_mean: Rolling 7-day mean price for same hour-of-day

All features are D-1 12:00 CET auction-safe.
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engine_v6 import NegativePriceFeatureEngineV6


class NegativePriceFeatureEngineV7(NegativePriceFeatureEngineV6):
    """
    D-1 auction-safe feature engine v7 — v6 features + price-anchoring features.

    Addresses v8 systematic upward bias by adding:
    - Fresher price anchors (D-1 vs D-2)
    - Regime-tracking features (7-day rolling means)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Override: Enhanced lagged features
    # ------------------------------------------------------------------

    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features including v6 features + D-1 same-QH price."""
        # Call parent v6 lagged features first
        df = super().create_lagged_features(df)

        # D-1 same quarter-hour lag (24h = 96 periods for 15-min resolution)
        d1 = 24 * self.pph  # 96 periods

        # Feature: D-1 same-QH price (24h-fresh price anchor)
        df['price_d1_same_qh'] = df['price_eur_mwh'].shift(d1)

        # Feature: D-1 same-QH negative indicator
        df['is_negative_d1_same_qh'] = df['is_negative_price'].shift(d1)

        # Add to feature list
        self._lagged_features.extend([
            'price_d1_same_qh',
            'is_negative_d1_same_qh'
        ])

        return df

    # ------------------------------------------------------------------
    # Override: Enhanced snapshot features
    # ------------------------------------------------------------------

    def create_snapshot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create snapshot features including v6 features + 7-day rolling same-hour mean."""
        # Call parent v6 snapshot features first
        df = super().create_snapshot_features(df)

        # Add delivery date for grouping
        df['delivery_date'] = df.index.normalize()

        freq = pd.Timedelta(minutes=15)
        unique_dates = sorted(df['delivery_date'].unique())

        snapshot_rows = []

        for d in unique_dates:
            # 7-day window: D-2 00:00 through D-8 23:45
            # D-2 start: 48 hours before delivery day
            d2_start = d - pd.Timedelta(hours=48)
            # D-8 end: 8 days before delivery + 23:45 of that day
            d8_end = d - pd.Timedelta(days=8, hours=0)

            week_window = df.loc[d8_end:d2_start]

            row = {'delivery_date': d}

            # For each hour, compute mean price across the 7-day window
            for hour in range(24):
                hour_mask = week_window.index.hour == hour
                hour_prices = week_window.loc[hour_mask, 'price_eur_mwh'].dropna()

                if len(hour_prices) > 0:
                    row[f'price_same_hour_{hour}_7d_mean'] = hour_prices.mean()

            snapshot_rows.append(row)

        # Create snapshot dataframe
        snapshot_df = pd.DataFrame(snapshot_rows).set_index('delivery_date')

        # Merge back to main df
        df = df.join(snapshot_df, on='delivery_date')

        # Map hour-specific columns to single feature based on hour of day
        def get_hour_mean(row):
            hour = row.name.hour
            col_name = f'price_same_hour_{hour}_7d_mean'
            return row.get(col_name)

        df['price_same_hour_7d_mean'] = df.apply(get_hour_mean, axis=1)

        # Clean up temporary hour-specific columns
        temp_cols = [f'price_same_hour_{h}_7d_mean' for h in range(24)]
        df = df.drop(columns=temp_cols + ['delivery_date'], errors='ignore')

        # Add to feature list
        self._snapshot_features.append('price_same_hour_7d_mean')

        return df

    # ------------------------------------------------------------------
    # No override needed for transform - uses parent v6 transform
    # which calls our overridden create_lagged_features and
    # create_snapshot_features
    # ------------------------------------------------------------------
