"""
Feature Engineering Module v6 for NegaPriceNL Project

Extends v5 with NTC (Net Transfer Capacity) features derived from
day-ahead transfer capacity data for all 4 NL interconnectors.

All NTC features are D-1 safe — NTC is published before the DA auction.

New features (~10):
- Pass-through: total export/import/net NTC
- Derived: total capacity, export ratio, RES-vs-export constraint,
  BritNed/NorNed availability, D-2 capacity change
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_engine_v5 import NegativePriceFeatureEngineV5


class NegativePriceFeatureEngineV6(NegativePriceFeatureEngineV5):
    """
    D-1 auction-safe feature engine v6 — v5 features + NTC features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ntc_features: List[str] = []

    # ------------------------------------------------------------------
    # F. NTC features — day-ahead transfer capacity
    # ------------------------------------------------------------------

    def create_ntc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create NTC-based features from day-ahead transfer capacity data."""
        df = df.copy()

        ntc_features = []

        # --- Pass-through (already in unified dataset) ---
        ntc_passthrough = [
            'ntc_nl_total_export_mw',
            'ntc_nl_total_import_mw',
            'ntc_nl_net_mw',
        ]
        for col in ntc_passthrough:
            if col in df.columns:
                ntc_features.append(col)

        # --- Derived features ---

        # Total interconnector capacity (both directions)
        if 'ntc_nl_total_export_mw' in df.columns and 'ntc_nl_total_import_mw' in df.columns:
            df['ntc_nl_total_mw'] = (
                df['ntc_nl_total_export_mw'] + df['ntc_nl_total_import_mw']
            )
            ntc_features.append('ntc_nl_total_mw')

            # Export asymmetry ratio (0.5 = symmetric, >0.5 = more export capacity)
            df['ntc_export_ratio'] = np.where(
                df['ntc_nl_total_mw'] > 0,
                df['ntc_nl_total_export_mw'] / df['ntc_nl_total_mw'],
                0.5,
            )
            ntc_features.append('ntc_export_ratio')

        # RES surplus vs export capacity — can NL export its surplus?
        if 'forecast_res_surplus_mw' in df.columns and 'ntc_nl_total_export_mw' in df.columns:
            df['ntc_res_surplus_vs_export'] = np.where(
                df['ntc_nl_total_export_mw'] > 0,
                df['forecast_res_surplus_mw'] / df['ntc_nl_total_export_mw'],
                0.0,
            )
            ntc_features.append('ntc_res_surplus_vs_export')

            # Binary: RES surplus exceeds total export NTC (congestion risk)
            df['ntc_export_constrained'] = (
                df['forecast_res_surplus_mw'] > df['ntc_nl_total_export_mw']
            ).astype(int)
            ntc_features.append('ntc_export_constrained')

        # Individual cable availability (BritNed and NorNed)
        if 'ntc_da_nl_gb_mw' in df.columns:
            df['ntc_gb_available'] = df['ntc_da_nl_gb_mw'].fillna(0)
            ntc_features.append('ntc_gb_available')

        if 'ntc_da_nl_no2_mw' in df.columns:
            df['ntc_no2_available'] = df['ntc_da_nl_no2_mw'].fillna(0)
            ntc_features.append('ntc_no2_available')

        # NTC change vs D-2 daily mean (capacity shift signal)
        if 'ntc_nl_total_export_mw' in df.columns:
            d2_periods = 48 * self.pph  # 192 QH = 48 hours
            df['ntc_change_vs_d2'] = (
                df['ntc_nl_total_export_mw']
                - df['ntc_nl_total_export_mw'].shift(d2_periods)
            )
            ntc_features.append('ntc_change_vs_d2')

        self._ntc_features = ntc_features
        return df

    # ------------------------------------------------------------------
    # Override transform pipeline
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full v6 feature engineering pipeline (v5 + NTC)."""
        # Run all v5 feature engineering first
        df = super().transform(df)

        # Add NTC features
        df = self.create_ntc_features(df)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of all v6 feature column names."""
        return super().get_feature_columns() + self._ntc_features
