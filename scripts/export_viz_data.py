"""
Export BESS Simulation Data for Visualization

Exports MTU-level (quarter-hourly) simulation data to JSON for the React
visualization app. Creates both a full export and a 7-day sample.

Usage:
    python scripts/export_viz_data.py [--sample-only]
"""

import sys
from pathlib import Path
import logging
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

# Try to import joblib (may be standalone or from sklearn)
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings_bess import (
    WIND_FARM_CAPACITY_MW,
    NL_INSTALLED_WIND_CAPACITY_MW,
    WIND_FARM_SHARE,
    BESS_POWER_MW,
    BESS_ENERGY_MWH,
    BESS_SOC_MIN_MWH,
    BESS_SOC_MAX_MWH,
    BESS_INITIAL_SOC_MWH,
    CHARGE_BID_QUANTILE,
    DISCHARGE_ASK_QUANTILE,
)
from config.settings_v10 import (
    MODELS_DIR,
    DATA_DIR,
    QUANTILES_V10,
    TEST_START_DATE,
    TEST_END_DATE,
)

from src.simulation.portfolio_backtester import PortfolioBacktester, SimulationConfig
from src.simulation.market import DailyOutcome

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Output paths
VIZ_DATA_DIR = PROJECT_ROOT / "viz" / "src" / "data"

# Sample week: May 12-18, 2025 (most negative price hours)
SAMPLE_START = pd.Timestamp('2025-05-12', tz='UTC')
SAMPLE_END = pd.Timestamp('2025-05-18 23:45:00', tz='UTC')


def load_model_and_data():
    """Load v10 quantile model and feature matrix."""
    logger.info("Loading model and data...")

    # Load model
    model_path = MODELS_DIR / "quantile_regressor_v10.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train_quantile_model_v10.py first."
        )
    model = joblib.load(model_path)
    logger.info(f"  Loaded model: {model_path.name}")

    # Load feature matrix
    feature_matrix_path = DATA_DIR / "processed" / "feature_matrix_v7.csv"
    if not feature_matrix_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {feature_matrix_path}")

    df = pd.read_csv(feature_matrix_path, parse_dates=['datetime'])
    df = df.set_index('datetime')
    logger.info(f"  Loaded feature matrix: {len(df):,} rows")

    # Filter to test period
    test_mask = (df.index >= TEST_START_DATE) & (df.index <= TEST_END_DATE)
    test_df = df[test_mask].copy()
    logger.info(f"  Test period: {len(test_df):,} rows ({len(test_df)//96} days)")

    # Handle missing values
    if test_df['wind_generation_mw'].isna().sum() > 0:
        logger.warning("  Filling NaN in wind_generation_mw with 0")
        test_df['wind_generation_mw'] = test_df['wind_generation_mw'].fillna(0)

    return model, test_df


def get_feature_columns(test_df: pd.DataFrame) -> list:
    """Get feature column names (exclude target and actuals)."""
    exclude = [
        'price_eur_mwh',
        'is_negative_price',
        'price_is_15min',
        'solar_generation_mw',
        'wind_generation_mw',
        'load_mw',
        'flow_NL_DE_mw',
        'ntc_da_nl_be_mw', 'ntc_da_be_nl_mw',
        'ntc_da_nl_delu_mw', 'ntc_da_delu_nl_mw',
        'ntc_da_nl_gb_mw', 'ntc_da_gb_nl_mw',
        'ntc_da_nl_no2_mw', 'ntc_da_no2_nl_mw',
    ]
    return [c for c in test_df.columns if c not in exclude]


def run_simulation(model, test_df, feature_columns):
    """Run BESS simulation and return outcomes."""
    logger.info("Running BESS simulation...")

    config = SimulationConfig(
        wind_capacity_mw=WIND_FARM_CAPACITY_MW,
        bess_power_mw=BESS_POWER_MW,
        bess_energy_mwh=BESS_ENERGY_MWH,
        charge_quantile=CHARGE_BID_QUANTILE,
        discharge_quantile=DISCHARGE_ASK_QUANTILE,
    )

    backtester = PortfolioBacktester(model, QUANTILES_V10, config)
    outcomes = backtester.run(
        test_df,
        feature_columns=feature_columns,
        price_column='price_eur_mwh',
        wind_column='wind_generation_mw',
        res_penetration_column='forecast_res_penetration',
    )

    return outcomes


def build_export_data(outcomes: list, test_df: pd.DataFrame) -> dict:
    """Build JSON export structure from outcomes and test data."""
    logger.info("Building export data structure...")

    # Metadata
    metadata = {
        "wind_capacity_mw": WIND_FARM_CAPACITY_MW,
        "bess_power_mw": BESS_POWER_MW,
        "bess_energy_mwh": BESS_ENERGY_MWH,
        "bess_soc_min_mwh": BESS_SOC_MIN_MWH,
        "bess_soc_max_mwh": BESS_SOC_MAX_MWH,
        "test_start": str(TEST_START_DATE.date()),
        "test_end": str(TEST_END_DATE.date()),
        "strategy": "Conservative (q25/q75)",
        "generated_at": datetime.now().isoformat(),
    }

    # Group test_df by date
    test_df = test_df.copy()
    test_df['_date'] = test_df.index.date

    # Build days array
    days = []
    daily_groups = list(test_df.groupby('_date'))

    for outcome, (date, day_df) in zip(outcomes, daily_groups):
        if len(day_df) != 96:
            continue

        # Daily summary
        daily_summary = {
            "wind_revenue_eur": round(outcome.wind_revenue_eur, 2),
            "bess_charge_cost_eur": round(outcome.bess_charge_cost_eur, 2),
            "bess_discharge_revenue_eur": round(outcome.bess_discharge_revenue_eur, 2),
            "bess_net_pnl_eur": round(outcome.bess_net_pnl_eur, 2),
            "total_portfolio_revenue_eur": round(outcome.total_portfolio_revenue_eur, 2),
            "energy_charged_mwh": round(outcome.energy_charged_mwh, 2),
            "energy_discharged_mwh": round(outcome.energy_discharged_mwh, 2),
            "cycles": round(outcome.cycles, 3),
            "n_charge_mtus": outcome.n_charge_mtus,
            "n_discharge_mtus": outcome.n_discharge_mtus,
        }

        # MTU-level data
        mtus = []
        for t in range(96):
            row = day_df.iloc[t]

            # Weather columns with safe fallbacks
            wind_speed = row.get('forecast_wind_speed_100m_ms', 0)
            cloud_cover = row.get('forecast_cloud_cover_pct', 0)
            temperature = row.get('forecast_temperature_2m_c', 0)

            # Handle NaN values
            wind_speed = 0.0 if pd.isna(wind_speed) else float(wind_speed)
            cloud_cover = 0.0 if pd.isna(cloud_cover) else float(cloud_cover)
            temperature = 0.0 if pd.isna(temperature) else float(temperature)

            mtu = {
                "t": t,
                "hour": t // 4,
                "quarter": t % 4,
                "price_eur_mwh": round(float(row['price_eur_mwh']), 2),
                "wind_generation_mw": round(float(row['wind_generation_mw']) * WIND_FARM_SHARE, 2),
                "wind_speed_ms": round(wind_speed, 1),
                "cloud_cover_pct": round(cloud_cover, 0),
                "temperature_c": round(temperature, 1),
                "soc_mwh": round(float(outcome.soc_timeseries[t + 1]), 2),
                "charge_mw": round(float(outcome.actual_charge_mw[t]), 2),
                "discharge_mw": round(float(outcome.actual_discharge_mw[t]), 2),
                "charge_cleared": bool(outcome.charge_cleared[t]),
                "discharge_cleared": bool(outcome.discharge_cleared[t]),
            }
            mtus.append(mtu)

        days.append({
            "date": str(date),
            "daily_summary": daily_summary,
            "mtus": mtus,
        })

    return {
        "metadata": metadata,
        "days": days,
    }


def filter_sample_week(export_data: dict) -> dict:
    """Filter export data to just the sample week."""
    sample_start_str = str(SAMPLE_START.date())
    sample_end_str = str(SAMPLE_END.date())

    sample_days = [
        d for d in export_data["days"]
        if sample_start_str <= d["date"] <= sample_end_str
    ]

    sample_metadata = export_data["metadata"].copy()
    sample_metadata["test_start"] = sample_start_str
    sample_metadata["test_end"] = sample_end_str
    sample_metadata["is_sample"] = True

    return {
        "metadata": sample_metadata,
        "days": sample_days,
    }


def save_json(data: dict, output_path: Path):
    """Save data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    # File size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved: {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Export BESS simulation data for visualization')
    parser.add_argument('--sample-only', action='store_true', help='Only export 7-day sample')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BESS VISUALIZATION DATA EXPORT")
    logger.info("=" * 80)

    # Load model and data
    model, test_df = load_model_and_data()
    feature_columns = get_feature_columns(test_df)

    # Run simulation
    outcomes = run_simulation(model, test_df, feature_columns)

    # Build export data
    export_data = build_export_data(outcomes, test_df)
    logger.info(f"  Built export: {len(export_data['days'])} days, ~{len(export_data['days']) * 96} MTUs")

    # Save sample
    sample_data = filter_sample_week(export_data)
    sample_path = VIZ_DATA_DIR / "simulation_output_sample.json"
    save_json(sample_data, sample_path)
    logger.info(f"  Sample week: {SAMPLE_START.date()} to {SAMPLE_END.date()}")

    # Save full export (unless sample-only)
    if not args.sample_only:
        full_path = VIZ_DATA_DIR / "simulation_output.json"
        save_json(export_data, full_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
