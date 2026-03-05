"""
BESS Arbitrage Simulation Configuration

Extends settings_v10.py with parameters for:
- 50 MW onshore wind farm (scaled from national wind)
- 25 MW / 50 MWh co-located battery storage
- Day-ahead trading strategy using quantile forecasts
"""

import math
from config.settings_v10 import (
    MODELS_DIR,
    DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    QUANTILES_V10,
    TEST_START_DATE,
    TEST_END_DATE,
)

# =============================================================================
# WIND FARM PARAMETERS
# =============================================================================

WIND_FARM_CAPACITY_MW = 50.0
"""Nameplate capacity of the wind farm (MW)"""

NL_INSTALLED_WIND_CAPACITY_MW = 11700.0
"""Total installed wind capacity in Netherlands end-2024 (MW)"""

WIND_FARM_SHARE = WIND_FARM_CAPACITY_MW / NL_INSTALLED_WIND_CAPACITY_MW
"""Farm's share of national wind generation (~0.43%)"""

# =============================================================================
# BATTERY STORAGE PARAMETERS
# =============================================================================

BESS_POWER_MW = 25.0
"""Maximum charge/discharge power (MW)"""

BESS_ENERGY_MWH = 50.0
"""Nameplate energy capacity (MWh) - 2-hour duration"""

BESS_SOC_MIN_PCT = 0.10
"""Minimum state of charge (10% floor for battery health)"""

BESS_SOC_MAX_PCT = 0.90
"""Maximum state of charge (90% ceiling for battery health)"""

BESS_SOC_MIN_MWH = BESS_ENERGY_MWH * BESS_SOC_MIN_PCT
"""Minimum usable SoC in MWh (5 MWh)"""

BESS_SOC_MAX_MWH = BESS_ENERGY_MWH * BESS_SOC_MAX_PCT
"""Maximum usable SoC in MWh (45 MWh)"""

BESS_USABLE_CAPACITY_MWH = BESS_SOC_MAX_MWH - BESS_SOC_MIN_MWH
"""Usable capacity between floor and ceiling (40 MWh)"""

BESS_INITIAL_SOC_MWH = 25.0
"""Initial state of charge at simulation start (50%)"""

BESS_RTE = 0.85
"""Round-trip efficiency (85% - conservative for Li-ion)"""

BESS_CHARGE_EFF = math.sqrt(BESS_RTE)
"""One-way charge efficiency (~92.2%)"""

BESS_DISCHARGE_EFF = math.sqrt(BESS_RTE)
"""One-way discharge efficiency (~92.2%)"""

BESS_MAX_DAILY_CYCLES = 1.5
"""Maximum daily cycles for warranty compliance"""

# =============================================================================
# MARKET PARAMETERS
# =============================================================================

MTU_DURATION_HOURS = 0.25
"""Market time unit duration (15 minutes = 0.25 hours)"""

MTUS_PER_DAY = 96
"""Number of 15-minute periods per day"""

EPEX_PRICE_FLOOR = -500.0
"""EPEX day-ahead price floor (EUR/MWh)"""

EPEX_PRICE_CAP = 4000.0
"""EPEX day-ahead price cap (EUR/MWh)"""

# =============================================================================
# BIDDING STRATEGY PARAMETERS
# =============================================================================

CHARGE_BID_QUANTILE = 0.25
"""Quantile for charge bid price (buy if clearing price <= q25 forecast)"""

DISCHARGE_ASK_QUANTILE = 0.75
"""Quantile for discharge ask price (sell if clearing price >= q75 forecast)"""

# Aggressive strategy variant
CHARGE_BID_QUANTILE_AGGRESSIVE = 0.40
"""Tighter threshold for aggressive charging"""

DISCHARGE_ASK_QUANTILE_AGGRESSIVE = 0.60
"""Tighter threshold for aggressive discharging"""

# Threshold for aggressive negative price charging
HIGH_RES_PENETRATION_THRESHOLD = 0.80
"""When RES penetration > 80%, charge at price floor"""

# =============================================================================
# QUANTILE INDICES (for easy lookup)
# =============================================================================

# QUANTILES_V10 = [0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95]
Q05_IDX = 0
Q10_IDX = 1
Q25_IDX = 2
Q40_IDX = 3
Q50_IDX = 4  # Median
Q60_IDX = 5
Q75_IDX = 6
Q90_IDX = 7
Q95_IDX = 8

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

BESS_FIGURES_DIR = FIGURES_DIR / "bess"
"""Directory for BESS-specific visualizations"""

BESS_REPORTS_DIR = REPORTS_DIR
"""Directory for BESS reports (same as main reports)"""
