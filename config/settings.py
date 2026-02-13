"""
Global settings and constants for the NegaPriceNL project.
"""
from pathlib import Path
from datetime import datetime

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# ENTSO-E specific directories
ENTSOE_DATA_DIR = RAW_DATA_DIR / "entsoe"
WEATHER_DATA_DIR = RAW_DATA_DIR / "weather"
EXTERNAL_DATA_DIR = RAW_DATA_DIR / "external"
WEATHER_FORECAST_DATA_DIR = RAW_DATA_DIR / "weather_forecast"

# Output directories
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# Ensure directories exist
for dir_path in [ENTSOE_DATA_DIR, WEATHER_DATA_DIR, WEATHER_FORECAST_DATA_DIR,
                 EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR,
                 FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data collection settings
COUNTRY_CODE_NL = "NL"
COUNTRY_CODE_DE = "DE"
COUNTRY_CODE_BE = "BE"

# Historical data range
DATA_START_DATE = datetime(2019, 1, 1)
DATA_END_DATE = datetime(2025, 12, 31)

# ENTSO-E API rate limits
ENTSOE_RATE_LIMIT_PER_MINUTE = 400
ENTSOE_REQUESTS_PER_BATCH = 50  # Conservative batch size
ENTSOE_DELAY_BETWEEN_BATCHES = 10  # seconds

# Timezone
TIMEZONE = "Europe/Amsterdam"

# Installed capacity (MW) - approximate values for 2025
INSTALLED_SOLAR_CAPACITY_MW = 24000
INSTALLED_WIND_CAPACITY_MW = 12000

# v5 pipeline settings (Phase 1 — auction-aligned)
PERIODS_PER_HOUR = 4
QH_FREQ = '15min'
QH_PRICE_START_DATE = datetime(2025, 10, 1)  # NL 15-min MTU start
WEATHER_FORECAST_AVAILABLE_FROM = datetime(2022, 1, 1)

# Model settings
RANDOM_STATE = 42
TEST_START_DATE = datetime(2025, 1, 1)
VALIDATION_START_DATE = datetime(2024, 1, 1)
TRAIN_END_DATE = datetime(2023, 12, 31)
