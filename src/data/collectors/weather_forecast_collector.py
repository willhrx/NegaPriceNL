"""
Open-Meteo Historical Forecast Weather Data Collector

Collects D-1 weather forecast data for the Netherlands.

For 2022+: Uses the Historical Forecast API which stores past NWP model outputs
           (what was forecasted on D-1, not what actually happened).
For 2019-2021: Falls back to the Archive API (observed weather as proxy),
               since no archived forecasts exist before 2022.

API Documentation:
  Historical Forecast: https://open-meteo.com/en/docs/historical-forecast-api
  Archive:            https://open-meteo.com/en/docs/historical-weather-api
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import requests
import pandas as pd

logger = logging.getLogger(__name__)


class OpenMeteoForecastCollector:
    """
    Collects D-1 weather forecast data from Open-Meteo.

    Uses the Historical Forecast API for 2022+ (archived NWP forecasts)
    and falls back to the Archive API for pre-2022 (observed weather
    as a proxy for D-1 forecasts).
    """

    FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Cutoff: Historical Forecast API available from this date
    FORECAST_AVAILABLE_FROM = datetime(2022, 1, 1)

    # Dutch locations (matching existing weather collector)
    DUTCH_LOCATIONS = [
        {'name': 'Amsterdam', 'lat': 52.3676, 'lon': 4.9041},
        {'name': 'Rotterdam', 'lat': 51.9244, 'lon': 4.4777},
        {'name': 'Groningen', 'lat': 53.2194, 'lon': 6.5665},
    ]

    # Variables relevant for renewable energy forecasting
    DEFAULT_VARIABLES = [
        "shortwave_radiation",   # GHI (W/m2)
        "wind_speed_100m",       # Hub height wind (m/s)
        "temperature_2m",        # Temperature (C)
        "cloud_cover",           # Cloud cover (%)
        "surface_pressure",      # Pressure (hPa)
        "relative_humidity_2m",  # Humidity (%)
    ]

    # Map API variable names to clean column names with forecast_ prefix
    VARIABLE_MAPPING = {
        'shortwave_radiation': 'forecast_ghi_wm2',
        'wind_speed_100m': 'forecast_wind_speed_100m_ms',
        'temperature_2m': 'forecast_temperature_2m_c',
        'cloud_cover': 'forecast_cloud_cover_pct',
        'surface_pressure': 'forecast_pressure_hpa',
        'relative_humidity_2m': 'forecast_humidity_pct',
    }

    def __init__(self, request_delay: float = 0.5):
        """
        Initialize the forecast weather collector.

        Args:
            request_delay: Delay between API requests in seconds.
        """
        self.request_delay = request_delay
        logger.info("OpenMeteoForecastCollector initialized")

    def _fetch(self, url: str, params: dict) -> Optional[Dict]:
        """Make an API request with error handling and rate limiting."""
        try:
            time.sleep(self.request_delay)
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def get_weather_data(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Fetch weather data, routing to the correct API based on date.

        For dates >= 2022-01-01: Historical Forecast API
        For dates <  2022-01-01: Archive API (observed proxy)

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: Weather variables to fetch

        Returns:
            Tuple of (json_response, source_label) or None
        """
        if variables is None:
            variables = self.DEFAULT_VARIABLES

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')

        if start_dt >= self.FORECAST_AVAILABLE_FROM:
            url = self.FORECAST_URL
            source = "forecast"
        else:
            url = self.ARCHIVE_URL
            source = "observed_proxy"

        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(variables),
            'timezone': 'Europe/Amsterdam',
        }

        logger.info(f"  Fetching {source} weather: {start_date} to {end_date}")
        data = self._fetch(url, params)
        return (data, source) if data else None

    def process_weather_data(
        self,
        weather_response: Dict,
        location_name: str,
        source: str
    ) -> Optional[pd.DataFrame]:
        """
        Process Open-Meteo API response into a structured DataFrame.

        Args:
            weather_response: JSON response from API
            location_name: Name of the location
            source: "forecast" or "observed_proxy"

        Returns:
            DataFrame with processed weather data
        """
        if not weather_response or 'hourly' not in weather_response:
            logger.warning(f"No valid data in response for {location_name}")
            return None

        hourly_data = weather_response['hourly']

        data = {
            'location': location_name,
            'datetime': hourly_data['time'],
            'weather_source': source,
        }

        for api_name, column_name in self.VARIABLE_MAPPING.items():
            if api_name in hourly_data:
                data[column_name] = hourly_data[api_name]

        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])

        logger.info(f"  Processed {len(df)} records for {location_name} ({source})")
        return df

    def collect_netherlands_forecasts(
        self,
        start_date: str,
        end_date: str,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Collect weather forecast data for Dutch locations.

        Chunks requests by year to stay within API limits.
        Uses Historical Forecast API for 2022+ and Archive API for pre-2022.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save output files

        Returns:
            Combined DataFrame with all weather forecast data
        """
        logger.info("=" * 60)
        logger.info("Starting Netherlands Weather Forecast Collection")
        logger.info("=" * 60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Locations: {len(self.DUTCH_LOCATIONS)}")
        logger.info(f"Output directory: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build year chunks
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        year_chunks = []
        for year in range(start_year, end_year + 1):
            chunk_start = f"{year}-01-01"
            chunk_end = f"{year}-12-31"
            # Clamp to actual date range
            if chunk_start < start_date:
                chunk_start = start_date
            if chunk_end > end_date:
                chunk_end = end_date
            year_chunks.append((chunk_start, chunk_end))

        all_weather_data = []

        for i, location in enumerate(self.DUTCH_LOCATIONS):
            logger.info(f"\nLocation {i+1}/{len(self.DUTCH_LOCATIONS)}: {location['name']}")

            location_dfs = []

            for chunk_start, chunk_end in year_chunks:
                result = self.get_weather_data(
                    location['lat'], location['lon'],
                    chunk_start, chunk_end
                )

                if result:
                    response, source = result
                    df = self.process_weather_data(response, location['name'], source)
                    if df is not None:
                        location_dfs.append(df)

            if location_dfs:
                location_df = pd.concat(location_dfs, ignore_index=True)
                all_weather_data.append(location_df)

                # Save per-location file
                loc_name = location['name'].lower().replace(' ', '_')
                loc_file = output_dir / (
                    f"nl_weather_forecast_{loc_name}_"
                    f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
                )
                location_df.to_csv(loc_file, index=False)
                logger.info(f"  Saved {len(location_df)} records to {loc_file.name}")

            # Delay between locations
            if i < len(self.DUTCH_LOCATIONS) - 1:
                time.sleep(1)

        # Combine all locations
        if all_weather_data:
            combined_df = pd.concat(all_weather_data, ignore_index=True)

            combined_file = output_dir / (
                f"nl_weather_forecast_all_locations_"
                f"{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
            )
            combined_df.to_csv(combined_file, index=False)

            logger.info("\n" + "=" * 60)
            logger.info("Weather Forecast Collection Complete!")
            logger.info("=" * 60)
            logger.info(f"Total records: {len(combined_df):,}")
            logger.info(f"Locations: {combined_df['location'].nunique()}")
            logger.info(f"Date range: {combined_df['datetime'].min()} to "
                         f"{combined_df['datetime'].max()}")

            # Source breakdown
            source_counts = combined_df['weather_source'].value_counts()
            for source, count in source_counts.items():
                logger.info(f"  {source}: {count:,} records "
                             f"({count / len(combined_df) * 100:.1f}%)")

            logger.info(f"Saved to: {combined_file}")
            logger.info("=" * 60)

            return combined_df
        else:
            logger.error("No weather forecast data collected!")
            return pd.DataFrame()
