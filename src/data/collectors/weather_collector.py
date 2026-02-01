"""
Open-Meteo Historical Weather Data Collector

Collects historical weather data for the Netherlands from Open-Meteo Archive API.
Focuses on variables relevant for solar and wind generation forecasting.

API Documentation: https://open-meteo.com/en/docs/historical-weather-api
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenMeteoWeatherCollector:
    """
    Collects historical weather data from Open-Meteo Archive API.

    Focuses on variables relevant for renewable energy forecasting:
    - Solar radiation (GHI, direct, diffuse)
    - Wind speed and direction
    - Temperature (affects demand)
    - Cloud cover (affects solar)
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    # Netherlands locations for spatial coverage
    DUTCH_LOCATIONS = [
        {'name': 'Amsterdam', 'lat': 52.3676, 'lon': 4.9041},  # Central/West
        {'name': 'Rotterdam', 'lat': 51.9244, 'lon': 4.4777},  # Southwest
        {'name': 'Groningen', 'lat': 53.2194, 'lon': 6.5665},  # Northeast
        {'name': 'Maastricht', 'lat': 50.8514, 'lon': 5.6909}, # Southeast
        {'name': 'Den Helder', 'lat': 52.9597, 'lon': 4.7580}, # North coast (offshore wind)
    ]

    def __init__(self):
        """Initialize the weather data collector."""
        logger.info("OpenMeteoWeatherCollector initialized")

    def get_historical_weather(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Fetch historical weather data from Open-Meteo API.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of weather variables to fetch

        Returns:
            JSON response from API or None if failed
        """
        if variables is None:
            # Variables relevant for renewable energy forecasting
            variables = [
                # Solar radiation
                "shortwave_radiation",        # Global Horizontal Irradiance (GHI)
                "direct_radiation",           # Direct Normal Irradiance
                "diffuse_radiation",          # Diffuse radiation

                # Wind
                "wind_speed_10m",             # Wind speed at 10m
                "wind_speed_100m",            # Wind speed at 100m (hub height)
                "wind_direction_10m",         # Wind direction at 10m
                "wind_direction_100m",        # Wind direction at 100m
                "wind_gusts_10m",             # Wind gusts

                # Temperature (demand proxy)
                "temperature_2m",             # Temperature at 2m

                # Cloud cover (solar proxy)
                "cloud_cover",                # Total cloud cover

                # Pressure and humidity
                "surface_pressure",           # Surface pressure
                "relative_humidity_2m",       # Relative humidity

                # Precipitation
                "precipitation",              # Precipitation
            ]

        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join(variables),
            'timezone': 'Europe/Amsterdam'  # Match ENTSO-E timezone
        }

        try:
            logger.info(f"Fetching weather data for ({lat}, {lon}) from {start_date} to {end_date}")
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return None

    def process_weather_data(
        self,
        weather_response: Dict,
        location_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Process Open-Meteo API response into structured DataFrame.

        Args:
            weather_response: JSON response from API
            location_name: Name of the location

        Returns:
            DataFrame with processed weather data
        """
        if not weather_response or 'hourly' not in weather_response:
            logger.warning(f"No valid data in response for {location_name}")
            return None

        hourly_data = weather_response['hourly']
        times = hourly_data['time']

        # Create DataFrame
        data = {
            'location': location_name,
            'datetime': times,
            'latitude': weather_response.get('latitude'),
            'longitude': weather_response.get('longitude'),
        }

        # Add all weather variables
        variable_mapping = {
            'shortwave_radiation': 'ghi_wm2',
            'direct_radiation': 'dni_wm2',
            'diffuse_radiation': 'diffuse_radiation_wm2',
            'wind_speed_10m': 'wind_speed_10m_ms',
            'wind_speed_100m': 'wind_speed_100m_ms',
            'wind_direction_10m': 'wind_direction_10m_deg',
            'wind_direction_100m': 'wind_direction_100m_deg',
            'wind_gusts_10m': 'wind_gusts_10m_ms',
            'temperature_2m': 'temperature_2m_c',
            'cloud_cover': 'cloud_cover_pct',
            'surface_pressure': 'pressure_hpa',
            'relative_humidity_2m': 'humidity_pct',
            'precipitation': 'precipitation_mm',
        }

        for api_name, column_name in variable_mapping.items():
            if api_name in hourly_data:
                data[column_name] = hourly_data[api_name]

        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])

        logger.info(f"Processed {len(df)} records for {location_name}")
        return df

    def collect_netherlands_weather(
        self,
        start_date: str,
        end_date: str,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Collect historical weather data for multiple Dutch locations.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save output files

        Returns:
            Combined DataFrame with all weather data
        """
        logger.info("="*60)
        logger.info("Starting Netherlands Weather Data Collection")
        logger.info("="*60)
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Locations: {len(self.DUTCH_LOCATIONS)}")
        logger.info(f"Output directory: {output_dir}")

        all_weather_data = []

        for i, location in enumerate(self.DUTCH_LOCATIONS):
            logger.info(f"\nProcessing {location['name']} ({i+1}/{len(self.DUTCH_LOCATIONS)})")

            # Get weather data for this location
            weather_response = self.get_historical_weather(
                location['lat'],
                location['lon'],
                start_date,
                end_date
            )

            if weather_response:
                df = self.process_weather_data(weather_response, location['name'])

                if df is not None:
                    all_weather_data.append(df)

                    # Save individual location file
                    location_file = output_dir / f"nl_weather_{location['name'].lower()}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
                    df.to_csv(location_file, index=False)
                    logger.info(f"✓ Saved {len(df)} records to {location_file.name}")

            # Be respectful to the API - small delay between requests
            if i < len(self.DUTCH_LOCATIONS) - 1:
                time.sleep(1)

        # Combine all locations
        if all_weather_data:
            combined_df = pd.concat(all_weather_data, ignore_index=True)

            # Save combined file
            combined_file = output_dir / f"nl_weather_all_locations_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
            combined_df.to_csv(combined_file, index=False)

            logger.info("\n" + "="*60)
            logger.info("Weather Data Collection Complete!")
            logger.info("="*60)
            logger.info(f"✓ Total records: {len(combined_df):,}")
            logger.info(f"✓ Locations: {combined_df['location'].nunique()}")
            logger.info(f"✓ Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
            logger.info(f"✓ Saved to: {combined_file}")
            logger.info("="*60)

            return combined_df
        else:
            logger.error("No weather data collected!")
            return pd.DataFrame()


def main():
    """Main execution function for weather data collection."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from config.settings import DATA_START_DATE, DATA_END_DATE, WEATHER_DATA_DIR

    # Initialize collector
    collector = OpenMeteoWeatherCollector()

    # Collect data
    start_date = DATA_START_DATE.strftime('%Y-%m-%d')
    end_date = DATA_END_DATE.strftime('%Y-%m-%d')

    df = collector.collect_netherlands_weather(
        start_date=start_date,
        end_date=end_date,
        output_dir=WEATHER_DATA_DIR
    )

    if not df.empty:
        logger.info("\n📊 Dataset Summary:")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info(f"   Shape: {df.shape}")
        logger.info("\n🔍 Sample data:")
        print(df.head())
    else:
        logger.error("Failed to collect weather data")
        sys.exit(1)


if __name__ == "__main__":
    main()
