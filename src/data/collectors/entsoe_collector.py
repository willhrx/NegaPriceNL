"""
ENTSO-E Transparency Platform Data Collector

Collects electricity market data for the Netherlands from the ENTSO-E
Transparency Platform API using the entsoe-py library.

Rate Limits: 400 requests/minute
API Documentation: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntsoeDataCollector:
    """
    Collects data from ENTSO-E Transparency Platform for the Netherlands.

    Implements rate limiting and error handling for reliable data collection
    over extended time periods.

    Attributes:
        api_key (str): ENTSO-E API key
        client (EntsoePandasClient): ENTSO-E API client
        country_code (str): Country code (default: 'NL' for Netherlands)
        requests_made (int): Counter for rate limiting
        last_request_time (float): Timestamp of last request
    """

    COUNTRY_CODE_NL = 'NL'
    COUNTRY_CODE_DE = 'DE'
    COUNTRY_CODE_BE = 'BE'

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 400
    REQUEST_DELAY = 60 / MAX_REQUESTS_PER_MINUTE  # ~0.15 seconds

    # Chunk size for large date ranges (days)
    CHUNK_SIZE_DAYS = 30

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ENTSO-E data collector.

        Args:
            api_key: ENTSO-E API key. If None, reads from ENTSOE_API_KEY env variable.

        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ENTSOE_API_KEY not found. Please set it in .env file or pass as argument."
            )

        self.client = EntsoePandasClient(api_key=self.api_key)
        self.requests_made = 0
        self.last_request_time = time.time()

        logger.info("EntsoeDataCollector initialized successfully")

    def _rate_limit(self):
        """
        Implement rate limiting to stay within API limits.
        Ensures we don't exceed 400 requests per minute.
        """
        self.requests_made += 1

        # Add delay between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)

        self.last_request_time = time.time()

        # Log progress every 50 requests
        if self.requests_made % 50 == 0:
            logger.info(f"Made {self.requests_made} API requests")

    def _chunk_date_range(
        self,
        start: datetime,
        end: datetime,
        chunk_days: int = CHUNK_SIZE_DAYS
    ) -> list[Tuple[datetime, datetime]]:
        """
        Split a date range into smaller chunks to handle large requests.

        Args:
            start: Start date
            end: End date
            chunk_days: Number of days per chunk

        Returns:
            List of (start, end) datetime tuples
        """
        chunks = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + timedelta(days=chunk_days), end)
            chunks.append((current_start, current_end))
            current_start = current_end

        logger.info(f"Split date range into {len(chunks)} chunks of ~{chunk_days} days")
        return chunks

    def _safe_query(self, query_func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        Execute an ENTSO-E query with error handling and rate limiting.

        Args:
            query_func: The ENTSO-E client query function to call
            *args: Positional arguments for the query function
            **kwargs: Keyword arguments for the query function

        Returns:
            DataFrame with query results, or None if query fails
        """
        self._rate_limit()

        try:
            result = query_func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            logger.info("Waiting 5 seconds before continuing...")
            time.sleep(5)
            return None

    def get_day_ahead_prices(
        self,
        start: datetime,
        end: datetime,
        country_code: str = COUNTRY_CODE_NL
    ) -> pd.DataFrame:
        """
        Fetch day-ahead electricity prices.

        Args:
            start: Start datetime
            end: End datetime
            country_code: Country code (default: NL)

        Returns:
            DataFrame with datetime index and 'price' column (EUR/MWh)
        """
        logger.info(f"Fetching day-ahead prices for {country_code} from {start} to {end}")

        chunks = self._chunk_date_range(start, end)
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_day_ahead_prices,
                country_code,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam')
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]  # Remove duplicates
            result = result.to_frame(name='price')
            logger.info(f"✓ Collected {len(result)} price records")
            return result
        else:
            logger.warning("No price data collected")
            return pd.DataFrame()

    def get_generation_actual(
        self,
        start: datetime,
        end: datetime,
        country_code: str = COUNTRY_CODE_NL
    ) -> pd.DataFrame:
        """
        Fetch actual generation by fuel type.

        Args:
            start: Start datetime
            end: End datetime
            country_code: Country code (default: NL)

        Returns:
            DataFrame with datetime index and generation columns (MW)
        """
        logger.info(f"Fetching actual generation for {country_code} from {start} to {end}")

        chunks = self._chunk_date_range(start, end)
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_generation,
                country_code,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam')
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]
            logger.info(f"✓ Collected {len(result)} generation records")
            return result
        else:
            logger.warning("No generation data collected")
            return pd.DataFrame()

    def get_generation_forecast(
        self,
        start: datetime,
        end: datetime,
        country_code: str = COUNTRY_CODE_NL
    ) -> pd.DataFrame:
        """
        Fetch day-ahead generation forecast.

        Args:
            start: Start datetime
            end: End datetime
            country_code: Country code (default: NL)

        Returns:
            DataFrame with datetime index and generation forecast columns (MW)
        """
        logger.info(f"Fetching generation forecast for {country_code} from {start} to {end}")

        chunks = self._chunk_date_range(start, end)
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_generation_forecast,
                country_code,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam')
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]
            logger.info(f"✓ Collected {len(result)} forecast records")
            return result
        else:
            logger.warning("No forecast data collected")
            return pd.DataFrame()

    def get_wind_and_solar_forecast(
        self,
        start: datetime,
        end: datetime,
        country_code: str = COUNTRY_CODE_NL
    ) -> pd.DataFrame:
        """
        Fetch day-ahead generation forecasts for wind and solar.

        Uses ENTSO-E document type A69 which provides separate forecasts
        for solar, wind onshore, and wind offshore generation.

        Args:
            start: Start datetime
            end: End datetime
            country_code: Country code (default: NL)

        Returns:
            DataFrame with datetime index and columns for solar, wind onshore,
            and wind offshore generation forecasts (MW)
        """
        logger.info(f"Fetching wind & solar forecast for {country_code} from {start} to {end}")

        chunks = self._chunk_date_range(start, end)
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_wind_and_solar_forecast,
                country_code,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam')
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]
            logger.info(f"✓ Collected {len(result)} wind/solar forecast records")
            logger.info(f"  Columns: {list(result.columns)}")
            return result
        else:
            logger.warning("No wind/solar forecast data collected")
            return pd.DataFrame()

    def get_load_actual(
        self,
        start: datetime,
        end: datetime,
        country_code: str = COUNTRY_CODE_NL
    ) -> pd.DataFrame:
        """
        Fetch actual total load.

        Args:
            start: Start datetime
            end: End datetime
            country_code: Country code (default: NL)

        Returns:
            DataFrame with datetime index and 'load' column (MW)
        """
        logger.info(f"Fetching actual load for {country_code} from {start} to {end}")

        chunks = self._chunk_date_range(start, end)
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_load,
                country_code,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam')
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]

            # Convert to DataFrame if it's a Series
            if isinstance(result, pd.Series):
                result = result.to_frame(name='load_mw')
            elif isinstance(result, pd.DataFrame) and 'load_mw' not in result.columns:
                # Rename first column to load_mw if it exists
                if len(result.columns) > 0:
                    result = result.rename(columns={result.columns[0]: 'load_mw'})

            logger.info(f"✓ Collected {len(result)} load records")
            return result
        else:
            logger.warning("No load data collected")
            return pd.DataFrame()

    def get_load_forecast(
        self,
        start: datetime,
        end: datetime,
        country_code: str = COUNTRY_CODE_NL
    ) -> pd.DataFrame:
        """
        Fetch day-ahead load forecast.

        Args:
            start: Start datetime
            end: End datetime
            country_code: Country code (default: NL)

        Returns:
            DataFrame with datetime index and 'load_forecast' column (MW)
        """
        logger.info(f"Fetching load forecast for {country_code} from {start} to {end}")

        chunks = self._chunk_date_range(start, end)
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_load_forecast,
                country_code,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam')
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]

            # Convert to DataFrame if it's a Series
            if isinstance(result, pd.Series):
                result = result.to_frame(name='load_forecast_mw')
            elif isinstance(result, pd.DataFrame) and 'load_forecast_mw' not in result.columns:
                # Rename first column to load_forecast_mw if it exists
                if len(result.columns) > 0:
                    result = result.rename(columns={result.columns[0]: 'load_forecast_mw'})

            logger.info(f"✓ Collected {len(result)} load forecast records")
            return result
        else:
            logger.warning("No load forecast data collected")
            return pd.DataFrame()

    def get_cross_border_flows(
        self,
        start: datetime,
        end: datetime,
        from_country: str = COUNTRY_CODE_NL,
        to_country: str = COUNTRY_CODE_DE
    ) -> pd.DataFrame:
        """
        Fetch cross-border physical flows between countries.

        Args:
            start: Start datetime
            end: End datetime
            from_country: Origin country code
            to_country: Destination country code

        Returns:
            DataFrame with datetime index and 'flow' column (MW, positive = export)
        """
        logger.info(f"Fetching cross-border flows {from_country} -> {to_country} from {start} to {end}")

        chunks = self._chunk_date_range(start, end, chunk_days=90)  # Larger chunks for flows
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_crossborder_flows,
                from_country,
                to_country,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam')
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]

            # Convert to DataFrame if it's a Series
            column_name = f'flow_{from_country}_{to_country}_mw'
            if isinstance(result, pd.Series):
                result = result.to_frame(name=column_name)
            elif isinstance(result, pd.DataFrame) and column_name not in result.columns:
                # Rename first column if it exists
                if len(result.columns) > 0:
                    result = result.rename(columns={result.columns[0]: column_name})

            logger.info(f"✓ Collected {len(result)} flow records")
            return result
        else:
            logger.warning(f"No flow data collected for {from_country} -> {to_country}")
            return pd.DataFrame()

    def get_net_transfer_capacity_dayahead(
        self,
        start: datetime,
        end: datetime,
        from_country: str = COUNTRY_CODE_NL,
        to_country: str = COUNTRY_CODE_DE,
    ) -> pd.DataFrame:
        """
        Fetch day-ahead net transfer capacity (NTC) between two bidding zones.

        Args:
            start: Start datetime
            end: End datetime
            from_country: Origin bidding zone code (e.g. 'NL', 'DE_LU', 'GB', 'NO_2')
            to_country: Destination bidding zone code

        Returns:
            DataFrame with datetime index and NTC column (MW)
        """
        logger.info(f"Fetching day-ahead NTC {from_country} -> {to_country} from {start} to {end}")

        chunks = self._chunk_date_range(start, end, chunk_days=90)
        all_data = []

        for chunk_start, chunk_end in chunks:
            logger.info(f"  Fetching chunk: {chunk_start.date()} to {chunk_end.date()}")

            data = self._safe_query(
                self.client.query_net_transfer_capacity_dayahead,
                from_country,
                to_country,
                start=pd.Timestamp(chunk_start, tz='Europe/Amsterdam'),
                end=pd.Timestamp(chunk_end, tz='Europe/Amsterdam'),
            )

            if data is not None:
                all_data.append(data)

        if all_data:
            result = pd.concat(all_data).sort_index()
            result = result[~result.index.duplicated(keep='first')]

            # Normalise country codes for column naming (DE_LU -> de_lu, NO_2 -> no2)
            from_label = from_country.lower().replace('_', '')
            to_label = to_country.lower().replace('_', '')
            column_name = f'ntc_da_{from_label}_{to_label}_mw'

            if isinstance(result, pd.Series):
                result = result.to_frame(name=column_name)
            elif isinstance(result, pd.DataFrame) and column_name not in result.columns:
                if len(result.columns) > 0:
                    result = result.rename(columns={result.columns[0]: column_name})

            logger.info(f"  Collected {len(result)} NTC records")
            return result
        else:
            logger.warning(f"No NTC data collected for {from_country} -> {to_country}")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filepath: Path):
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            filepath: Path to output CSV file
        """
        if df.empty:
            logger.warning(f"DataFrame is empty, not saving to {filepath}")
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)
        logger.info(f"✓ Saved {len(df)} records to {filepath}")


def main():
    """
    Quick test of the collector functionality.
    """
    # Test with a small date range
    collector = EntsoeDataCollector()

    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 7)

    print("\n=== Testing ENTSO-E Data Collector ===\n")

    # Test day-ahead prices
    prices = collector.get_day_ahead_prices(start, end)
    print(f"\nDay-ahead prices:\n{prices.head()}")

    # Test actual generation
    generation = collector.get_generation_actual(start, end)
    print(f"\nActual generation:\n{generation.head()}")

    # Test load
    load = collector.get_load_actual(start, end)
    print(f"\nActual load:\n{load.head()}")

    print(f"\nTotal API requests made: {collector.requests_made}")


if __name__ == "__main__":
    main()
