# NegaPriceNL - Dutch Negative Price Predictor

A machine learning system that predicts when Dutch day-ahead electricity prices will go negative, quantifies the economic value of these predictions, and provides compelling visualizations for portfolio presentation.

## Project Overview

The Netherlands recorded 700+ negative price hours in 2025, making it the 2nd highest in Europe. This project builds a predictive model to help renewable energy producers optimize their operations and avoid losses during negative price events.

## Quick Start

### 1. Prerequisites

- Python 3.9+
- ENTSO-E Transparency Platform API key (free registration at [transparency.entsoe.eu](https://transparency.entsoe.eu/))

### 2. Installation

```bash
# Create and activate virtual environment
python -m venv quant
source quant/bin/activate  # On Windows: quant\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy the example environment file and add your API key:

```bash
cp .env.example .env
# Edit .env and add your ENTSOE_API_KEY
```

### 4. Download Data

Download all required ENTSO-E data (2019-2025):

```bash
python scripts/download_data.py
```

This will download:
- Dutch, German, and Belgian day-ahead prices
- Dutch solar and wind generation (actual + forecasts)
- Dutch load data (actual + forecasts)
- Cross-border flows (NL-DE, NL-BE)

**Note:** The full download may take 2-3 hours due to API rate limits (400 requests/minute). The script automatically handles rate limiting and saves progress.

#### Download Options

Download specific data types:

```bash
# Download only price data
python scripts/download_data.py --skip-generation --skip-forecast --skip-load --skip-flows

# Download data for a specific date range
python scripts/download_data.py --start 2024-01-01 --end 2024-12-31

# Download everything except cross-border flows
python scripts/download_data.py --skip-flows
```

### 5. Monitor Progress

The script logs progress to both console and `data_download.log`. You can monitor in real-time:

```bash
tail -f data_download.log
```

## Project Structure

```
negaprice-nl/
├── config/                 # Configuration files
├── data/                   # Data directory (git-ignored)
│   ├── raw/entsoe/         # Raw ENTSO-E data downloads
│   ├── processed/          # Cleaned and merged data
│   └── outputs/            # Model predictions
├── src/                    # Source code
│   ├── data/collectors/    # Data collection modules
│   ├── features/           # Feature engineering
│   └── models/             # ML models
├── scripts/                # Standalone scripts
│   └── download_data.py    # Main data download script
├── notebooks/              # Jupyter notebooks for analysis
├── outputs/                # Final outputs (figures, models, reports)
└── requirements.txt        # Python dependencies
```

## Data Collection Details

### ENTSO-E API Rate Limits

- **Limit:** 400 requests per minute
- **Implementation:** The collector automatically throttles requests with ~0.15 second delays
- **Chunking:** Large date ranges are split into 30-day chunks to prevent timeouts
- **Error Handling:** Failed requests are logged and skipped, not retrying indefinitely

### Expected Data Volumes

For 2019-2025 (7 years):
- **Hourly records:** ~61,000 per dataset
- **Total CSV files:** ~20 files
- **Approximate size:** 200-300 MB total

### Handling Issues

If the download is interrupted:
1. Check `data_download.log` for the last successful download
2. Re-run the script with `--start` set to the date where it failed
3. The script will resume from that point

Common issues:
- **"API key not found"** - Ensure `.env` file exists with `ENTSOE_API_KEY=your_key`
- **Timeout errors** - Some data types may have gaps; this is normal for ENTSO-E
- **Empty DataFrames** - Some data may not be available for all time periods

## API Key Registration

To obtain your free ENTSO-E API key:

1. Go to [transparency.entsoe.eu](https://transparency.entsoe.eu/)
2. Register for an account
3. Request an API key (usually instant approval)
4. Add to your `.env` file

## Contact

For questions or feedback, feel free to reach out or open an issue.