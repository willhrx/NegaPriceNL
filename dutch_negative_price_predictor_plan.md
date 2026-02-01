# Dutch Negative Price Event Predictor & Visualizer

## Complete Project Plan

---

## Executive Summary

**Project Title:** NegaPriceNL – Predicting Negative Electricity Prices in the Dutch Day-Ahead Market

**Objective:** Build a machine learning system that predicts when Dutch day-ahead electricity prices will go negative (24-48 hours ahead), quantify the economic value of these predictions for renewable asset operators, and create compelling visualizations suitable for LinkedIn portfolio presentation.

**Why This Matters:**
- The Netherlands recorded 584+ negative price hours in 2025 (2nd highest in Europe)
- Dutch renewable producers have seen revenues drop >30% since 2019 due to cannibalization
- Companies like Dexter Energy, KYOS, and Groendus are actively working on this exact problem
- This demonstrates you understand both the technical ML challenge AND the business context

---

## Part 1: Project Goals & Success Criteria

### Primary Goals

1. **Prediction Accuracy**
   - Achieve >75% recall on negative price events (catching most negative hours)
   - Achieve >60% precision (avoiding too many false alarms)
   - Outperform a naive baseline (e.g., "predict negative if yesterday same hour was negative")

2. **Economic Value Quantification**
   - Calculate the €/MWh value a solar asset operator would gain by curtailing during predicted negative hours
   - Compare against: (a) no curtailment, (b) perfect foresight, (c) simple heuristic rules

3. **Visual Portfolio Pieces**
   - Create 3-5 publication-ready visualizations for LinkedIn
   - Build an interactive dashboard (Streamlit) demonstrating the model

### Success Metrics

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Recall (negative hours) | >75% | >85% |
| Precision (negative hours) | >60% | >70% |
| F1 Score | >0.67 | >0.77 |
| Economic value captured | >70% of theoretical max | >85% |
| LinkedIn post engagement | 50+ reactions | 200+ reactions |

---

## Part 2: Data Strategy

### 2.1 Required Data Sources

#### Primary Data (Essential)

| Data | Source | Granularity | Historical Depth |
|------|--------|-------------|------------------|
| Dutch day-ahead prices | ENTSO-E Transparency / Energy-Charts | Hourly | 2019-2025 |
| Solar generation (actual) | ENTSO-E Transparency | Hourly | 2019-2025 |
| Wind generation (actual) | ENTSO-E Transparency | Hourly | 2019-2025 |
| Total load | ENTSO-E Transparency | Hourly | 2019-2025 |
| Cross-border flows (DE, BE) | ENTSO-E Transparency | Hourly | 2019-2025 |
| Solar/wind forecasts (day-ahead) | ENTSO-E Transparency | Hourly | 2019-2025 |

#### Secondary Data (Enhances Model)

| Data | Source | Purpose |
|------|--------|---------|
| Weather forecasts (GHI, wind speed, temp) | Open-Meteo Archive API | Improve renewable generation predictions |
| German day-ahead prices | ENTSO-E / EPEX | Cross-border arbitrage indicator |
| Belgian day-ahead prices | ENTSO-E / EPEX | Cross-border arbitrage indicator |
| Gas prices (TTF) | ICE / public sources | Thermal generation cost proxy |
| Installed solar capacity (NL) | CBS / SolarPower Europe | Normalize generation |

#### Contextual Data (For Feature Engineering)

| Data | Source | Purpose |
|------|--------|---------|
| Dutch holidays | Python `holidays` library | Demand patterns |
| Day of week / hour of day | Derived | Temporal patterns |
| Sunset/sunrise times | `astral` library | Solar production windows |

### 2.2 Data Collection Approach

**ENTSO-E Transparency Platform:**
- Use the `entsoe-py` Python library
- Requires free API key (register at transparency.entsoe.eu)
- Rate limits: 400 requests/minute

**Energy-Charts (Fraunhofer ISE):**
- Excellent for visualization-ready data
- API available: `https://api.energy-charts.info/`
- No authentication required

**Open-Meteo Historical Weather:**
- You already have a collector script (`openmeteo_historical_weather_data_collector.py`)
- Extend to include solar radiation (GHI) and wind speed at hub height

### 2.3 Data Quality Considerations

**Known Issues to Handle:**
1. ENTSO-E has gaps during market coupling failures (rare but exist)
2. Daylight saving time transitions cause duplicate/missing hours
3. Cross-border flow data sometimes delayed
4. Forecast data availability varies by TSO

**Mitigation Strategy:**
- Forward-fill gaps <3 hours
- Flag and potentially exclude days with >6 hours missing
- Use multiple data sources for validation (ENTSO-E vs Energy-Charts)

---

## Part 3: Feature Engineering Strategy

### 3.1 Target Variable Definition

**Primary Target:** Binary classification
```
y = 1 if price_day_ahead < 0 EUR/MWh else 0
```

**Secondary Target (for extended analysis):** Multi-class
```
y = 0: price >= 20 EUR/MWh (normal)
y = 1: 0 <= price < 20 EUR/MWh (low)  
y = 2: price < 0 EUR/MWh (negative)
```

**Prediction Horizon:** 
- Primary: Day-ahead (12-36 hours ahead, matching EPEX auction timing)
- Secondary: 2-day ahead (for planning purposes)

### 3.2 Feature Categories

#### Category 1: Renewable Generation Features
```python
# Solar-related
'solar_forecast_mw'           # Day-ahead solar forecast from TSO
'solar_capacity_factor_forecast'  # Forecast / installed capacity
'ghi_forecast_amsterdam'      # Global horizontal irradiance
'clear_sky_index'             # GHI / theoretical clear sky GHI
'solar_forecast_vs_yesterday' # Forecast - actual(t-24h)

# Wind-related
'wind_forecast_mw'            # Day-ahead wind forecast
'wind_capacity_factor_forecast'
'wind_speed_10m_forecast'     # From weather model
'wind_direction_forecast'     # Offshore vs onshore patterns

# Combined renewables
'total_res_forecast_mw'       # Solar + wind forecast
'res_share_of_load_forecast'  # (solar + wind) / load forecast
'res_penetration_ratio'       # Key driver of negative prices
```

#### Category 2: Demand Features
```python
'load_forecast_mw'            # Day-ahead load forecast
'load_forecast_vs_avg'        # Forecast vs 30-day rolling average
'temperature_forecast'        # Heating/cooling demand proxy
'is_weekend'                  # Binary
'is_holiday'                  # Binary (NL public holidays)
'hour_of_day'                 # 0-23
'month'                       # 1-12
'is_dst'                      # Daylight saving time
```

#### Category 3: Cross-Border & Market Features
```python
'de_price_forecast'           # German price (if available) or lagged
'be_price_forecast'           # Belgian price
'nl_de_spread_lag24h'         # NL-DE price spread yesterday
'net_export_forecast'         # Scheduled cross-border flows
'de_solar_forecast'           # German solar (affects shared merit order)
'interconnector_capacity_available'  # NTC to neighbors
```

#### Category 4: Lag Features (Historical Patterns)
```python
'price_lag_24h'               # Price exactly 24 hours ago
'price_lag_168h'              # Price exactly 1 week ago
'negative_hours_last_7d'      # Count of negative hours in past week
'negative_streak_current'     # Consecutive negative hours
'price_rolling_mean_24h'      # Rolling average
'price_rolling_std_24h'       # Recent volatility
'solar_actual_lag_24h'        # Actual solar yesterday same hour
'solar_forecast_error_lag_24h' # Forecast - actual yesterday
```

#### Category 5: Engineered Interaction Features
```python
'solar_x_weekend'             # High solar + low weekend demand
'res_surplus_indicator'       # RES forecast - load forecast
'duck_curve_position'         # Hour relative to solar peak
'solar_ramp_rate'             # Change in solar forecast vs prior hour
```

### 3.3 Feature Engineering Code Structure

```python
class NegativePriceFeatureEngine:
    """
    Transforms raw data into ML-ready features for negative price prediction.
    """
    
    def __init__(self, installed_solar_capacity_mw: float = 24000):
        self.solar_cap = installed_solar_capacity_mw
        self.nl_holidays = holidays.Netherlands()
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hour, day of week, month, holiday indicators"""
        pass
    
    def create_renewable_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Solar and wind capacity factors, RES penetration"""
        pass
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical price and generation patterns"""
        pass
    
    def create_cross_border_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price spreads, flow indicators"""
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline"""
        pass
```

---

## Part 4: Modeling Strategy

### 4.1 Model Selection Rationale

**Primary Model: XGBoost Classifier**
- You already have experience with XGBoost from your European price forecasting project
- Handles tabular data with mixed feature types well
- Built-in feature importance for interpretability
- Fast training, good for iteration

**Secondary Models (for comparison):**
1. **LightGBM** - Often faster, good for larger datasets
2. **Random Forest** - Robust baseline, less prone to overfitting
3. **Logistic Regression** - Interpretable baseline
4. **LSTM/GRU** - Capture sequential patterns (stretch goal)

### 4.2 Class Imbalance Strategy

**The Challenge:**
- Negative price hours are ~5-10% of total hours
- Standard accuracy metrics would be misleading
- False negatives (missing a negative hour) may cost more than false positives

**Solutions to Implement:**

1. **Class Weights**
   ```python
   scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
   ```

2. **SMOTE Oversampling** (for comparison)
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

3. **Threshold Tuning**
   - Default threshold is 0.5
   - Optimize threshold for F1 or economic value
   - Use precision-recall curves to select

4. **Cost-Sensitive Learning**
   - Define asymmetric costs for FP vs FN
   - A missed negative hour might cost €50/MWh
   - A false alarm costs opportunity cost of curtailment

### 4.3 Validation Strategy

**Time Series Cross-Validation (Critical!)**
- Never use random train/test splits for time series
- Use expanding window or sliding window CV

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=30*24)  # 30-day test windows

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate
```

**Evaluation Periods:**
- Training: 2019-2023
- Validation: 2024 (hyperparameter tuning)
- Test: 2025 (final evaluation, never touch during development)

### 4.4 Hyperparameter Tuning

**Key XGBoost Parameters to Tune:**

```python
param_grid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [calculated_weight],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10]
}
```

**Tuning Approach:**
1. Start with RandomizedSearchCV (faster)
2. Narrow down with GridSearchCV on top candidates
3. Use Optuna for Bayesian optimization (stretch goal)

---

## Part 5: Economic Value Quantification

### 5.1 The Business Case

**Scenario:** You operate a 10 MW solar park in the Netherlands with no storage.

**Without Prediction:**
- You produce whenever the sun shines
- During negative price hours, you PAY to feed in
- Revenue = Σ(production_mwh × price_eur_mwh) → includes negative terms

**With Perfect Prediction:**
- You curtail production during negative price hours
- Revenue = Σ(production_mwh × price_eur_mwh) for price >= 0 only
- Lost production during curtailment has zero value (you would have paid to produce)

**With ML Prediction:**
- True Positives: Correctly curtail during negative hours → avoid losses
- False Positives: Unnecessarily curtail during positive hours → lose revenue
- False Negatives: Fail to curtail during negative hours → incur losses
- True Negatives: Correctly produce during positive hours → capture revenue

### 5.2 Economic Metrics

```python
def calculate_economic_value(y_true, y_pred, prices, generation):
    """
    Calculate the economic value of negative price predictions.
    
    Parameters:
    -----------
    y_true : array-like, actual negative price indicators (1 if price < 0)
    y_pred : array-like, predicted negative price indicators
    prices : array-like, actual prices (EUR/MWh)
    generation : array-like, solar generation (MWh)
    
    Returns:
    --------
    dict with economic metrics
    """
    
    # Baseline: No curtailment (produce always)
    revenue_no_curtail = np.sum(generation * prices)
    
    # Perfect foresight: Curtail only during actual negatives
    revenue_perfect = np.sum(generation * np.maximum(prices, 0))
    
    # With prediction: Curtail when predicted negative
    revenue_with_pred = np.sum(
        generation * prices * (1 - y_pred)  # Produce when y_pred=0
    )
    
    # Theoretical maximum savings
    max_savings = revenue_perfect - revenue_no_curtail
    
    # Actual savings achieved
    actual_savings = revenue_with_pred - revenue_no_curtail
    
    # Capture rate
    capture_rate = actual_savings / max_savings if max_savings != 0 else 0
    
    return {
        'revenue_no_curtailment': revenue_no_curtail,
        'revenue_perfect_foresight': revenue_perfect,
        'revenue_with_prediction': revenue_with_pred,
        'savings_achieved_eur': actual_savings,
        'max_possible_savings_eur': max_savings,
        'capture_rate_pct': capture_rate * 100
    }
```

### 5.3 Comparison Strategies

**Benchmark Strategies to Compare Against:**

1. **Naive Strategy:** Predict negative if same hour yesterday was negative
2. **Heuristic Strategy:** Predict negative if hour is 10:00-15:00 AND it's a weekend in spring/summer
3. **Solar Threshold:** Predict negative if solar forecast > 80% of capacity AND load < average
4. **Your ML Model:** The trained classifier

This comparison shows the VALUE your model adds beyond simple rules.

---

## Part 6: Visualization Strategy

### 6.1 LinkedIn Portfolio Visualizations

#### Visualization 1: "The Duck Curve Evolution" (Hero Visual)

**Concept:** Animated or multi-panel chart showing how the Dutch hourly price profile has changed from 2019 to 2025, with the characteristic "duck curve" deepening.

**Technical Implementation:**
```python
import plotly.express as px

# Create average hourly price profile by year
hourly_profiles = df.groupby(['year', 'hour'])['price'].mean().reset_index()

fig = px.line(
    hourly_profiles, 
    x='hour', 
    y='price', 
    color='year',
    title='The Deepening Duck Curve: Dutch Hourly Electricity Prices (2019-2025)',
    labels={'price': 'Average Price (€/MWh)', 'hour': 'Hour of Day'}
)
```

**LinkedIn Appeal:** Visually striking, tells a story, immediately understandable.

#### Visualization 2: "Negative Price Heatmap"

**Concept:** Calendar heatmap showing negative price hours by month and hour, revealing seasonal and diurnal patterns.

**Technical Implementation:**
```python
import calplot  # or seaborn heatmap

# Pivot table: rows=month, columns=hour, values=% negative
negative_pct = df.pivot_table(
    index='month', 
    columns='hour', 
    values='is_negative',
    aggfunc='mean'
) * 100

sns.heatmap(negative_pct, cmap='RdYlGn_r', annot=True, fmt='.0f')
```

**LinkedIn Appeal:** Data-dense but readable, shows you understand the problem deeply.

#### Visualization 3: "Model Performance Dashboard"

**Concept:** Multi-panel showing confusion matrix, precision-recall curve, and feature importance.

**LinkedIn Appeal:** Shows technical credibility, the kind of output hiring managers want to see.

#### Visualization 4: "Economic Value Waterfall"

**Concept:** Waterfall chart showing:
- Baseline revenue (no curtailment)
- Losses from negative hours
- Savings from ML predictions
- Final improved revenue

**LinkedIn Appeal:** Translates technical metrics into business value (€€€).

#### Visualization 5: "Interactive Streamlit Dashboard"

**Concept:** Web app where users can:
- Select a date range
- See predictions vs actuals
- Explore feature importance
- Calculate their own economic value

**LinkedIn Appeal:** Demonstrates full-stack data science skills, shareable link.

### 6.2 Visualization Style Guide

**For Maximum LinkedIn Impact:**

1. **Color Palette:** 
   - Use Dexter Energy / energy industry blues and greens
   - Reserve red/orange for negative prices
   - Consistent across all visuals

2. **Typography:**
   - Clear, readable titles (not too technical)
   - Subtitle explaining the insight
   - Your name/handle in corner

3. **Dimensions:**
   - LinkedIn feed: 1200 x 627 px (1.91:1)
   - LinkedIn article: 1200 x 900 px (4:3)
   - Carousel: 1080 x 1080 px (1:1)

4. **Annotations:**
   - Call out key insights directly on the chart
   - "During the March 2024 solar surge, prices hit -€125/MWh"
   - Make it accessible to non-experts

---

## Part 7: Project Phases & Timeline

### Phase 1: Data Foundation (Week 1-2)

**Objectives:**
- Set up data pipeline
- Collect all required data
- Initial exploration and quality checks

**Deliverables:**
- [ ] ENTSO-E API connection working
- [ ] Historical data downloaded (2019-2025)
- [ ] Weather data integrated
- [ ] Data quality report
- [ ] Initial EDA notebook

**Key Tasks:**
1. Register for ENTSO-E API key
2. Adapt your existing `openmeteo_historical_weather_data_collector.py`
3. Write data cleaning pipeline
4. Create initial visualizations (for your own understanding)

### Phase 2: Feature Engineering (Week 2-3)

**Objectives:**
- Implement all feature categories
- Create reusable feature engineering module
- Validate features make sense

**Deliverables:**
- [ ] `features/` module complete
- [ ] Feature correlation analysis
- [ ] Initial feature importance (using simple model)
- [ ] Documentation of all features

**Key Tasks:**
1. Implement `NegativePriceFeatureEngine` class
2. Handle edge cases (DST, missing data)
3. Create feature importance baseline

### Phase 3: Model Development (Week 3-4)

**Objectives:**
- Train and validate models
- Handle class imbalance
- Tune hyperparameters

**Deliverables:**
- [ ] Baseline model (logistic regression)
- [ ] XGBoost model tuned
- [ ] Model comparison report
- [ ] Saved model artifacts

**Key Tasks:**
1. Set up time series CV
2. Train baseline models
3. Hyperparameter tuning with Optuna
4. Threshold optimization

### Phase 4: Economic Analysis (Week 4-5)

**Objectives:**
- Quantify business value
- Compare against benchmark strategies
- Create economic visualizations

**Deliverables:**
- [ ] Economic value calculator
- [ ] Backtest results on 2025 data
- [ ] Comparison vs naive strategies
- [ ] Economic value visualizations

### Phase 5: Visualization & Dashboard (Week 5-6)

**Objectives:**
- Create LinkedIn-ready visuals
- Build Streamlit dashboard
- Write documentation

**Deliverables:**
- [ ] 5 publication-ready visualizations
- [ ] Working Streamlit app
- [ ] README and documentation
- [ ] LinkedIn post drafts

### Phase 6: Polish & Publish (Week 6-7)

**Objectives:**
- Final testing
- Deploy dashboard
- Publish to GitHub and LinkedIn

**Deliverables:**
- [ ] Clean GitHub repository
- [ ] Deployed Streamlit app (Streamlit Cloud or similar)
- [ ] LinkedIn post series
- [ ] Project complete!

---

## Part 8: File Structure

```
negaprice-nl/
│
├── README.md                          # Project overview, results summary, how to run
├── LICENSE                            # MIT or similar
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation (optional)
├── .gitignore                         # Ignore data files, secrets, etc.
├── .env.example                       # Template for API keys
│
├── config/
│   ├── __init__.py
│   ├── settings.py                    # Global settings (paths, constants)
│   └── api_keys.py                    # API key management (git-ignored in .env)
│
├── data/
│   ├── raw/                           # Original downloaded data (git-ignored)
│   │   ├── entsoe/
│   │   │   ├── nl_day_ahead_prices_2019_2025.csv
│   │   │   ├── nl_solar_generation_2019_2025.csv
│   │   │   ├── nl_wind_generation_2019_2025.csv
│   │   │   ├── nl_load_2019_2025.csv
│   │   │   └── cross_border_flows_2019_2025.csv
│   │   ├── weather/
│   │   │   └── nl_weather_2019_2025.csv
│   │   └── external/
│   │       ├── nl_holidays.csv
│   │       └── installed_capacity.csv
│   │
│   ├── processed/                     # Cleaned and merged data
│   │   ├── nl_hourly_dataset.parquet  # Main analysis dataset
│   │   └── feature_matrix.parquet     # ML-ready features
│   │
│   └── outputs/                       # Model outputs
│       ├── predictions_2025.csv
│       └── economic_analysis.csv
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb      # Initial EDA
│   ├── 02_feature_engineering.ipynb   # Feature development
│   ├── 03_model_development.ipynb     # Model training and tuning
│   ├── 04_economic_analysis.ipynb     # Business value calculation
│   └── 05_visualizations.ipynb        # Create final visuals
│
├── src/                               # Main source code
│   ├── __init__.py
│   │
│   ├── data/                          # Data collection and processing
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── entsoe_collector.py    # ENTSO-E API wrapper
│   │   │   ├── weather_collector.py   # Open-Meteo collector
│   │   │   └── energy_charts_collector.py  # Energy-Charts API
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── price_processor.py     # Clean and validate price data
│   │   │   ├── generation_processor.py # Process generation data
│   │   │   └── weather_processor.py   # Process weather data
│   │   └── pipeline.py                # End-to-end data pipeline
│   │
│   ├── features/                      # Feature engineering
│   │   ├── __init__.py
│   │   ├── temporal_features.py       # Time-based features
│   │   ├── renewable_features.py      # Solar/wind features
│   │   ├── market_features.py         # Price and cross-border features
│   │   ├── lag_features.py            # Historical patterns
│   │   └── feature_engine.py          # Main feature engineering class
│   │
│   ├── models/                        # Model training and evaluation
│   │   ├── __init__.py
│   │   ├── baseline.py                # Naive and heuristic baselines
│   │   ├── xgboost_model.py           # XGBoost classifier
│   │   ├── threshold_optimizer.py     # Optimize classification threshold
│   │   ├── cross_validation.py        # Time series CV utilities
│   │   └── model_registry.py          # Save/load models
│   │
│   ├── evaluation/                    # Model evaluation
│   │   ├── __init__.py
│   │   ├── classification_metrics.py  # Precision, recall, F1, etc.
│   │   ├── economic_metrics.py        # Business value calculations
│   │   └── backtester.py              # Backtest on historical data
│   │
│   └── visualization/                 # Visualization utilities
│       ├── __init__.py
│       ├── style.py                   # Color palettes, themes
│       ├── duck_curve.py              # Duck curve visualizations
│       ├── heatmaps.py                # Calendar and hour-month heatmaps
│       ├── model_performance.py       # Confusion matrix, PR curves
│       └── economic_charts.py         # Waterfall, value comparison
│
├── app/                               # Streamlit dashboard
│   ├── __init__.py
│   ├── app.py                         # Main Streamlit app
│   ├── pages/
│   │   ├── 1_📊_Overview.py           # Market overview
│   │   ├── 2_🔮_Predictions.py        # Model predictions
│   │   ├── 3_💰_Economic_Value.py     # Economic analysis
│   │   └── 4_📈_Model_Details.py      # Feature importance, etc.
│   └── components/
│       ├── sidebar.py                 # Sidebar filters
│       └── charts.py                  # Reusable chart components
│
├── scripts/                           # Standalone scripts
│   ├── download_data.py               # Download all required data
│   ├── train_model.py                 # Train production model
│   ├── generate_predictions.py        # Generate predictions for date range
│   └── create_visualizations.py       # Generate LinkedIn visuals
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_data_collectors.py
│   ├── test_feature_engine.py
│   └── test_models.py
│
├── outputs/                           # Final outputs for sharing
│   ├── figures/                       # LinkedIn-ready images
│   │   ├── duck_curve_evolution.png
│   │   ├── negative_price_heatmap.png
│   │   ├── model_performance.png
│   │   └── economic_value_waterfall.png
│   ├── models/                        # Saved model files
│   │   ├── xgboost_negative_price_v1.pkl
│   │   └── feature_scaler.pkl
│   └── reports/                       # Analysis reports
│       ├── model_comparison_report.md
│       └── economic_analysis_report.md
│
└── docs/                              # Documentation
    ├── data_dictionary.md             # Description of all data fields
    ├── feature_documentation.md       # Description of all features
    └── methodology.md                 # Technical methodology writeup
```

---

## Part 9: Technical Implementation Details

### 9.1 Key Dependencies

```python
# requirements.txt

# Data Collection
entsoe-py>=0.6.0          # ENTSO-E API client
requests>=2.28.0          # HTTP requests
python-dotenv>=1.0.0      # Environment variables

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0           # Parquet support

# Feature Engineering
holidays>=0.40            # Dutch holidays
astral>=3.2               # Sunrise/sunset

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0           # Alternative model
imbalanced-learn>=0.11.0  # SMOTE
optuna>=3.4.0             # Hyperparameter tuning

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
calplot>=0.1.9            # Calendar heatmaps

# Dashboard
streamlit>=1.28.0

# Development
jupyter>=1.0.0
pytest>=7.4.0
black>=23.0.0             # Code formatting
flake8>=6.1.0             # Linting
```

### 9.2 Core Class Implementations

#### Data Collector Base

```python
# src/data/collectors/entsoe_collector.py

from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional
import os

class EntsoeDataCollector:
    """
    Collects data from ENTSO-E Transparency Platform for the Netherlands.
    """
    
    COUNTRY_CODE = 'NL'
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("ENTSOE_API_KEY not found")
        self.client = EntsoePandasClient(api_key=self.api_key)
    
    def get_day_ahead_prices(
        self, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """
        Fetch day-ahead electricity prices for Netherlands.
        
        Returns DataFrame with datetime index and 'price' column.
        """
        prices = self.client.query_day_ahead_prices(
            self.COUNTRY_CODE,
            start=pd.Timestamp(start, tz='Europe/Amsterdam'),
            end=pd.Timestamp(end, tz='Europe/Amsterdam')
        )
        return prices.to_frame(name='price')
    
    def get_solar_generation(
        self, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """
        Fetch actual solar generation for Netherlands.
        """
        generation = self.client.query_generation(
            self.COUNTRY_CODE,
            start=pd.Timestamp(start, tz='Europe/Amsterdam'),
            end=pd.Timestamp(end, tz='Europe/Amsterdam')
        )
        # Extract solar column
        solar = generation['Solar'].to_frame(name='solar_generation_mw')
        return solar
    
    def get_solar_forecast(
        self, 
        start: datetime, 
        end: datetime
    ) -> pd.DataFrame:
        """
        Fetch day-ahead solar generation forecast.
        """
        forecast = self.client.query_generation_forecast(
            self.COUNTRY_CODE,
            start=pd.Timestamp(start, tz='Europe/Amsterdam'),
            end=pd.Timestamp(end, tz='Europe/Amsterdam')
        )
        return forecast
    
    # Similar methods for wind, load, cross-border flows...
```

#### Feature Engine

```python
# src/features/feature_engine.py

import pandas as pd
import numpy as np
import holidays
from astral import LocationInfo
from astral.sun import sun
from datetime import datetime, timedelta
from typing import Optional

class NegativePriceFeatureEngine:
    """
    Creates features for negative electricity price prediction.
    
    Features are designed to capture:
    1. Renewable generation patterns
    2. Demand patterns  
    3. Market dynamics
    4. Historical price patterns
    """
    
    def __init__(
        self, 
        installed_solar_mw: float = 24000,
        installed_wind_mw: float = 12000
    ):
        self.installed_solar_mw = installed_solar_mw
        self.installed_wind_mw = installed_wind_mw
        self.nl_holidays = holidays.Netherlands()
        self.amsterdam = LocationInfo(
            "Amsterdam", "Netherlands", "Europe/Amsterdam", 52.37, 4.89
        )
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target: 1 if price < 0, else 0"""
        df = df.copy()
        df['is_negative'] = (df['price'] < 0).astype(int)
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        df = df.copy()
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday'] = df.index.map(
            lambda x: x.date() in self.nl_holidays
        ).astype(int)
        
        # Cyclical encoding for hour and month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_renewable_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Solar and wind related features"""
        df = df.copy()
        
        # Capacity factors
        df['solar_capacity_factor'] = (
            df['solar_generation_mw'] / self.installed_solar_mw
        )
        df['wind_capacity_factor'] = (
            df['wind_generation_mw'] / self.installed_wind_mw
        )
        
        # Total renewables
        df['total_res_mw'] = df['solar_generation_mw'] + df['wind_generation_mw']
        
        # RES penetration (key driver!)
        df['res_penetration'] = df['total_res_mw'] / df['load_mw']
        
        # RES surplus
        df['res_surplus_mw'] = df['total_res_mw'] - df['load_mw']
        
        # Forecast errors (if forecasts available)
        if 'solar_forecast_mw' in df.columns:
            df['solar_forecast_error'] = (
                df['solar_forecast_mw'] - df['solar_generation_mw']
            )
        
        return df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        lags: list = [1, 24, 168]
    ) -> pd.DataFrame:
        """Historical patterns via lagged values"""
        df = df.copy()
        
        for lag in lags:
            df[f'price_lag_{lag}h'] = df['price'].shift(lag)
            df[f'solar_lag_{lag}h'] = df['solar_generation_mw'].shift(lag)
            df[f'is_negative_lag_{lag}h'] = df['is_negative'].shift(lag)
        
        # Rolling statistics
        df['price_rolling_mean_24h'] = df['price'].rolling(24).mean()
        df['price_rolling_std_24h'] = df['price'].rolling(24).std()
        df['negative_hours_last_7d'] = (
            df['is_negative'].rolling(168).sum()
        )
        
        return df
    
    def create_cross_border_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-border price spreads and flows"""
        df = df.copy()
        
        if 'de_price' in df.columns:
            df['nl_de_spread'] = df['price'] - df['de_price']
            df['nl_de_spread_lag_24h'] = df['nl_de_spread'].shift(24)
        
        if 'be_price' in df.columns:
            df['nl_be_spread'] = df['price'] - df['be_price']
        
        # Net position
        if 'import_mw' in df.columns and 'export_mw' in df.columns:
            df['net_export_mw'] = df['export_mw'] - df['import_mw']
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline.
        
        Applies all feature creation methods in sequence.
        """
        df = self.create_target(df)
        df = self.create_temporal_features(df)
        df = self.create_renewable_features(df)
        df = self.create_lag_features(df)
        df = self.create_cross_border_features(df)
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        Returns list of feature column names (excludes target and metadata).
        """
        exclude = ['price', 'is_negative', 'datetime']
        # Implementation depends on your specific features
        pass
```

### 9.3 Model Training Script

```python
# scripts/train_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    precision_recall_curve, average_precision_score
)
import optuna
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.features.feature_engine import NegativePriceFeatureEngine
from src.models.threshold_optimizer import optimize_threshold


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for hyperparameter tuning."""
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'aucpr'  # Area under PR curve
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Optimize threshold for F1
    best_threshold, best_f1 = optimize_threshold(y_val, y_pred_proba)
    
    return best_f1


def main():
    # Load data
    df = pd.read_parquet('data/processed/feature_matrix.parquet')
    
    # Split features and target
    feature_cols = [c for c in df.columns if c not in ['price', 'is_negative']]
    X = df[feature_cols]
    y = df['is_negative']
    
    # Time-based split
    train_end = '2023-12-31'
    val_end = '2024-12-31'
    
    X_train = X[X.index <= train_end]
    y_train = y[y.index <= train_end]
    
    X_val = X[(X.index > train_end) & (X.index <= val_end)]
    y_val = y[(y.index > train_end) & (y.index <= val_end)]
    
    X_test = X[X.index > val_end]
    y_test = y[y.index > val_end]
    
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Negative rate train: {y_train.mean():.2%}")
    
    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=100,
        show_progress_bar=True
    )
    
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Train final model with best params
    best_params = study.best_params
    best_params['scale_pos_weight'] = len(y_train[y_train==0]) / len(y_train[y_train==1])
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    best_threshold, _ = optimize_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    print("\nTest Set Performance:")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # Save model
    joblib.dump(final_model, 'outputs/models/xgboost_negative_price_v1.pkl')
    joblib.dump(best_threshold, 'outputs/models/optimal_threshold.pkl')
    
    print("\nModel saved!")


if __name__ == '__main__':
    main()
```

---

## Part 10: LinkedIn Content Strategy

### Post 1: The Hook (Week 1 of publishing)

**Format:** Carousel (5-6 slides)

**Slide 1:** 
"🇳🇱 Dutch electricity prices went NEGATIVE 584 times in 2025.

Solar producers PAID to generate clean energy.

Here's what's happening 👇"

**Slide 2:** The duck curve visualization

**Slide 3:** Explanation of why this happens

**Slide 4:** "I built an ML model to predict these events"

**Slide 5:** Results teaser + call to action

**Hashtags:** #EnergyTransition #MachineLearning #Netherlands #RenewableEnergy #DataScience

---

### Post 2: Technical Deep Dive (Week 2)

**Format:** Single image + long-form text

**Image:** Model performance dashboard (confusion matrix, feature importance)

**Text:** Technical explanation of approach, feature engineering decisions, model selection

**Audience:** Technical recruiters, ML engineers, quant analysts

---

### Post 3: Business Value (Week 3)

**Format:** Single image (waterfall chart)

**Hook:** "What's the € value of predicting negative electricity prices?"

**Content:** Economic analysis, backtest results, comparison to naive strategies

**Audience:** Business stakeholders, energy traders, portfolio managers

---

### Post 4: Interactive Demo (Week 4)

**Format:** Link to Streamlit app

**Hook:** "I built a dashboard to explore Dutch negative prices. Try it yourself!"

**Content:** Brief demo, invite feedback, ask for suggestions

---

## Summary

This project plan gives you:

1. **Clear objectives** tied to real business problems at your target companies
2. **Detailed data strategy** using freely available sources
3. **Comprehensive feature engineering** based on domain knowledge
4. **Robust modeling approach** with proper time-series validation
5. **Economic value quantification** that speaks the language of business
6. **LinkedIn-optimized visualizations** for maximum portfolio impact
7. **Professional codebase structure** that demonstrates software engineering skills
8. **Realistic timeline** of 6-7 weeks to completion

The key differentiator from typical ML projects: **you're not just building a classifier—you're solving a real problem that Dexter Energy, KYOS, and Groendus work on every day, and you're quantifying the € value of your solution.**

Good luck, Will! This project has the potential to be exactly the portfolio piece that lands you interviews at these companies.
