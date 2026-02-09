"""
Exploratory Data Analysis for NegaPriceNL Project

This script loads the collected data and creates beautiful visualizations
to understand patterns in Dutch electricity prices, generation, and weather.

Generates plots in outputs/figures/ directory.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from config.settings import (
    ENTSOE_DATA_DIR,
    WEATHER_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("="*70)
print("NegaPriceNL - Exploratory Data Analysis")
print("="*70)

# Create figures directory
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[*] Loading data...")

# Load prices
nl_prices = pd.read_csv(ENTSOE_DATA_DIR / "nl_day_ahead_prices_2019_2025.csv",
                        index_col=0, parse_dates=True)
be_prices = pd.read_csv(ENTSOE_DATA_DIR / "be_day_ahead_prices_2019_2025.csv",
                        index_col=0, parse_dates=True)

# Load generation
nl_solar = pd.read_csv(ENTSOE_DATA_DIR / "nl_solar_generation_2019_2025.csv",
                       index_col=0, parse_dates=True)
nl_wind = pd.read_csv(ENTSOE_DATA_DIR / "nl_wind_generation_2019_2025.csv",
                      index_col=0, parse_dates=True)

# Load weather (Amsterdam only for simplicity)
weather = pd.read_csv(WEATHER_DATA_DIR / "nl_weather_amsterdam_20190101_20251231.csv")
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather.set_index('datetime', inplace=True)

print(f"[+] Loaded {len(nl_prices)} price records")
print(f"[+] Loaded {len(nl_solar)} solar generation records")
print(f"[+] Loaded {len(nl_wind)} wind generation records")
print(f"[+] Loaded {len(weather)} weather records")

# ============================================================================
# 2. BASIC STATISTICS & NEGATIVE PRICE ANALYSIS
# ============================================================================

print("\n[*] Analyzing negative price events...")

# Identify negative prices
nl_prices['is_negative'] = (nl_prices['price'] < 0).astype(int)
nl_prices.index = pd.to_datetime(nl_prices.index, utc=True)
nl_prices['year'] = nl_prices.index.year
nl_prices['month'] = nl_prices.index.month
nl_prices['hour'] = nl_prices.index.hour
nl_prices['day_of_week'] = nl_prices.index.dayofweek

# Count negative hours by year
negative_by_year = nl_prices.groupby('year')['is_negative'].sum()
print("\n[!] Negative Price Hours by Year:")
for year, count in negative_by_year.items():
    total_hours = len(nl_prices[nl_prices['year'] == year])
    pct = (count / total_hours) * 100
    print(f"   {year}: {count} hours ({pct:.2f}%)")

# Most extreme negative price
most_negative = nl_prices['price'].min()
most_negative_date = nl_prices[nl_prices['price'] == most_negative].index[0]
print(f"\n[!] Most extreme negative price: EUR{most_negative:.2f}/MWh on {most_negative_date}")

# ============================================================================
# 3. VISUALIZATION 1: Price Distribution Over Time
# ============================================================================

print("\n[*] Creating Visualization 1: Price Time Series...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Top plot: Full time series
axes[0].plot(nl_prices.index, nl_prices['price'], linewidth=0.5, alpha=0.7, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Price')
axes[0].fill_between(nl_prices.index, 0, nl_prices['price'],
                     where=(nl_prices['price'] < 0),
                     color='red', alpha=0.3, label='Negative Prices')
axes[0].set_title('Dutch Day-Ahead Electricity Prices (2019-2025)', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Price (€/MWh)', fontsize=12)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Bottom plot: Distribution
axes[1].hist(nl_prices['price'], bins=200, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Price')
axes[1].set_title('Price Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Price (€/MWh)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_price_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 01_price_timeseries.png")

# ============================================================================
# 4. VISUALIZATION 2: Negative Price Heatmap (Hour x Month)
# ============================================================================

print("\n[*] Creating Visualization 2: Negative Price Patterns...")

# Calculate percentage of negative prices by hour and month
negative_pct = nl_prices.pivot_table(
    index='month',
    columns='hour',
    values='is_negative',
    aggfunc='mean'
) * 100

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(negative_pct, cmap='RdYlGn_r', annot=True, fmt='.1f',
            cbar_kws={'label': '% of Hours with Negative Prices'},
            linewidths=0.5, ax=ax)
ax.set_title('Negative Price Patterns: % of Hours by Month and Hour',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Month', fontsize=12)
ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_negative_price_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 02_negative_price_heatmap.png")

# ============================================================================
# 5. VISUALIZATION 3: The Duck Curve Evolution
# ============================================================================

print("\n[*] Creating Visualization 3: Duck Curve Evolution...")

# Calculate average hourly price profile by year
hourly_profiles = nl_prices.groupby(['year', 'hour'])['price'].mean().reset_index()

fig, ax = plt.subplots(figsize=(16, 9))

# Plot each year
for year in sorted(y for y in hourly_profiles['year'].unique() if y >= 2019):
    year_data = hourly_profiles[hourly_profiles['year'] == year]
    ax.plot(year_data['hour'], year_data['price'],
            linewidth=2.5, marker='o', markersize=4, label=str(year), alpha=0.8)

ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_title('The Deepening Duck Curve: Dutch Hourly Price Profiles (2019-2025)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Hour of Day', fontsize=13)
ax.set_ylabel('Average Price (€/MWh)', fontsize=13)
ax.set_xticks(range(0, 24))
ax.legend(title='Year', fontsize=11, title_fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_duck_curve_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 03_duck_curve_evolution.png")

# ============================================================================
# 6. VISUALIZATION 4: Solar & Wind Generation Patterns
# ============================================================================

print("\n[*] Creating Visualization 4: Generation Patterns...")

# Combine solar and wind
generation = pd.DataFrame({
    'solar': pd.to_numeric(nl_solar['solar_generation_mw'], errors='coerce'),
    'wind': pd.to_numeric(nl_wind['wind_generation_mw'], errors='coerce')
})
generation.index = pd.to_datetime(generation.index, utc=True)
generation = generation.dropna()
generation['total_res'] = generation['solar'] + generation['wind']
generation['year'] = generation.index.year
generation['hour'] = generation.index.hour

# Average generation by hour
hourly_gen = generation.groupby('hour')[['solar', 'wind']].mean()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top left: Solar generation profile
axes[0, 0].plot(hourly_gen.index, hourly_gen['solar'],
                linewidth=3, marker='o', color='orange', markersize=6)
axes[0, 0].fill_between(hourly_gen.index, 0, hourly_gen['solar'], alpha=0.3, color='orange')
axes[0, 0].set_title('Average Solar Generation Profile', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Hour of Day')
axes[0, 0].set_ylabel('Solar Generation (MW)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(0, 24))

# Top right: Wind generation profile
axes[0, 1].plot(hourly_gen.index, hourly_gen['wind'],
                linewidth=3, marker='o', color='steelblue', markersize=6)
axes[0, 1].fill_between(hourly_gen.index, 0, hourly_gen['wind'], alpha=0.3, color='steelblue')
axes[0, 1].set_title('Average Wind Generation Profile', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Wind Generation (MW)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(0, 24))

# Bottom left: Generation by year
yearly_gen = generation.groupby('year')[['solar', 'wind']].mean()
x = np.arange(len(yearly_gen.index))
width = 0.35
axes[1, 0].bar(x - width/2, yearly_gen['solar'], width, label='Solar', color='orange', alpha=0.8)
axes[1, 0].bar(x + width/2, yearly_gen['wind'], width, label='Wind', color='steelblue', alpha=0.8)
axes[1, 0].set_title('Average Generation by Year', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Average Generation (MW)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(yearly_gen.index)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Bottom right: Time series (last 30 days of 2024)
recent = generation['2024-12-01':'2024-12-31']
axes[1, 1].plot(recent.index, recent['solar'], label='Solar', color='orange', linewidth=1.5)
axes[1, 1].plot(recent.index, recent['wind'], label='Wind', color='steelblue', linewidth=1.5)
axes[1, 1].fill_between(recent.index, 0, recent['solar'], alpha=0.2, color='orange')
axes[1, 1].fill_between(recent.index, 0, recent['wind'], alpha=0.2, color='steelblue')
axes[1, 1].set_title('Generation Time Series (December 2024)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Generation (MW)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '04_generation_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 04_generation_patterns.png")

# ============================================================================
# 7. VISUALIZATION 5: Weather Patterns
# ============================================================================

print("\n[*] Creating Visualization 5: Weather Data...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# Temperature
axes[0, 0].plot(weather.index, weather['temperature_2m_c'], linewidth=0.5, alpha=0.6, color='darkred')
axes[0, 0].set_title('Temperature (2m)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].grid(True, alpha=0.3)

# Solar radiation (GHI)
axes[0, 1].plot(weather.index, weather['ghi_wm2'], linewidth=0.5, alpha=0.6, color='orange')
axes[0, 1].set_title('Global Horizontal Irradiance (GHI)', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('GHI (W/m²)')
axes[0, 1].grid(True, alpha=0.3)

# Wind speed at 100m
axes[1, 0].plot(weather.index, weather['wind_speed_100m_ms'], linewidth=0.5, alpha=0.6, color='steelblue')
axes[1, 0].set_title('Wind Speed at 100m (Hub Height)', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Wind Speed (m/s)')
axes[1, 0].grid(True, alpha=0.3)

# Cloud cover
axes[1, 1].plot(weather.index, weather['cloud_cover_pct'], linewidth=0.5, alpha=0.6, color='gray')
axes[1, 1].set_title('Cloud Cover', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Cloud Cover (%)')
axes[1, 1].grid(True, alpha=0.3)

# Average GHI by hour
weather['hour'] = weather.index.hour
hourly_ghi = weather.groupby('hour')['ghi_wm2'].mean()
axes[2, 0].plot(hourly_ghi.index, hourly_ghi.values, linewidth=3, marker='o', color='orange')
axes[2, 0].fill_between(hourly_ghi.index, 0, hourly_ghi.values, alpha=0.3, color='orange')
axes[2, 0].set_title('Average GHI by Hour of Day', fontsize=13, fontweight='bold')
axes[2, 0].set_xlabel('Hour of Day')
axes[2, 0].set_ylabel('GHI (W/m²)')
axes[2, 0].set_xticks(range(0, 24))
axes[2, 0].grid(True, alpha=0.3)

# Average wind speed by hour
hourly_wind = weather.groupby('hour')['wind_speed_100m_ms'].mean()
axes[2, 1].plot(hourly_wind.index, hourly_wind.values, linewidth=3, marker='o', color='steelblue')
axes[2, 1].fill_between(hourly_wind.index, 0, hourly_wind.values, alpha=0.3, color='steelblue')
axes[2, 1].set_title('Average Wind Speed by Hour of Day', fontsize=13, fontweight='bold')
axes[2, 1].set_xlabel('Hour of Day')
axes[2, 1].set_ylabel('Wind Speed (m/s)')
axes[2, 1].set_xticks(range(0, 24))
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_weather_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 05_weather_patterns.png")

# ============================================================================
# 8. VISUALIZATION 6: Price vs Generation Correlation
# ============================================================================

print("\n[*] Creating Visualization 6: Price-Generation Relationship...")

# Merge data for correlation analysis (sample to avoid overplotting)
merged = pd.DataFrame({
    'price': nl_prices['price'],
    'solar': nl_solar['solar_generation_mw'],
    'wind': nl_wind['wind_generation_mw']
}).dropna()
merged['total_res'] = merged['solar'] + merged['wind']
merged['is_negative'] = (merged['price'] < 0).astype(int)

# Sample for plotting
sample = merged.sample(n=min(5000, len(merged)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Scatter: Solar vs Price
mask_pos = sample['is_negative'] == 0
mask_neg = sample['is_negative'] == 1
axes[0].scatter(sample.loc[mask_pos, 'solar'], sample.loc[mask_pos, 'price'],
                alpha=0.3, s=20, color='#2ca02c', label='Positive price')
axes[0].scatter(sample.loc[mask_neg, 'solar'], sample.loc[mask_neg, 'price'],
                alpha=0.3, s=20, color='#d62728', label='Negative price')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[0].set_title('Price vs Solar Generation', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Solar Generation (MW)')
axes[0].set_ylabel('Price (€/MWh)')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10, markerscale=2)

# Scatter: Total RES vs Price
axes[1].scatter(sample.loc[mask_pos, 'total_res'], sample.loc[mask_pos, 'price'],
                alpha=0.3, s=20, color='#2ca02c', label='Positive price')
axes[1].scatter(sample.loc[mask_neg, 'total_res'], sample.loc[mask_neg, 'price'],
                alpha=0.3, s=20, color='#d62728', label='Negative price')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].set_title('Price vs Total Renewable Generation', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Total RES Generation (MW)')
axes[1].set_ylabel('Price (€/MWh)')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10, markerscale=2)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_price_generation_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 06_price_generation_correlation.png")

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================

# ============================================================================
# 10. VISUALIZATION 06a: Price vs Renewable Generation Proportion
# ============================================================================

print("\n[*] Creating Visualization 06a: Renewable Power Proportion vs Price...")

# Load unified dataset with total_generation_mw
unified = pd.read_csv(PROCESSED_DATA_DIR / "unified_dataset2.csv",
                       index_col=0, parse_dates=True)
unified.index = pd.to_datetime(unified.index, utc=True)

# Filter to 2024 only, and rows where total_generation_mw is valid and positive
unified_valid = unified[(unified.index.year == 2024) & unified['total_generation_mw'].notna() & (unified['total_generation_mw'] > 0)].copy()

# Compute renewable proportion
unified_valid['res_proportion'] = (
    (unified_valid['solar_generation_mw'].fillna(0) + unified_valid['wind_generation_mw'].fillna(0))
    / unified_valid['total_generation_mw']
)
unified_valid['is_negative'] = (unified_valid['price_eur_mwh'] < 0).astype(int)

# Sample for plotting (avoid overplotting)
sample_06a = unified_valid.sample(n=min(5000, len(unified_valid)), random_state=42)

fig, ax = plt.subplots(figsize=(16, 7))

mask_pos = sample_06a['is_negative'] == 0
mask_neg = sample_06a['is_negative'] == 1
ax.scatter(sample_06a.loc[mask_pos, 'res_proportion'] * 100, sample_06a.loc[mask_pos, 'price_eur_mwh'],
           alpha=0.3, s=20, color='#2ca02c', label='Positive price')
ax.scatter(sample_06a.loc[mask_neg, 'res_proportion'] * 100, sample_06a.loc[mask_neg, 'price_eur_mwh'],
           alpha=0.3, s=20, color='#d62728', label='Negative price')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_title('Price vs Renewable Generation Proportion (Solar + Wind / Total)',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Renewable Share of Total Generation (%)', fontsize=12)
ax.set_ylabel('Price (EUR/MWh)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, markerscale=2)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06a_renewable_power_proportion_generation.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 06a_renewable_power_proportion_generation.png")

# ============================================================================
# 11. VISUALIZATION 06b: Price vs Renewable Generation / Load
# ============================================================================

print("\n[*] Creating Visualization 06b: Renewable Power Proportion (of Load) vs Price...")

# Filter to 2024 only, and rows where load_mw is valid and positive
unified_load = unified[(unified.index.year == 2024) & unified['load_mw'].notna() & (unified['load_mw'] > 0)].copy()

# Compute renewable-to-load ratio
unified_load['res_to_load'] = (
    (unified_load['solar_generation_mw'].fillna(0) + unified_load['wind_generation_mw'].fillna(0))
    / unified_load['load_mw']
)
unified_load['is_negative'] = (unified_load['price_eur_mwh'] < 0).astype(int)

print(f"   Valid records: {len(unified_load):,}")
print(f"   RES/Load range: {unified_load['res_to_load'].min():.1%} – {unified_load['res_to_load'].max():.1%}")

# Sample for plotting
sample_06b = unified_load.sample(n=min(5000, len(unified_load)), random_state=42)
print(f"   Sampled {len(sample_06b):,} points ({sample_06b['is_negative'].sum()} negative)")

fig, ax = plt.subplots(figsize=(16, 7))

mask_pos = sample_06b['is_negative'] == 0
mask_neg = sample_06b['is_negative'] == 1
ax.scatter(sample_06b.loc[mask_pos, 'res_to_load'] * 100, sample_06b.loc[mask_pos, 'price_eur_mwh'],
           alpha=0.3, s=20, color='#2ca02c', label='Positive price')
ax.scatter(sample_06b.loc[mask_neg, 'res_to_load'] * 100, sample_06b.loc[mask_neg, 'price_eur_mwh'],
           alpha=0.3, s=20, color='#d62728', label='Negative price')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_title('Price vs Renewable Generation as Share of Load (Solar + Wind / Load)',
             fontsize=16, fontweight='bold')
ax.set_xlabel('Renewable Generation / Load (%)', fontsize=12)
ax.set_ylabel('Price (EUR/MWh)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, markerscale=2)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06b_renewable_power_proportion_load.png', dpi=300, bbox_inches='tight')
plt.close()
print("[+] Saved: 06b_renewable_power_proportion_load.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\n[*] Price Statistics:")
print(f"   Mean price: €{nl_prices['price'].mean():.2f}/MWh")
print(f"   Median price: €{nl_prices['price'].median():.2f}/MWh")
print(f"   Std deviation: €{nl_prices['price'].std():.2f}/MWh")
print(f"   Min price: €{nl_prices['price'].min():.2f}/MWh")
print(f"   Max price: €{nl_prices['price'].max():.2f}/MWh")
print(f"   Total negative hours: {nl_prices['is_negative'].sum()}")
print(f"   % negative hours: {(nl_prices['is_negative'].sum() / len(nl_prices)) * 100:.2f}%")

print(f"\n[*] Generation Statistics:")
print(f"   Average solar: {generation['solar'].mean():.2f} MW")
print(f"   Max solar: {generation['solar'].max():.2f} MW")
print(f"   Average wind: {generation['wind'].mean():.2f} MW")
print(f"   Max wind: {generation['wind'].max():.2f} MW")

print("\n" + "="*70)
print("[+] ANALYSIS COMPLETE!")
print("="*70)
print(f"\n[*] All visualizations saved to: {FIGURES_DIR}")
print("\nGenerated plots:")
print("   1. 01_price_timeseries.png - Full price history and distribution")
print("   2. 02_negative_price_heatmap.png - When negative prices occur")
print("   3. 03_duck_curve_evolution.png - Hourly price profile evolution")
print("   4. 04_generation_patterns.png - Solar and wind generation analysis")
print("   5. 05_weather_patterns.png - Weather data overview")
print("   6. 06_price_generation_correlation.png - Price vs renewable output")
print("\n" + "="*70)
