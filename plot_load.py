import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the data
load_df = pd.read_csv('data/raw/entsoe/nl_load_2019_2025.csv', index_col=0)
load_df.index = pd.to_datetime(load_df.index, utc=True)
load_forecast_df = pd.read_csv('data/raw/entsoe/nl_load_forecast_2019_2025.csv', index_col=0)
load_forecast_df.index = pd.to_datetime(load_forecast_df.index, utc=True)

# Get the last 3 days of data
last_date = load_df.index.max()
start_date = last_date - pd.Timedelta(days=3)

load_last3 = load_df[load_df.index >= start_date]
forecast_last3 = load_forecast_df[load_forecast_df.index >= start_date]

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(load_last3.index, load_last3['load_mw'], label='Actual Load', color='blue', linewidth=1.5)
ax.plot(forecast_last3.index, forecast_last3['load_forecast_mw'], label='Load Forecast', color='orange', linewidth=1.5, linestyle='--')

ax.set_xlabel('Date')
ax.set_ylabel('Load (MW)')
ax.set_title(f'Netherlands Load vs Load Forecast\n{start_date.strftime("%Y-%m-%d")} to {last_date.strftime("%Y-%m-%d")}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('load_vs_forecast_last3days.png', dpi=150)
plt.close()

print(f"Plot saved as 'load_vs_forecast_last3days.png'")
