# BESS Animation Visualisation — Claude Code Action Plan

## Goal

Build an animated, browser-based visualisation that replays the NegaPriceNL BESS simulation day-by-day. It should show a wind farm scene with weather conditions changing to match real data, a battery system charging/discharging, the day-ahead price, and cumulative revenue — all driven by the actual simulation output data.

The end product is a single-page React app (Vite) in a `viz/` folder at the repo root. It reads a static JSON export of the simulation results. No backend.

---

## Step 0 — Read Before You Start

Familiarise yourself with these files to understand the data structures:

- `src/simulation/market.py` — `DailyOutcome` dataclass (the core data unit)
- `src/simulation/metrics.py` — `outcomes_to_dataframe()` shows which fields exist per day
- `src/simulation/assets.py` — `WindFarm` and `BatteryStorage` classes
- `src/simulation/portfolio_backtester.py` — `PortfolioBacktester.run()` is the main loop
- `scripts/run_bess_simulation.py` — the runner that executes everything
- `config/settings_bess.py` — all physical parameters (capacities, efficiencies, quantiles)
- `config/settings_v10.py` — test period dates, model paths

---

## Step 1 — Data Export Script

**Create `scripts/export_viz_data.py`**

This script re-runs (or loads cached results from) the BESS simulation and exports a JSON file that the React app will consume. It needs MTU-level (quarter-hourly) granularity, not just daily summaries.

### What to export

For each day in the test period (2025-01-01 to 2025-12-14), for each of the 96 quarter-hours:

```python
{
    "metadata": {
        "wind_capacity_mw": 50.0,
        "bess_power_mw": 25.0,
        "bess_energy_mwh": 50.0,
        "bess_soc_min_mwh": 5.0,
        "bess_soc_max_mwh": 45.0,
        "test_start": "2025-01-01",
        "test_end": "2025-12-14",
        "strategy": "Conservative (q25/q75)"
    },
    "days": [
        {
            "date": "2025-01-01",
            "daily_summary": {
                "wind_revenue_eur": 12345.67,
                "bess_net_pnl_eur": 890.12,
                "total_portfolio_revenue_eur": 13235.79,
                "energy_charged_mwh": 30.5,
                "energy_discharged_mwh": 28.1,
                "cycles": 0.56
            },
            "mtus": [
                {
                    "t": 0,
                    "hour": 0,
                    "quarter": 0,
                    "price_eur_mwh": 45.20,
                    "wind_generation_mw": 18.5,
                    "wind_speed_100m": 8.2,
                    "cloud_cover_pct": 65,
                    "temperature_c": 3.1,
                    "soc_mwh": 25.0,
                    "charge_mw": 0.0,
                    "discharge_mw": 0.0,
                    "charge_cleared": false,
                    "discharge_cleared": false
                },
                // ... 95 more MTUs
            ]
        },
        // ... more days
    ]
}
```

### How to get MTU-level data

The `DailyOutcome` dataclass already stores:
- `soc_timeseries`: shape (97,) — SoC at end of each MTU + initial
- `actual_charge_mw`: shape (96,) — charge power per MTU
- `actual_discharge_mw`: shape (96,) — discharge power per MTU
- `charge_cleared`: shape (96,) boolean
- `discharge_cleared`: shape (96,) boolean

For prices and wind, extract from the feature matrix (`test_df`) for each day:
- `price_eur_mwh` column
- `wind_generation_mw` column (national, then scale by farm share)
- `wind_speed_100m` from Open-Meteo features
- `cloud_cover_pct` from Open-Meteo features
- `temperature_2m` from Open-Meteo features

**Important**: The feature matrix is `data/processed/feature_matrix_v7.csv`. Weather columns may be named `wind_speed_100m`, `cloud_cover`, `temperature_2m` — check the actual column names. The wind generation column is `wind_generation_mw` (national level). Scale to farm level using `WIND_FARM_CAPACITY_MW / NL_INSTALLED_WIND_CAPACITY_MW` from settings.

### Implementation approach

The simplest approach: modify the existing simulation loop to also capture the per-MTU data. Or better — run the simulation normally, then for each `DailyOutcome`, zip it with the corresponding day's rows from `test_df` to build the export.

```python
# Pseudocode
model, test_df = load_model_and_data()
outcomes = backtester.run(test_df, feature_columns)

# Build export
test_df['_date'] = test_df.index.date
for outcome, (date, day_df) in zip(outcomes, test_df.groupby('_date')):
    day_export = {
        "date": str(date),
        "daily_summary": { ... from outcome ... },
        "mtus": []
    }
    for t in range(96):
        mtu = {
            "t": t,
            "price_eur_mwh": float(day_df.iloc[t]['price_eur_mwh']),
            "wind_generation_mw": float(day_df.iloc[t]['wind_generation_mw']) * farm_share,
            "wind_speed_100m": float(day_df.iloc[t].get('wind_speed_100m', 0)),
            "cloud_cover_pct": float(day_df.iloc[t].get('cloud_cover', 0)),
            "temperature_c": float(day_df.iloc[t].get('temperature_2m', 0)),
            "soc_mwh": float(outcome.soc_timeseries[t + 1]),  # +1 because index 0 is initial
            "charge_mw": float(outcome.actual_charge_mw[t]),
            "discharge_mw": float(outcome.actual_discharge_mw[t]),
            "charge_cleared": bool(outcome.charge_cleared[t]),
            "discharge_cleared": bool(outcome.discharge_cleared[t]),
        }
        day_export["mtus"].append(mtu)
```

Save to `viz/src/data/simulation_output.json`. The file will be large (~35MB for 348 days × 96 MTUs). Consider also creating a compressed/sampled version for development.

**Dev shortcut**: Also export a `simulation_output_sample.json` with just 7 days (one interesting week with negative prices) for fast iteration.

---

## Step 2 — React App Scaffolding

### Folder structure

```
viz/
├── package.json
├── vite.config.js
├── index.html
├── .gitignore                    # node_modules, dist
├── README.md
├── src/
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css                 # Tailwind or global styles
│   ├── hooks/
│   │   └── useSimulation.js      # Data loading + playback state machine
│   ├── components/
│   │   ├── Scene.jsx             # Main layout container
│   │   ├── WindFarm.jsx          # Animated turbines (SVG)
│   │   ├── BatteryGauge.jsx      # SoC bar/fill animation
│   │   ├── PriceTicker.jsx       # Current price with colour coding
│   │   ├── RevenuePanel.jsx      # Cumulative revenue counters
│   │   ├── WeatherOverlay.jsx    # Wind speed, temp, cloud indicators
│   │   ├── TimelineControls.jsx  # Play/pause, speed, scrubber, date display
│   │   └── DayChart.jsx          # Optional: small price chart for current day
│   └── data/
│       ├── simulation_output.json
│       └── simulation_output_sample.json
└── public/
    └── favicon.ico
```

### Setup commands

```bash
cd viz
npm create vite@latest . -- --template react
npm install
npm install framer-motion recharts lucide-react
npm install -D tailwindcss @tailwindcss/vite
```

Configure Tailwind in vite.config.js and index.css per Tailwind v4 docs.

---

## Step 3 — Playback Engine (`useSimulation` hook)

This is the core logic. It manages:

1. **Loading**: Fetch and parse the JSON data
2. **Playback state**: current day index, current MTU index, playing/paused, speed
3. **Derived values**: cumulative revenue up to current point, current weather, etc.

```javascript
// Key state
const [dayIndex, setDayIndex] = useState(0);
const [mtuIndex, setMtuIndex] = useState(0);
const [isPlaying, setIsPlaying] = useState(false);
const [speed, setSpeed] = useState(1); // 1 = 1 MTU per 100ms, up to 10x

// Derived
const currentDay = data.days[dayIndex];
const currentMtu = currentDay.mtus[mtuIndex];
const cumulativeRevenue = data.days
    .slice(0, dayIndex)
    .reduce((sum, d) => sum + d.daily_summary.total_portfolio_revenue_eur, 0)
    + currentDay.mtus
        .slice(0, mtuIndex + 1)
        .reduce((sum, m) => {
            // Approximate MTU-level revenue from price × generation × 0.25h
            return sum + (m.wind_generation_mw * m.price_eur_mwh * 0.25)
                + (m.discharge_mw * m.price_eur_mwh * 0.25)
                - (m.charge_mw * m.price_eur_mwh * 0.25);
        }, 0);

// Playback loop
useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
        setMtuIndex(prev => {
            if (prev >= 95) {
                setDayIndex(d => {
                    if (d >= data.days.length - 1) {
                        setIsPlaying(false);
                        return d;
                    }
                    return d + 1;
                });
                return 0;
            }
            return prev + 1;
        });
    }, 100 / speed);
    return () => clearInterval(interval);
}, [isPlaying, speed]);
```

### Playback speeds
- 1× = 1 quarter-hour per 100ms → one full day in ~10 seconds
- 4× = one day in ~2.5 seconds
- 10× = one day in ~1 second
- Also allow "skip to next day" and day-level scrubbing

---

## Step 4 — Visual Components

### 4.1 WindFarm.jsx (SVG animated turbines)

Draw 3–5 stylised wind turbines as SVG. Each has a rotating blade group.

- **Rotation speed** ∝ `currentMtu.wind_speed_100m` (map ~0–30 m/s to 0–360°/s)
- **Blade count**: 3 per turbine (standard)
- Use CSS `animation` with `animation-duration` dynamically set, or `framer-motion`'s `animate` with `rotate` and `transition.duration`

```jsx
// Simplified concept
const rotationDuration = Math.max(0.3, 10 / (windSpeed + 0.1)); // faster wind = faster spin
<g style={{ 
    transformOrigin: 'center',
    animation: `spin ${rotationDuration}s linear infinite` 
}}>
    <line x1="0" y1="0" x2="0" y2="-40" stroke="white" strokeWidth="3" />
    <line x1="0" y1="0" x2="35" y2="20" stroke="white" strokeWidth="3" />
    <line x1="0" y1="0" x2="-35" y2="20" stroke="white" strokeWidth="3" />
</g>
```

**Sky colour**: Transition between dark (night) and light (day) based on hour. Use a gradient that shifts:
- Hours 0–5: dark navy (#0a1628)
- Hours 6–8: sunrise gradient
- Hours 9–17: sky blue (#87CEEB), darker when cloud_cover_pct is high
- Hours 18–20: sunset gradient
- Hours 21–23: dark navy

**Clouds**: Optionally render some SVG cloud shapes with opacity ∝ cloud_cover_pct.

### 4.2 BatteryGauge.jsx

A vertical or horizontal battery icon with a fill level.

- **Fill %** = `(currentMtu.soc_mwh - 5) / (45 - 5) × 100` (usable range: 5–45 MWh)
- **Fill colour**: gradient from red (< 20%) → yellow (20–60%) → green (> 60%)
- **Charging indicator**: When `charge_mw > 0`, show a ⚡ icon or pulsing border (green/blue)
- **Discharging indicator**: When `discharge_mw > 0`, show outflow arrows or pulsing border (orange)
- **Label**: Show SoC in MWh (e.g., "32.5 / 50 MWh")
- **Charge/discharge rate**: Show current MW (e.g., "+25 MW" or "−25 MW")

Animate fill transitions with framer-motion `animate={{ height: fillPct + '%' }}`.

### 4.3 PriceTicker.jsx

Large, prominent display of the current quarter-hourly price.

- **Colour coding**:
  - Deep red + flash: price < 0 (negative!)
  - Orange: price < 20 €/MWh
  - White/neutral: 20–100 €/MWh
  - Green: > 100 €/MWh
- **Format**: "€45.20/MWh"
- **Subtext**: "Day-ahead price" and the current timestamp (e.g., "14 Mar 2025, 13:45")
- Optional sparkline of today's 96 prices up to current MTU (use recharts `AreaChart`, tiny)

### 4.4 RevenuePanel.jsx

Running totals with animated number counting (use framer-motion or a counter library).

Display:
- **Wind Revenue** (cumulative): rolling total in €
- **BESS P&L** (cumulative): rolling total in € (green if positive, red if negative)
- **Total Portfolio**: sum of above
- **Today's P&L**: just today so far

Use `Intl.NumberFormat('de-DE', { style: 'currency', currency: 'EUR' })` for formatting.

### 4.5 WeatherOverlay.jsx

Small info panel showing current conditions:
- Wind speed: `currentMtu.wind_speed_100m` m/s (with a small wind icon)
- Temperature: `currentMtu.temperature_c` °C
- Cloud cover: `currentMtu.cloud_cover_pct` % (or just an icon)
- Wind generation: `currentMtu.wind_generation_mw` MW out of 50 MW capacity

### 4.6 TimelineControls.jsx

Bottom bar with:
- Play/Pause button
- Speed selector: 1×, 4×, 10×, 50× (50× = skip through fast)
- Day scrubber: slider from day 0 to day N, with date labels
- Within-day scrubber: 0–95 (quarter-hours)
- Current date/time display: "2025-03-14 13:45 CET"
- "Skip to next day" / "Previous day" buttons

### 4.7 DayChart.jsx (optional, nice-to-have)

A small recharts `ComposedChart` showing today's full price profile + charge/discharge overlay:
- Area fill for price (coloured by sign: blue positive, red negative)
- Bar overlay for charge (downward, blue) and discharge (upward, green) volumes
- Vertical line marker at current MTU

---

## Step 5 — Layout (Scene.jsx / App.jsx)

Suggested layout (landscape/desktop optimised):

```
┌─────────────────────────────────────────────────────┐
│  NegaPriceNL — BESS Simulation            [date]    │
├────────────────────────┬────────────────────────────┤
│                        │  Price Ticker              │
│   Wind Farm Scene      │  Battery Gauge             │
│   (SVG, large)         │  Weather Info              │
│                        │  Revenue Counters          │
│                        ├────────────────────────────┤
│                        │  Day Price Chart           │
├────────────────────────┴────────────────────────────┤
│  ◀ ▶ ⏸  [1× 4× 10×]  ═══════●═══════  Day 74/348  │
└─────────────────────────────────────────────────────┘
```

Use a dark theme (dark navy/charcoal background) — it looks better for this kind of dashboard and matches the energy trading aesthetic.

---

## Step 6 — Polish & Deploy

### 6.1 Visual polish
- Add a subtle "NordWind BV" branding in the scene (the fictional BRP name)
- Smooth all transitions (price colour changes, battery fill, sky colour)
- Add a subtle particle effect for wind (small dots blowing across the scene)
- Flash/pulse effect when negative prices hit

### 6.2 README.md
Write a short README explaining:
- What this visualises
- How to run it (`npm install && npm run dev`)
- How to regenerate the data (`python scripts/export_viz_data.py`)
- That it's part of the NegaPriceNL project

### 6.3 GitHub Pages deploy (optional)
Add a `vite.config.js` base path and a GitHub Actions workflow:
```yaml
# .github/workflows/deploy-viz.yml
# Build viz/ and deploy to gh-pages
```

---

## Key Design Decisions

1. **Static JSON, not live computation**: The React app just replays pre-computed data. No Python in the browser. This keeps things simple and fast.

2. **MTU-level granularity**: Daily summaries aren't enough for a smooth animation. We need the 96 quarter-hourly data points per day.

3. **SVG over Canvas/Three.js**: For stylised turbines and a battery gauge, SVG with framer-motion is more maintainable and looks clean. No need for WebGL complexity.

4. **Dark theme**: Matches energy trading dashboards and makes the colour-coded prices pop.

5. **Cumulative revenue as the "score"**: This is the number that tells the story — watching it climb (especially during negative price episodes where the BESS earns by charging cheap) makes the value proposition visceral.

---

## What NOT to Build

- No backend / API — everything is static
- No user authentication
- No real-time data fetching
- No mobile-first design (desktop dashboard is fine)
- No 3D rendering — SVG is sufficient
- Don't modify any existing simulation code — just add the export script and the viz folder
