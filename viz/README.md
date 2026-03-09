# NegaPriceNL BESS Simulation Visualization

Interactive browser-based visualization that replays the BESS (Battery Energy Storage System) arbitrage simulation day-by-day.

## Features

- Animated wind farm with turbines spinning based on wind speed
- Dynamic sky colors based on time of day
- Real-time battery state-of-charge gauge
- Price ticker with negative price alerts
- Cumulative revenue counter
- Day price profile chart with BESS activity overlay
- Playback controls with variable speed (1x, 4x, 10x, 50x)

## Getting Started

### Prerequisites

- Node.js 18+
- npm

### Installation

```bash
cd viz
npm install
```

### Development

```bash
npm run dev
```

Open <http://localhost:5173> in your browser.

### Production Build

```bash
npm run build
```

Output will be in `dist/`.

## Data

The visualization uses pre-computed simulation data:

- `src/data/simulation_output_sample.json` - 7-day sample (May 12-18, 2025)
- `src/data/simulation_output.json` - Full year (generated separately)

### Regenerating Data

From the project root:

```bash
# Sample only (fast)
python scripts/export_viz_data.py --sample-only

# Full year (slower)
python scripts/export_viz_data.py
```

## Tech Stack

- [Vite](https://vite.dev/) - Build tool
- [React](https://react.dev/) - UI framework
- [Tailwind CSS v4](https://tailwindcss.com/) - Styling
- [Framer Motion](https://www.framer.com/motion/) - Animations
- [Recharts](https://recharts.org/) - Charts
- [Lucide React](https://lucide.dev/) - Icons

## Architecture

```text
viz/
├── src/
│   ├── hooks/
│   │   └── useSimulation.js    # Playback state machine
│   ├── components/
│   │   ├── Scene.jsx           # Main layout
│   │   ├── WindFarm.jsx        # SVG turbines
│   │   ├── BatteryGauge.jsx    # SoC visualization
│   │   ├── PriceTicker.jsx     # Current price display
│   │   ├── RevenuePanel.jsx    # Cumulative revenue
│   │   ├── WeatherOverlay.jsx  # Weather conditions
│   │   ├── DayChart.jsx        # Price profile
│   │   └── TimelineControls.jsx # Playback controls
│   └── data/
│       └── simulation_output_sample.json
└── public/
```

## Part of NegaPriceNL

This visualization is part of the NegaPriceNL project, which explores negative electricity price forecasting and BESS arbitrage strategies in the Dutch day-ahead market.

The fictional BRP (Balance Responsible Party) is called **NordWind BV**.
