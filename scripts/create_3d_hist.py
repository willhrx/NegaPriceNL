"""
3D Histogram: Negative Price Patterns by Month and Hour

Creates a 3D bar chart showing the percentage of hours with negative
electricity prices, broken down by month (y-axis) and hour of day (x-axis).
Same data as Visualization 2 in exploratory_analysis.py, rendered in 3D.

Output: outputs/figures/3d_negative_price_histogram.png

Usage:
    py scripts/create_3d_hist.py
"""

import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import ENTSOE_DATA_DIR, FIGURES_DIR

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("3D Histogram: Negative Price Patterns (Month x Hour)")
    print("=" * 60)

    # --- Load data (same pattern as exploratory_analysis.py) ---
    print("\n[*] Loading price data...")
    nl_prices = pd.read_csv(
        ENTSOE_DATA_DIR / "nl_day_ahead_prices_2019_2025.csv",
        index_col=0, parse_dates=True,
    )
    nl_prices.index = pd.to_datetime(nl_prices.index, utc=True)
    nl_prices['is_negative'] = (nl_prices['price'] < 0).astype(int)
    nl_prices['month'] = nl_prices.index.month
    nl_prices['hour'] = nl_prices.index.hour
    print(f"[+] Loaded {len(nl_prices):,} price records")

    # --- Compute pivot table ---
    negative_pct = nl_prices.pivot_table(
        index='month', columns='hour',
        values='is_negative', aggfunc='mean',
    ) * 100

    # --- Build 3D bar coordinates ---
    hours = negative_pct.columns.values        # 0–23
    months = negative_pct.index.values          # 1–12
    xpos, ypos = np.meshgrid(hours, months)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dz = negative_pct.values.flatten()

    bar_width = 0.7
    bar_depth = 0.7

    # --- Color mapping (height → RdYlGn_r like the original heatmap) ---
    cmap = cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=max(dz.max(), 1))
    colors = cmap(norm(dz))

    # --- Create figure ---
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(
        xpos - bar_width / 2,
        ypos - bar_depth / 2,
        zpos,
        bar_width, bar_depth, dz,
        color=colors, edgecolor='#444444', linewidth=0.15,
        alpha=0.92, zsort='average',
    )

    # --- Axes labels & ticks ---
    ax.set_xlabel('Hour of Day', fontsize=12, labelpad=12)
    ax.set_ylabel('Month', fontsize=12, labelpad=12)
    ax.set_zlabel('% Negative Price Hours', fontsize=12, labelpad=10)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], fontsize=8)

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(month_labels, fontsize=8)

    ax.set_zlim(0, max(dz.max() * 1.05, 1))

    # --- Viewing angle ---
    ax.view_init(elev=25, azim=-50)

    # --- Title ---
    ax.set_title(
        'Negative Price Patterns: % of Hours by Month and Hour of Day\n'
        '(Netherlands Day-Ahead Prices, 2019–2025)',
        fontsize=15, fontweight='bold', pad=20,
    )

    # --- Colorbar ---
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(dz)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label('% of Hours with Negative Prices', fontsize=10, labelpad=10)

    # --- Reduce pane clutter ---
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#dddddd')
    ax.yaxis.pane.set_edgecolor('#dddddd')
    ax.zaxis.pane.set_edgecolor('#dddddd')
    ax.grid(True, alpha=0.3)

    # --- Save ---
    output_path = FIGURES_DIR / '3d_negative_price_histogram.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n[+] Saved: {output_path}")
    print("[+] Done!")


if __name__ == "__main__":
    main()
