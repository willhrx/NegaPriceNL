"""
Calendar Heatmap: 2025 Negative Price Events & Model v4 Predictions

Creates a publication-quality GitHub-style calendar heatmap showing:
- Days with negative electricity prices in 2025
- V4 model prediction outcomes (correctly predicted, partially detected, missed, false alarm)
- Color intensity scaled by number of negative hours per day

Output: outputs/figures/calendar_heatmap_2025.png

Usage:
    py scripts/create_calendar_heatmap.py
"""

import sys
from pathlib import Path
import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, TEST_START_DATE
)
from src.evaluation.benchmark_strategies import MLStrategy

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_VERSION = "v4"
YEAR = 2025

# Category definitions
CAT_NO_EVENT = 0
CAT_ALL_DETECTED = 1
CAT_PARTIAL = 2
CAT_MISSED = 3
CAT_FALSE_ALARM = 4

CATEGORY_BASE_COLORS = {
    CAT_NO_EVENT: '#EBEDF0',
    CAT_ALL_DETECTED: '#27AE60',
    CAT_PARTIAL: "#FBFF00",
    CAT_MISSED: '#E74C3C',
    CAT_FALSE_ALARM: "#2270E6",
}

CATEGORY_LABELS = {
    CAT_NO_EVENT: 'No negative prices',
    CAT_ALL_DETECTED: 'Correctly predicted',
    CAT_PARTIAL: 'Partially detected',
    CAT_MISSED: 'Missed by model',
    CAT_FALSE_ALARM: 'False alarm',
}


def load_data_and_predict():
    """Load 2025 test data and generate v4 model predictions."""
    input_path = PROCESSED_DATA_DIR / "feature_matrix.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {input_path}")

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Filter to 2025
    test_start = pd.Timestamp(TEST_START_DATE, tz='UTC')
    test_df = df[df.index >= test_start].copy()
    logger.info(f"  Test records: {len(test_df):,} ({test_df.index.min()} to {test_df.index.max()})")

    # Load model and predict
    model_path = MODELS_DIR / f"gradient_boost_negative_price_{MODEL_VERSION}.pkl"
    threshold_path = MODELS_DIR / f"optimal_threshold_{MODEL_VERSION}.pkl"
    feature_cols_path = MODELS_DIR / f"feature_columns_{MODEL_VERSION}.pkl"

    ml_strategy = MLStrategy(
        model_path=model_path,
        threshold_path=threshold_path,
        feature_columns_path=feature_cols_path,
        model_version=MODEL_VERSION,
    )
    predictions = ml_strategy.predict(test_df)
    test_df['predicted'] = predictions

    neg_hours = int(test_df['is_negative_price'].sum())
    pred_pos = int(predictions.sum())
    logger.info(f"  Actual negative hours: {neg_hours}")
    logger.info(f"  Predicted negative hours: {pred_pos}")

    return test_df


def compute_daily_categories(test_df):
    """Resample to hourly, then aggregate to daily prediction categories."""
    df = test_df[['is_negative_price', 'predicted']].copy()

    # Resample to hourly (handles mixed 15-min / hourly granularity)
    hourly = df.resample('h').max().dropna()

    # Per-hour confusion matrix flags
    actual = hourly['is_negative_price']
    pred = hourly['predicted']
    hourly['tp'] = ((actual == 1) & (pred == 1)).astype(int)
    hourly['fp'] = ((actual == 0) & (pred == 1)).astype(int)
    hourly['fn'] = ((actual == 1) & (pred == 0)).astype(int)

    # Daily aggregation
    daily = hourly.resample('D').agg({
        'is_negative_price': 'sum',
        'tp': 'sum',
        'fp': 'sum',
        'fn': 'sum',
    })
    daily.columns = ['neg_hours', 'tp_hours', 'fp_hours', 'fn_hours']

    # Assign categories
    daily['category'] = CAT_NO_EVENT
    daily.loc[(daily['neg_hours'] > 0) & (daily['fn_hours'] == 0) & (daily['fp_hours'] == 0), 'category'] = CAT_ALL_DETECTED
    daily.loc[(daily['neg_hours'] > 0) & (daily['fn_hours'] == 0) & (daily['fp_hours'] > 0), 'category'] = CAT_ALL_DETECTED
    daily.loc[(daily['neg_hours'] > 0) & (daily['tp_hours'] > 0) & (daily['fn_hours'] > 0), 'category'] = CAT_PARTIAL
    daily.loc[(daily['neg_hours'] > 0) & (daily['tp_hours'] == 0), 'category'] = CAT_MISSED
    daily.loc[(daily['neg_hours'] == 0) & (daily['fp_hours'] > 0), 'category'] = CAT_FALSE_ALARM

    logger.info(f"  Daily category counts:")
    for cat, label in CATEGORY_LABELS.items():
        count = (daily['category'] == cat).sum()
        logger.info(f"    {label}: {count} days")

    return daily


def get_cell_color(category, neg_hours, max_neg_hours):
    """Get cell color with intensity variation based on severity."""
    base_hex = CATEGORY_BASE_COLORS[category]
    if category == CAT_NO_EVENT:
        return base_hex

    # Scale intensity: more negative hours → more saturated
    intensity = min(neg_hours / max(max_neg_hours * 0.6, 1), 1.0)
    intensity = 0.30 + 0.70 * intensity  # range 30%–100% saturation

    base_rgb = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    blended = white + (base_rgb - white) * intensity
    return tuple(blended.clip(0, 1))


def build_calendar_grid(daily, year=2025):
    """Build a full-year calendar grid indexed by (week_offset, weekday)."""
    all_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D', tz='UTC')
    grid = pd.DataFrame(index=all_dates)
    grid = grid.join(daily[['category', 'neg_hours', 'tp_hours', 'fp_hours', 'fn_hours']])
    grid['category'] = grid['category'].fillna(CAT_NO_EVENT).astype(int)
    grid['neg_hours'] = grid['neg_hours'].fillna(0)

    # Continuous week offset (avoids ISO week boundary issues)
    first_day = grid.index[0]
    grid['weekday'] = grid.index.weekday  # 0=Mon, 6=Sun
    grid['week_offset'] = ((grid.index - first_day).days + first_day.weekday()) // 7

    return grid


def create_calendar_heatmap(grid, daily, year=2025):
    """Render the GitHub-style calendar heatmap."""
    max_neg = daily['neg_hours'].max() if len(daily) > 0 else 1
    max_week = grid['week_offset'].max()

    # --- Figure layout ---
    fig = plt.figure(figsize=(18, 5.5), facecolor='white')
    gs = GridSpec(
        2, 2,
        width_ratios=[5.5, 1],
        height_ratios=[1, 0.06],
        wspace=0.04,
        hspace=0.25,
        left=0.04, right=0.96, top=0.85, bottom=0.08,
    )
    ax_cal = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])

    # --- Draw calendar cells ---
    cell_size = 0.82
    corner_radius = 0.12
    for _, row in grid.iterrows():
        week = row['week_offset']
        weekday = row['weekday']
        cat = int(row['category'])
        neg_h = row['neg_hours']
        color = get_cell_color(cat, neg_h, max_neg)

        rect = FancyBboxPatch(
            (week - cell_size / 2, weekday - cell_size / 2),
            cell_size, cell_size,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            facecolor=color,
            edgecolor='white',
            linewidth=0.6,
        )
        ax_cal.add_patch(rect)

    # --- Month labels ---
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in range(1, 13):
        month_start = pd.Timestamp(f'{year}-{month:02d}-01', tz='UTC')
        if month_start in grid.index:
            week_pos = grid.loc[month_start, 'week_offset']
            ax_cal.text(
                week_pos, -0.9, month_names[month - 1],
                ha='left', va='center', fontsize=9.5,
                fontweight='medium', color='#444444',
                fontfamily='sans-serif',
            )

    # --- Weekday labels ---
    day_labels = ['Mon', '', 'Wed', '', 'Fri', '', 'Sun']
    for i, label in enumerate(day_labels):
        if label:
            ax_cal.text(
                -1.3, i, label,
                ha='right', va='center', fontsize=8.5,
                color='#666666', fontfamily='sans-serif',
            )

    ax_cal.set_xlim(-2.5, max_week + 1.2)
    ax_cal.set_ylim(-1.8, 7.0)
    ax_cal.invert_yaxis()
    ax_cal.set_aspect('equal')
    ax_cal.axis('off')

    # --- Title & subtitle ---
    fig.text(
        0.04, 0.95,
        'NegaPriceNL 2025: Negative Price Events & Model v4 Predictions',
        fontsize=16, fontweight='bold', color='#1a1a2e',
        fontfamily='sans-serif', va='top',
    )
    fig.text(
        0.04, 0.905,
        'Each cell is one day. Color shows prediction outcome; intensity shows severity (number of negative-price hours).',
        fontsize=9.5, color='#666666', fontfamily='sans-serif', va='top',
    )

    # --- Legend ---
    ax_legend.axis('off')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    legend_items = [
        (CAT_ALL_DETECTED, CATEGORY_LABELS[CAT_ALL_DETECTED]),
        (CAT_PARTIAL, CATEGORY_LABELS[CAT_PARTIAL]),
        (CAT_MISSED, CATEGORY_LABELS[CAT_MISSED]),
        (CAT_FALSE_ALARM, CATEGORY_LABELS[CAT_FALSE_ALARM]),
        (CAT_NO_EVENT, CATEGORY_LABELS[CAT_NO_EVENT]),
    ]

    y_start = 0.92
    spacing = 0.075
    box_w = 0.08
    box_h = 0.045

    ax_legend.text(0.0, 0.99, 'Legend', fontsize=11, fontweight='bold',
                   color='#1a1a2e', va='top', transform=ax_legend.transAxes,
                   fontfamily='sans-serif')

    for i, (cat, label) in enumerate(legend_items):
        y = y_start - i * spacing
        rect = FancyBboxPatch(
            (0.02, y - box_h / 2), box_w, box_h,
            boxstyle=f"round,pad=0,rounding_size=0.008",
            facecolor=CATEGORY_BASE_COLORS[cat],
            edgecolor='#cccccc',
            linewidth=0.5,
            transform=ax_legend.transAxes,
        )
        ax_legend.add_patch(rect)
        ax_legend.text(
            0.02 + box_w + 0.03, y, label,
            fontsize=8.5, va='center', color='#333333',
            transform=ax_legend.transAxes, fontfamily='sans-serif',
        )

    # --- Stats box ---
    neg_days = int((daily['neg_hours'] > 0).sum())
    perfect_days = int((daily['category'] == CAT_ALL_DETECTED).sum())
    partial_days = int((daily['category'] == CAT_PARTIAL).sum())
    missed_days = int((daily['category'] == CAT_MISSED).sum())
    fa_days = int((daily['category'] == CAT_FALSE_ALARM).sum())

    # Hourly metrics
    total_neg = daily['neg_hours'].sum()
    total_tp = daily['tp_hours'].sum()
    total_fp = daily['fp_hours'].sum()
    total_fn = daily['fn_hours'].sum()
    hourly_recall = total_tp / max(total_tp + total_fn, 1)
    hourly_precision = total_tp / max(total_tp + total_fp, 1)
    hourly_f1 = 2 * hourly_precision * hourly_recall / max(hourly_precision + hourly_recall, 1e-9)

    stats_y = y_start - len(legend_items) * spacing - 0.06
    stats_lines = [
        ('Summary', True),
        (f'{neg_days} days with negative prices', False),
        (f'{perfect_days} days fully detected', False),
        (f'{partial_days} days partially detected', False),
        (f'{missed_days} days missed', False),
        (f'{fa_days} days false alarm', False),
        ('', False),
        ('Hourly Performance', True),
        (f'Recall: {hourly_recall:.1%}', False),
        (f'Precision: {hourly_precision:.1%}', False),
        (f'F1 Score: {hourly_f1:.3f}', False),
    ]

    for j, (text, bold) in enumerate(stats_lines):
        ax_legend.text(
            0.02, stats_y - j * 0.052, text,
            fontsize=8.5 if not bold else 9.5,
            fontweight='bold' if bold else 'normal',
            color='#1a1a2e' if bold else '#555555',
            transform=ax_legend.transAxes,
            fontfamily='sans-serif', va='top',
        )

    # --- Monthly neg-hour bar chart at the bottom ---
    monthly = daily.groupby(daily.index.month).agg(
        neg_hours=('neg_hours', 'sum'),
        tp_hours=('tp_hours', 'sum'),
        fn_hours=('fn_hours', 'sum'),
    )
    months = np.arange(1, 13)
    # Ensure all months present
    monthly = monthly.reindex(months, fill_value=0)

    bar_width = 0.35
    x = np.arange(12)
    ax_bar.bar(x - bar_width / 2, monthly['tp_hours'], bar_width,
               color='#27AE60', label='Detected (TP)', edgecolor='white', linewidth=0.3)
    ax_bar.bar(x + bar_width / 2, monthly['fn_hours'], bar_width,
               color='#E74C3C', label='Missed (FN)', edgecolor='white', linewidth=0.3)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(month_names, fontsize=8, color='#666666', fontfamily='sans-serif')
    ax_bar.set_ylabel('Hours', fontsize=8, color='#666666', fontfamily='sans-serif')
    ax_bar.tick_params(axis='y', labelsize=7, colors='#888888')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('#cccccc')
    ax_bar.spines['bottom'].set_color('#cccccc')
    ax_bar.set_title('Monthly negative-price hours: detected vs missed',
                     fontsize=8.5, color='#555555', fontfamily='sans-serif', pad=3)
    ax_bar.legend(fontsize=7, loc='upper right', frameon=False)

    return fig


def main():
    logger.info("=" * 60)
    logger.info("CALENDAR HEATMAP: 2025 Negative Prices & v4 Predictions")
    logger.info("=" * 60)

    test_df = load_data_and_predict()
    daily = compute_daily_categories(test_df)
    grid = build_calendar_grid(daily, year=YEAR)
    fig = create_calendar_heatmap(grid, daily, year=YEAR)

    output_path = FIGURES_DIR / 'calendar_heatmap_2025.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved: {output_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
