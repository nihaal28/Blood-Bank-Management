"""
data_validator.py
-----------------
Checks that the generated dataset makes sense (10 sanity checks)
and produces 5 exploratory charts to help you understand the data visually.

Run AFTER data_generator.py.
Output: outputs/plots/*.png  and  outputs/reports/data_summary.csv
"""

import os, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")   # non-interactive backend (no pop-up windows)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import date

# Folder paths
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR   = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Holiday dates as pandas Timestamps (used to draw markers on charts)
HOLIDAYS_DT = pd.to_datetime([
    "2021-01-26","2021-03-29","2021-08-15","2021-10-02","2021-11-04","2021-12-25",
    "2022-01-26","2022-03-18","2022-08-15","2022-10-02","2022-10-24","2022-12-25",
    "2023-01-26","2023-03-08","2023-08-15","2023-10-02","2023-11-12","2023-12-25",
])


def load_data():
    """Load the CSV produced by data_generator.py."""
    return pd.read_csv(
        os.path.join(BASE_DIR, "data", "raw", "synthetic_demand.csv"),
        parse_dates=["date"]
    )


def _save(path):
    """Helper: tighten layout, save figure to disk, and close it to free memory."""
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# PART A — VALIDATION CHECKS
# ============================================================
def run_validations(df):
    """
    Run 10 automatic checks on the dataset.
    Each check either PASSES or FAILS — we want all 10 to PASS.

    The checks verify:
    - Data completeness (no missing values)
    - Logical constraints (stock can't be negative)
    - Correct date range and number of combinations
    - Expected real-world patterns (weekends quieter, dengue spike, etc.)
    - Discard rate in the 5–25% target range
    """
    results = []

    def chk(name, cond):
        """Print and record a single check result."""
        s = "PASS" if cond else "FAIL"
        results.append((name, s))
        print(f"  [{s}] {name}")

    chk("1. No null values",         df.isnull().sum().sum() == 0)
    chk("2. demand >= 0",            (df["demand"] >= 0).all())
    chk("3. stock_on_hand >= 0",     (df["stock_on_hand"] >= 0).all())
    chk("4. units_expired >= 0",     (df["units_expired"] >= 0).all())
    chk("5. Date range 2021..2023",
        df["date"].min().date() == date(2021,1,1) and df["date"].max().date() == date(2023,12,31))
    chk("6. Exactly 6 combos",       len(df.groupby(["hospital","blood_group"])) == 6)

    # Check 7: weekends should have LOWER demand than weekdays (fewer elective surgeries)
    chk("7. Weekend < Weekday demand",
        df[df["is_weekend"]==1]["demand"].mean() < df[df["is_weekend"]==0]["demand"].mean())

    # Check 8: public holidays should also have lower demand
    chk("8. Holiday < Non-holiday demand",
        df[df["is_holiday"]==1]["demand"].mean() < df[df["is_holiday"]==0]["demand"].mean())

    # Check 9: during dengue season (Jul-Oct) O+ demand should be higher
    chk("9. Dengue O+ > Non-dengue O+",
        df[(df["is_dengue_season"]==1) & (df["blood_group"]=="O_positive")]["demand"].mean() >
        df[(df["is_dengue_season"]==0) & (df["blood_group"]=="O_positive")]["demand"].mean())

    # Check 10: discard rate = expired / collected.  Target: 5–25% (too low = understocked, too high = wasteful)
    dr = df["units_expired"].sum() / df["units_collected"].sum()
    chk("10. Discard rate 5-25%",    0.05 <= dr <= 0.25)

    passed = sum(s == "PASS" for _, s in results)
    print(f"\n  {passed}/{len(results)} checks passed")
    return results


# ============================================================
# PART B — EXPLORATORY CHARTS
# ============================================================
def plot_demand_timeseries(df):
    """
    Plot daily demand for all 6 hospital-blood_group combinations over 3 years.
    Orange shading = dengue season.  Red dashed lines = public holidays.
    This helps you visually confirm the seasonal patterns are in the data.
    """
    combos = [("Hospital_A","O_positive"), ("Hospital_A","O_negative"),
              ("Hospital_B","O_positive"), ("Hospital_B","O_negative"),
              ("Hospital_C","O_positive"), ("Hospital_C","O_negative")]

    fig, axes = plt.subplots(6, 1, figsize=(16, 20), sharex=True)  # one row per combo
    fig.suptitle("Daily Blood Demand Time Series (2021–2023)", fontsize=14, fontweight="bold")

    for ax, (h, bg) in zip(axes, combos):
        sub = df[(df["hospital"]==h) & (df["blood_group"]==bg)].sort_values("date")
        ax.plot(sub["date"], sub["demand"], lw=0.8, color="steelblue")
        ax.set_ylabel(f"{h}\n{bg}", fontsize=8)

        # Shade July-October each year to highlight dengue season
        for yr in [2021, 2022, 2023]:
            ax.axvspan(pd.Timestamp(f"{yr}-07-01"), pd.Timestamp(f"{yr}-10-31"), color="orange", alpha=0.15)

        # Mark each public holiday with a dashed red vertical line
        for h_ in HOLIDAYS_DT:
            ax.axvline(h_, color="red", lw=0.5, linestyle="--", alpha=0.6)

    # Add a legend on the last subplot
    axes[-1].legend(handles=[
        mpatches.Patch(color="orange", alpha=0.3, label="Dengue Season"),
        plt.Line2D([0], [0], color="red", lw=1, linestyle="--", label="Public Holiday")
    ], fontsize=8)
    axes[-1].tick_params(axis="x", rotation=30)
    _save(os.path.join(PLOTS_DIR, "demand_timeseries.png"))


def plot_weekly_seasonality(df):
    """
    Grouped bar chart showing average demand for each day of the week.
    Confirms the Monday surge and weekend dip we built into the generator.
    """
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    g = df.groupby(["hospital","day_of_week"])["demand"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    x, w = np.arange(7), 0.25  # 7 days, bar width 0.25

    for i, h in enumerate(sorted(df["hospital"].unique())):
        s = g[g["hospital"]==h].sort_values("day_of_week")
        ax.bar(x + i*w, s["demand"], w, label=h)

    ax.set_xticks(x + w); ax.set_xticklabels(day_names)
    ax.set(xlabel="Day of Week", ylabel="Mean Daily Demand", title="Weekly Demand Seasonality by Hospital")
    ax.legend()
    _save(os.path.join(PLOTS_DIR, "weekly_seasonality.png"))


def plot_monthly_pattern(df):
    """
    Line chart of average demand per month, separated by blood group.
    The July-October spike for O_positive confirms the dengue season effect.
    """
    g = df.groupby(["blood_group", "month"])["demand"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))

    for bg in df["blood_group"].unique():
        s = g[g["blood_group"]==bg]
        ax.plot(s["month"], s["demand"], marker="o", label=bg)

    ax.axvspan(7, 10, color="orange", alpha=0.15, label="Dengue Season")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set(xlabel="Month", ylabel="Mean Daily Demand", title="Monthly Demand Pattern by Blood Group")
    ax.legend()
    _save(os.path.join(PLOTS_DIR, "monthly_pattern.png"))


def plot_wastage_analysis(df):
    """
    Stacked bar chart per hospital showing monthly collected vs used vs expired.
    The gap between 'collected' and 'used' bars is wastage — we want this small
    but not zero (zero would mean we're constantly running out of stock).
    """
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")  # group by year-month

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Monthly Wastage Analysis by Hospital", fontsize=13, fontweight="bold")

    for ax, h in zip(axes, sorted(df["hospital"].unique())):
        s = df[df["hospital"]==h].groupby("ym")[
            ["units_collected", "units_used", "units_expired"]
        ].sum().reset_index()
        s["ym_str"] = s["ym"].astype(str)
        x = range(len(s))

        ax.bar(x, s["units_collected"], label="Collected", color="steelblue", alpha=0.8)
        ax.bar(x, s["units_used"],      label="Used",      color="green",     alpha=0.8)
        ax.bar(x, s["units_expired"],   label="Expired",   color="red",       alpha=0.8)
        ax.set(title=h, ylabel="Units")
        # Only show every 3rd month label so the x-axis isn't crowded
        ax.set_xticks(list(x)[::3]); ax.set_xticklabels(s["ym_str"].iloc[::3], rotation=45, fontsize=7)
        ax.legend(fontsize=8)

    _save(os.path.join(PLOTS_DIR, "wastage_analysis.png"))


def plot_shortage_heatmap(df):
    """
    Heatmap of total shortage events by hospital (rows) and month (columns).
    Dark red = many shortages in that month.  Ideally most cells should be 0 or near 0.
    """
    pv = df.groupby(["hospital", "month"])["shortage"].sum().reset_index()
    pw = pv.pivot(index="hospital", columns="month", values="shortage")
    pw.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pw, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Shortage Events Heatmap (Hospital × Month)")
    _save(os.path.join(PLOTS_DIR, "shortage_heatmap.png"))


# ============================================================
# PART C — SUMMARY STATISTICS
# ============================================================
def save_summary_stats(df):
    """
    Compute key summary numbers per hospital-blood_group combination
    and save as a CSV for easy reference.
    Discard rate = expired / collected × 100 — our most important quality metric.
    """
    rows = []
    for (h, bg), g in df.groupby(["hospital", "blood_group"]):
        tc = g["units_collected"].sum()  # total collected over 3 years
        rows.append({
            "hospital":           h,
            "blood_group":        bg,
            "total_demand":       g["demand"].sum(),
            "total_collected":    tc,
            "total_used":         g["units_used"].sum(),
            "total_expired":      g["units_expired"].sum(),
            "discard_rate_pct":   round(100 * g["units_expired"].sum() / tc if tc > 0 else 0, 2),
            "total_shortages":    g["shortage"].sum(),
            "mean_daily_demand":  round(g["demand"].mean(), 2),
        })

    summary = pd.DataFrame(rows)
    path = os.path.join(REPORTS_DIR, "data_summary.csv")
    summary.to_csv(path, index=False)
    print(f"\n  Summary saved to: {path}")
    print(summary.to_string(index=False))
    return summary


# ----------------------------------------------------------------------
# Run all three parts when executed directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    print("PART A — Validation Checks:")
    run_validations(df)

    print("\nPART B — Generating Plots:")
    plot_demand_timeseries(df)
    plot_weekly_seasonality(df)
    plot_monthly_pattern(df)
    plot_wastage_analysis(df)
    plot_shortage_heatmap(df)

    print("\nPART C — Summary Statistics:")
    save_summary_stats(df)

    print("\nDone.")
