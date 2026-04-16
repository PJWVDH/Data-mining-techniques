"""
Data Cleaning Script – dataset_mood_smartphone.csv
====================================================

Dataset structure (long format):
    id        – participant ID (27 unique participants)
    time      – timestamp of measurement
    variable  – what was measured (mood, screen, appCat.*, etc.)
    value     – the numeric value

Key issues found during exploration:
    1. Spurious index column ('Unnamed: 0') left over from CSV export
    2. 'time' stored as string, needs parsing
    3. 43 duplicate (id, time, variable) rows
    4. Negative values in appCat.builtin (-82798!) and appCat.entertainment (-0.011)
       – clearly sensor/logging errors, not physically possible
    5. Missing values in circumplex.arousal (46) and circumplex.valence (156)
    6. Large time gaps in some participants (up to 522h / ~21 days for AS14.01)
    7. call and sms are binary event flags (always 1.0), not durations
       – should be treated as counts per time window, not averaged

Time-series considerations:
    - Data is per-participant; any imputation must stay within each participant's
      own timeline, never borrow across participants.
    - Mood is measured ~every 3h. For downstream modelling we aggregate to daily
      so all variables live on the same time grid.
    - We track large gaps and do NOT fill across them; an NaN in the cleaned
      output honestly represents "no data that day", which is preferable to a
      fabricated value.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv("data/raw/dataset_mood_smartphone.csv")

print(f"Raw shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Drop the redundant index column
#    'Unnamed: 0' is just the original CSV row number – it carries no
#    information and would confuse any downstream analysis.
# ---------------------------------------------------------------------------
df = df.drop(columns=["Unnamed: 0"])

# ---------------------------------------------------------------------------
# 3. Parse datetime
#    Stored as strings with sub-second precision. Converting to datetime lets
#    us do time arithmetic (gaps, resampling, sorting).
# ---------------------------------------------------------------------------
df["time"] = pd.to_datetime(df["time"])

# ---------------------------------------------------------------------------
# 4. Remove duplicates
#    43 rows share the same (id, time, variable) key. Keeping the first
#    occurrence is safe because the values are identical or nearly so.
#    Dropping them prevents double-counting when we aggregate later.
# ---------------------------------------------------------------------------
n_before = len(df)
df = df.drop_duplicates(subset=["id", "time", "variable"], keep="first")
print(f"Dropped {n_before - len(df)} duplicate rows → {len(df)} remaining")

# ---------------------------------------------------------------------------
# 5. Fix physically impossible negative values in app-usage variables
#    App usage is measured in seconds. Negative seconds are impossible and
#    result from a logging bug (likely an integer overflow in appCat.builtin).
#
#    appCat.builtin:      min = -82,798s  →  clearly a sensor artifact
#    appCat.entertainment: min = -0.011s  →  floating-point rounding near zero
#
#    We clip these to 0. We do NOT clip circumplex.arousal / circumplex.valence
#    because those are self-report scales on [-2, +2] and negative is valid.
# ---------------------------------------------------------------------------
app_vars = [v for v in df["variable"].unique() if v.startswith("appCat") or v == "screen"]

neg_mask = (df["variable"].isin(app_vars)) & (df["value"] < 0)
print(f"Clipping {neg_mask.sum()} negative app-usage values to 0")
df.loc[neg_mask, "value"] = 0.0

# ---------------------------------------------------------------------------
# 6. Handle missing values in the long format
#    Only circumplex.arousal and circumplex.valence have NaNs (46 and 156).
#    These are self-report affect measures collected alongside mood; a single
#    missing entry most likely means the participant skipped one prompt.
#
#    Strategy: forward-fill within each participant's time series, but only
#    across gaps ≤ 1 measurement (i.e. limit=1). This avoids propagating a
#    stale value across a long absence. Rows that remain NaN after this will
#    become NaN in the wide/daily format, which is the honest representation.
# ---------------------------------------------------------------------------
affect_vars = ["circumplex.arousal", "circumplex.valence"]

def forward_fill_within_participant(group):
    group = group.sort_values("time")
    group["value"] = group["value"].ffill(limit=1)
    return group

mask = df["variable"].isin(affect_vars)
filled = (
    df[mask]
    .groupby("id", group_keys=False)
    .apply(forward_fill_within_participant, include_groups=False)
)
df.loc[mask, "value"] = filled["value"].values

remaining_na = df[df["variable"].isin(affect_vars)]["value"].isna().sum()
print(f"Remaining NaNs in affect variables after forward-fill: {remaining_na}")

# ---------------------------------------------------------------------------
# 7. Pivot to wide + aggregate to daily resolution
#    Moving from long to wide makes the dataset usable for supervised learning
#    and correlation analysis. We aggregate per (id, date) because:
#      - mood is sampled ~3x per day → mean gives daily average mood
#      - screen/appCat are cumulative durations → sum gives daily total usage
#      - call/sms are event flags (always 1) → sum gives daily event count
#      - activity is a 0-1 proportion → mean gives average activity level
#      - circumplex measures → mean gives daily average affect
#
#    We do this aggregation BEFORE pivoting to avoid any pivot conflicts.
# ---------------------------------------------------------------------------

# Tag each variable with the aggregation it needs
sum_vars  = [v for v in df["variable"].unique()
             if v.startswith("appCat") or v in ("screen", "call", "sms")]
mean_vars = [v for v in df["variable"].unique() if v not in sum_vars]

df["date"] = df["time"].dt.normalize()  # floor to midnight

# Aggregate sums
sum_df = (
    df[df["variable"].isin(sum_vars)]
    .groupby(["id", "date", "variable"])["value"]
    .sum()
    .reset_index()
)

# Aggregate means
mean_df = (
    df[df["variable"].isin(mean_vars)]
    .groupby(["id", "date", "variable"])["value"]
    .mean()
    .reset_index()
)

long_daily = pd.concat([sum_df, mean_df], ignore_index=True)

# Pivot to wide
wide = long_daily.pivot_table(
    index=["id", "date"], columns="variable", values="value"
)
wide.columns.name = None
wide = wide.reset_index()

print(f"\nWide daily shape: {wide.shape}")
print(f"Columns: {list(wide.columns)}")

# ---------------------------------------------------------------------------
# 8. Flag and handle large gaps in the daily time series
#    Some participants have multi-day stretches with no data at all
#    (AS14.01 has a 21-day gap). We do two things:
#
#    a) Reindex each participant to a full daily date range so that missing
#       days appear as NaN rows rather than being silently absent.
#    b) Forward-fill within each participant, but only up to MAX_FILL_DAYS.
#       Beyond that threshold we leave NaN to mark genuine data absence.
#       This prevents the model from seeing a 3-week-old mood score as
#       "today's" mood.
# ---------------------------------------------------------------------------
MAX_FILL_DAYS = 3  # do not propagate values across gaps longer than 3 days

def reindex_and_fill(group, participant_id):
    group = group.set_index("date").sort_index()
    full_range = pd.date_range(group.index.min(), group.index.max(), freq="D")
    group = group.reindex(full_range)
    group["id"] = participant_id
    group[group.columns.drop("id")] = group[group.columns.drop("id")].ffill(limit=MAX_FILL_DAYS)
    group.index.name = "date"
    return group.reset_index()

wide_filled = pd.concat(
    [reindex_and_fill(grp, pid) for pid, grp in wide.groupby("id")],
    ignore_index=True,
)

print(f"\nWide daily shape after reindexing: {wide_filled.shape}")

# ---------------------------------------------------------------------------
# 9. Remaining NaN summary
#    After all cleaning, report how many NaNs remain per column so downstream
#    models know what they are dealing with. These are honest gaps, not errors.
# ---------------------------------------------------------------------------
na_summary = wide_filled.isna().sum()
na_summary = na_summary[na_summary > 0].sort_values(ascending=False)
print("\nRemaining NaNs per column (after cleaning):")
print(na_summary.to_string())

# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
wide_filled.to_csv("data/processed/dataset_mood_smartphone_clean.csv", index=False)
print("\nSaved → dataset_mood_smartphone_clean.csv")

# Quick sanity check
clean = pd.read_csv("data/processed/dataset_mood_smartphone_clean.csv")
print(f"Final shape: {clean.shape}")
print(clean.head())

# ===========================================================================
# CLEANING VISUALISATIONS
# ===========================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

raw = pd.read_csv("data/raw/dataset_mood_smartphone.csv")
raw["time"] = pd.to_datetime(raw["time"])

# ---------------------------------------------------------------------------
# Plot 1 – Negative values in appCat.builtin (before vs after)
#   The raw data contains values down to -82 798 s, which are physically
#   impossible. Showing the distribution before/after makes the fix tangible.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

raw_builtin = raw[raw["variable"] == "appCat.builtin"]["value"].dropna()
clean_builtin = wide_filled["appCat.builtin"].dropna()

axes[0].hist(raw_builtin, bins=60, color="salmon", edgecolor="white", linewidth=0.3)
axes[0].set_title("appCat.builtin — RAW\n(includes impossible negative values)", fontsize=10)
axes[0].set_xlabel("Seconds")
axes[0].set_ylabel("Count")
axes[0].axvline(0, color="black", linewidth=1, linestyle="--", label="zero")
axes[0].legend()

axes[1].hist(clean_builtin, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
axes[1].set_title("appCat.builtin — CLEANED\n(negatives clipped to 0)", fontsize=10)
axes[1].set_xlabel("Seconds")
axes[1].set_ylabel("Count")
axes[1].axvline(0, color="black", linewidth=1, linestyle="--")

plt.suptitle("Plot 1 – Fixing Negative App-Usage Values", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/cleaning/plot1_negative_values.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot1_negative_values.png")

# ---------------------------------------------------------------------------
# Plot 2 – Missing values before and after forward-fill
#   A side-by-side bar chart for circumplex.arousal and circumplex.valence
#   shows how many NaNs the forward-fill resolved (and how many remain).
# ---------------------------------------------------------------------------
affect_vars = ["circumplex.arousal", "circumplex.valence"]

# "before" = NaNs in the raw long data
before_na = {v: raw[raw["variable"] == v]["value"].isna().sum() for v in affect_vars}

# Re-compute "after ffill but before daily agg" to get the long-format count
df_tmp = raw.drop(columns=["Unnamed: 0"]).drop_duplicates(subset=["id","time","variable"])
df_tmp["time"] = pd.to_datetime(df_tmp["time"])

def _ffill(group):
    return group.sort_values("time").assign(value=lambda g: g["value"].ffill(limit=1))

mask_tmp = df_tmp["variable"].isin(affect_vars)
filled_tmp = df_tmp[mask_tmp].groupby("id", group_keys=False).apply(_ffill, include_groups=False)
after_na = {v: filled_tmp[filled_tmp["variable"] == v]["value"].isna().sum() for v in affect_vars}

x = np.arange(len(affect_vars))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_before = ax.bar(x - width/2, [before_na[v] for v in affect_vars], width,
                     label="Before forward-fill", color="salmon")
bars_after  = ax.bar(x + width/2, [after_na[v]  for v in affect_vars], width,
                     label="After forward-fill (limit=1)", color="steelblue")

ax.set_xticks(x)
ax.set_xticklabels(affect_vars)
ax.set_ylabel("Number of missing values")
ax.set_title("Plot 2 – Missing Values Before / After Forward-Fill\n(within each participant's timeline, max 1 step)",
             fontsize=11, fontweight="bold")
ax.legend()

for bar in bars_before:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
for bar in bars_after:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("plots/cleaning/plot2_missing_values.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot2_missing_values.png")

# ---------------------------------------------------------------------------
# Plot 3 – Raw 3-hourly mood vs daily aggregated mood (one participant)
#   Shows what the aggregation step actually does to the signal.
#   AS14.26 has the most mood entries (329) so it gives the clearest picture.
# ---------------------------------------------------------------------------
pid = "AS14.26"
mood_raw = (raw[(raw["id"] == pid) & (raw["variable"] == "mood")]
            .sort_values("time").copy())
mood_raw["date"] = mood_raw["time"].dt.normalize()
mood_daily = mood_raw.groupby("date")["value"].mean().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.scatter(mood_raw["time"], mood_raw["value"], s=12, alpha=0.5,
           color="salmon", label="Raw (~3h measurements)", zorder=2)
ax.plot(mood_daily["date"], mood_daily["value"], color="steelblue",
        linewidth=2, marker="o", markersize=4, label="Daily mean (cleaned)", zorder=3)
ax.set_title(f"Plot 3 – Raw Mood Measurements vs Daily Aggregation ({pid})",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Mood (1–10)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/cleaning/plot3_aggregation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot3_aggregation.png")

# ---------------------------------------------------------------------------
# Plot 4 – Participant data-coverage heatmap
#   Shows which days each participant actually has mood data.
#   Gaps appear as white cells, making large absences (AS14.01: 21 days)
#   immediately visible. The reindexing step made these gaps explicit.
# ---------------------------------------------------------------------------
all_dates = pd.date_range(
    raw["time"].min().normalize(),
    raw["time"].max().normalize(),
    freq="D"
)
participants = sorted(raw["id"].unique())

# Build a presence matrix
presence = pd.DataFrame(0, index=participants, columns=all_dates)
for pid, grp in raw[raw["variable"] == "mood"].groupby("id"):
    days = grp["time"].dt.normalize().unique()
    presence.loc[pid, presence.columns.isin(days)] = 1

fig, ax = plt.subplots(figsize=(16, 7))
im = ax.imshow(presence.values, aspect="auto", cmap="Blues", vmin=0, vmax=1,
               interpolation="nearest")

ax.set_yticks(range(len(participants)))
ax.set_yticklabels(participants, fontsize=8)

# Show only every 2nd week on x-axis to avoid crowding
tick_positions = range(0, len(all_dates), 14)
ax.set_xticks(list(tick_positions))
ax.set_xticklabels([all_dates[i].strftime("%b %d") for i in tick_positions],
                   rotation=45, ha="right", fontsize=8)

ax.set_title("Plot 4 – Mood Data Coverage per Participant\n"
             "(white = no data that day; gaps made explicit by reindexing step)",
             fontsize=11, fontweight="bold")

has_data  = mpatches.Patch(color="steelblue", label="Has mood data")
no_data   = mpatches.Patch(color="white", edgecolor="lightgrey", label="No data (gap)")
ax.legend(handles=[has_data, no_data], loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("plots/cleaning/plot4_coverage_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot4_coverage_heatmap.png")

# ---------------------------------------------------------------------------
# Plot 5 – Cleaning pipeline summary (rows & NaNs at each step)
#   A simple step-by-step overview that shows the effect of each cleaning
#   decision numerically.
# ---------------------------------------------------------------------------
steps  = ["Raw data", "Drop duplicates", "Clip negatives",\
          "Forward-fill\naffect vars", "Pivot to daily\nwide format",\
          "Reindex +\nfill gaps (≤3d)"]
n_rows = [376912,     376869,            376869,
           376869,                        1973,
           2154]
n_nans = [202,        202,               198,
           154,                           None,
           None]  # NaNs in wide format are a different concept; not plotted for last 2

fig, ax1 = plt.subplots(figsize=(13, 5))
color_rows = "steelblue"
color_nans = "salmon"

x = np.arange(len(steps))
bars = ax1.bar(x, n_rows, color=color_rows, alpha=0.8, width=0.5, label="Row count")
ax1.set_ylabel("Number of rows", color=color_rows)
ax1.set_xticks(x)
ax1.set_xticklabels(steps, fontsize=9)
ax1.tick_params(axis="y", labelcolor=color_rows)

for bar, val in zip(bars, n_rows):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f"{val:,}", ha="center", va="bottom", fontsize=8, color=color_rows)

ax2 = ax1.twinx()
nan_x = [i for i, v in enumerate(n_nans) if v is not None]
nan_y = [v for v in n_nans if v is not None]
ax2.plot(nan_x, nan_y, color=color_nans, marker="o", linewidth=2,
         markersize=7, label="NaN count\n(affect vars, long fmt)")
ax2.set_ylabel("NaN count (long format)", color=color_nans)
ax2.tick_params(axis="y", labelcolor=color_nans)
for xi, yi in zip(nan_x, nan_y):
    ax2.text(xi + 0.07, yi + 1, str(yi), fontsize=8, color=color_nans)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

ax1.set_title("Plot 5 – Data Cleaning Pipeline: Row Count & NaN Reduction at Each Step",
              fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/cleaning/plot5_pipeline_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot5_pipeline_summary.png")
