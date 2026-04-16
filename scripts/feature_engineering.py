"""
Feature Engineering – dataset_mood_smartphone_clean.csv
=========================================================

Input:  dataset_mood_smartphone_clean.csv  (2154 rows × 21 cols, daily, wide)
Output: dataset_mood_smartphone_features.csv

All features are engineered PER PARTICIPANT so that no participant's history
leaks into another's lag/rolling calculations. This is the most common mistake
in panel time-series and would cause optimistic bias in any downstream model.

Feature groups
--------------
1. Temporal          – day-of-week, is_weekend, days_into_study
2. Lag features      – previous 1/2/3/7 days of mood + key predictors
3. Rolling stats     – 3- and 7-day rolling mean & std
4. App aggregations  – domain-grouped sums of appCat.* columns
5. Ratios            – behavioural proportions (e.g. social / total app use)
6. Communication     – call + sms as a single social-contact signal
7. Within-person z   – centre + scale each variable by that participant's own
                       mean and std, so the model sees deviations from
                       personal baseline rather than absolute levels
8. Target            – next-day mood (mood shifted −1 within participant)
                       marked clearly so it is never used as an input feature
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/dataset_mood_smartphone_clean.csv", parse_dates=["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

print(f"Input shape: {df.shape}")

features = []   # will collect one processed DataFrame per participant

for pid, grp in df.groupby("id"):
    g = grp.copy().sort_values("date").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # 1. Temporal features
    #    Day-of-week and weekend flag capture weekly mood cycles (e.g. lower
    #    on Mondays, higher on Fridays/Saturdays) that are well-documented in
    #    ESM (experience-sampling) literature.
    #    days_into_study captures novelty/habituation: participants may report
    #    differently at the start of the study than at the end.
    # -----------------------------------------------------------------------
    g["day_of_week"]    = g["date"].dt.dayofweek          # 0=Mon … 6=Sun
    g["is_weekend"]     = (g["day_of_week"] >= 5).astype(int)
    g["days_into_study"] = (g["date"] - g["date"].min()).dt.days

    # -----------------------------------------------------------------------
    # 2. Lag features
    #    Autoregressive: yesterday's mood is the strongest single predictor
    #    of today's mood (typical r ≈ 0.5–0.7 in ESM studies).
    #    We also lag the affect dimensions, screen time, activity, and social
    #    contact — all have documented next-day effects on mood.
    #    7-day lag captures same-weekday last-week effect.
    # -----------------------------------------------------------------------
    lag_cols = ["mood", "circumplex.arousal", "circumplex.valence",
                "screen", "activity", "call", "sms"]

    for col in lag_cols:
        if col in g.columns:
            for lag in [1, 2, 3, 7]:
                g[f"{col}_lag{lag}"] = g[col].shift(lag)

    # -----------------------------------------------------------------------
    # 3. Rolling statistics (window computed on past data only via .shift(1))
    #    We shift by 1 before rolling so that day T's rolling window contains
    #    only days T-1, T-2, … — no future leakage.
    #    Mean  → trend / baseline level
    #    Std   → variability / instability (clinically meaningful for mood)
    # -----------------------------------------------------------------------
    roll_cols = ["mood", "screen", "activity",
                 "circumplex.arousal", "circumplex.valence"]

    for col in roll_cols:
        if col in g.columns:
            shifted = g[col].shift(1)
            for window in [3, 7]:
                g[f"{col}_roll{window}_mean"] = shifted.rolling(window, min_periods=1).mean()
                g[f"{col}_roll{window}_std"]  = shifted.rolling(window, min_periods=1).std()

    # -----------------------------------------------------------------------
    # 4. App usage aggregations
    #    Individual appCat columns are very sparse (e.g. appCat.weather has
    #    data on <10% of days). Grouping by psychological domain reduces
    #    sparsity and aligns with established constructs:
    #      social_time    → social connection / relatedness
    #      productive_time → goal-directed behaviour
    #      entertainment_time → passive leisure / escapism
    #      passive_time   → unclassified / background use
    #    We use sum(min_count=1) so that a day with ALL-NaN appCats stays NaN
    #    rather than becoming 0 (which would be a false observation).
    # -----------------------------------------------------------------------
    social_cols       = ["appCat.social",   "appCat.communication"]
    productive_cols   = ["appCat.office",   "appCat.utilities"]
    entertainment_cols= ["appCat.entertainment", "appCat.game"]
    passive_cols      = ["appCat.builtin",  "appCat.other", "appCat.unknown"]
    all_app_cols      = social_cols + productive_cols + entertainment_cols + \
                        passive_cols + ["appCat.travel", "appCat.finance",
                                        "appCat.weather"]

    def safe_sum(frame, cols):
        available = [c for c in cols if c in frame.columns]
        if not available:
            return pd.Series(np.nan, index=frame.index)
        return frame[available].sum(axis=1, min_count=1)

    g["social_time"]        = safe_sum(g, social_cols)
    g["productive_time"]    = safe_sum(g, productive_cols)
    g["entertainment_time"] = safe_sum(g, entertainment_cols)
    g["passive_time"]       = safe_sum(g, passive_cols)
    g["total_app_time"]     = safe_sum(g, all_app_cols)

    # -----------------------------------------------------------------------
    # 5. Ratios
    #    Absolute usage conflates "heavy phone user" with "social phone user".
    #    Dividing by total_app_time isolates the behavioural pattern.
    #    We guard against division by zero with np.where.
    # -----------------------------------------------------------------------
    for col in ["social_time", "productive_time", "entertainment_time", "passive_time"]:
        ratio_name = col.replace("_time", "_ratio")
        g[ratio_name] = np.where(
            g["total_app_time"] > 0,
            g[col] / g["total_app_time"],
            np.nan
        )

    # -----------------------------------------------------------------------
    # 6. Communication events
    #    call and sms are both binary event counts (how many times the phone
    #    was used for direct communication). Summing them gives a single,
    #    less-sparse signal for daily social contact via phone.
    # -----------------------------------------------------------------------
    g["comm_events"] = g[["call", "sms"]].sum(axis=1, min_count=1)

    # -----------------------------------------------------------------------
    # 7. Within-person centering (z-scores)
    #    Participants differ substantially in baseline mood (some people are
    #    consistently happier than others). A model trained on raw mood scores
    #    would learn "this person is always a 7" rather than "this person is
    #    lower than usual today".
    #    We centre and scale each variable by that participant's own mean and
    #    std computed on the FULL series. This is standard in multilevel /
    #    panel time-series modelling.
    # -----------------------------------------------------------------------
    centre_cols = ["mood", "screen", "activity",
                   "circumplex.arousal", "circumplex.valence",
                   "social_time", "total_app_time", "comm_events"]

    for col in centre_cols:
        if col in g.columns:
            mu  = g[col].mean()
            std = g[col].std()
            g[f"{col}_z"] = (g[col] - mu) / std if std > 0 else 0.0

    # -----------------------------------------------------------------------
    # 8. Target variable: next-day mood
    #    Shift mood backwards by 1 so that row T's target is T+1's mood.
    #    This is the standard framing for next-day mood prediction.
    #    We keep the original 'mood' column too (it will be used as a feature
    #    via mood_lag1 on the NEXT row, which is equivalent).
    #    Label clearly so it is never accidentally used as an input.
    # -----------------------------------------------------------------------
    g["mood_next_day"] = g["mood"].shift(-1)

    features.append(g)

# ---------------------------------------------------------------------------
# Combine all participants
# ---------------------------------------------------------------------------
feat_df = pd.concat(features, ignore_index=True)

print(f"\nOutput shape: {feat_df.shape}")
print(f"Total features created: {feat_df.shape[1] - 3}")  # minus id, date, target
print(f"\nAll columns ({feat_df.shape[1]}):")
for col in feat_df.columns:
    print(f"  {col}")

# ---------------------------------------------------------------------------
# Summary: usable rows (where both features and target are non-NaN)
# ---------------------------------------------------------------------------
usable = feat_df[feat_df["mood_next_day"].notna() & feat_df["mood_lag1"].notna()]
print(f"\nRows with mood_next_day + mood_lag1 available: {len(usable)}")
print(f"Participants represented:                       {usable['id'].nunique()}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
feat_df.to_csv("data/processed/dataset_mood_smartphone_features.csv", index=False)
print("\nSaved → dataset_mood_smartphone_features.csv")

# ---------------------------------------------------------------------------
# VISUALISATIONS – show what each feature group adds
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Use non-interactive backend if no display is available
import matplotlib
matplotlib.use("Agg")

# --- Plot 1: Lag features – autocorrelation of mood ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: scatter mood vs mood_lag1 (coloured by participant)
sample_pids = feat_df["id"].unique()[:6]
for pid in sample_pids:
    p = feat_df[feat_df["id"] == pid].dropna(subset=["mood", "mood_lag1"])
    axes[0].scatter(p["mood_lag1"], p["mood"], s=12, alpha=0.5, label=pid)
axes[0].set_xlabel("Mood (day t−1)")
axes[0].set_ylabel("Mood (day t)")
axes[0].set_title("Lag feature: mood_lag1 vs mood\n(each colour = one participant)", fontsize=10)
axes[0].legend(fontsize=7, ncol=2)

# Right: bar chart of correlation of each mood lag with current mood
lag_corrs = {}
for lag in [1, 2, 3, 7]:
    col = f"mood_lag{lag}"
    r = feat_df[["mood", col]].dropna().corr().iloc[0, 1]
    lag_corrs[f"lag {lag}d"] = r

axes[1].bar(lag_corrs.keys(), lag_corrs.values(), color="steelblue")
axes[1].set_title("Correlation of mood lags with current mood", fontsize=10)
axes[1].set_ylabel("Pearson r")
axes[1].set_ylim(0, 1)
for i, (k, v) in enumerate(lag_corrs.items()):
    axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

fig.suptitle("Feature Group 2 – Lag Features", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/features/feat_plot1_lags.png", dpi=150, bbox_inches="tight")
print("Saved feat_plot1_lags.png")

# --- Plot 2: Rolling statistics – mood_roll7_mean vs actual mood ---
fig, ax = plt.subplots(figsize=(14, 5))

pid = "AS14.26"
p = feat_df[feat_df["id"] == pid].dropna(subset=["mood"])
ax.plot(p["date"], p["mood"], color="salmon", alpha=0.6, linewidth=1.2, label="Daily mood")
ax.plot(p["date"], p["mood_roll7_mean"], color="steelblue", linewidth=2, label="7-day rolling mean")
ax.fill_between(
    p["date"],
    p["mood_roll7_mean"] - p["mood_roll7_std"].fillna(0),
    p["mood_roll7_mean"] + p["mood_roll7_std"].fillna(0),
    alpha=0.2, color="steelblue", label="±1 std (rolling 7d)"
)
ax.set_title(f"Feature Group 3 – Rolling Statistics ({pid})\n"
             "Rolling mean tracks trend; shaded band shows mood variability", fontsize=11, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Mood")
ax.legend()
plt.tight_layout()
plt.savefig("plots/features/feat_plot2_rolling.png", dpi=150, bbox_inches="tight")
print("Saved feat_plot2_rolling.png")

# --- Plot 3: App aggregations – stacked bar of domain time per participant ---
agg_cols   = ["social_time", "productive_time", "entertainment_time", "passive_time"]
agg_colors = ["steelblue", "seagreen", "salmon", "grey"]

per_person = feat_df.groupby("id")[agg_cols].mean()

fig, ax = plt.subplots(figsize=(14, 5))
bottom = np.zeros(len(per_person))
for col, color in zip(agg_cols, agg_colors):
    vals = per_person[col].fillna(0).values
    ax.bar(per_person.index, vals, bottom=bottom, label=col, color=color, alpha=0.85)
    bottom += vals

ax.set_title("Feature Group 4 – App Usage Aggregations\n"
             "Average daily seconds per domain per participant", fontsize=11, fontweight="bold")
ax.set_xlabel("Participant")
ax.set_ylabel("Seconds / day (mean)")
ax.set_xticks(range(len(per_person.index)))
ax.set_xticklabels(per_person.index, rotation=45, ha="right", fontsize=8)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("plots/features/feat_plot3_app_aggregations.png", dpi=150, bbox_inches="tight")
print("Saved feat_plot3_app_aggregations.png")

# --- Plot 4: Within-person centering – raw vs z-scored mood distributions ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for pid in feat_df["id"].unique():
    p = feat_df[feat_df["id"] == pid]
    axes[0].hist(p["mood"].dropna(),   bins=15, alpha=0.3, density=True)
    axes[1].hist(p["mood_z"].dropna(), bins=15, alpha=0.3, density=True)

axes[0].set_title("Raw mood distributions\n(participants differ in baseline)", fontsize=10)
axes[0].set_xlabel("Mood score")
axes[0].set_ylabel("Density")

axes[1].set_title("Within-person z-scored mood\n(centred on each participant's mean)", fontsize=10)
axes[1].set_xlabel("Mood z-score")
axes[1].set_ylabel("Density")

fig.suptitle("Feature Group 7 – Within-Person Centering", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/features/feat_plot4_centering.png", dpi=150, bbox_inches="tight")
print("Saved feat_plot4_centering.png")

# --- Plot 5: Feature correlation heatmap with mood_next_day ---
target_corr = (
    feat_df
    .select_dtypes(include="number")
    .drop(columns=["day_of_week"], errors="ignore")
    .corr()["mood_next_day"]
    .drop("mood_next_day")
    .dropna()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(7, 14))
colors = ["salmon" if v < 0 else "steelblue" for v in target_corr]
ax.barh(target_corr.index, target_corr.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Group 8 – Correlation of All Features\nwith mood_next_day (target)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Pearson r with next-day mood")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig("plots/features/feat_plot5_target_correlations.png", dpi=150, bbox_inches="tight")
print("Saved feat_plot5_target_correlations.png")
