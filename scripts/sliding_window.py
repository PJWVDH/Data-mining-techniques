"""
Sliding Window Dataset Builder
================================

Input:  dataset_mood_smartphone_clean.csv
Output: sliding_window_flat.csv   – 2D table, one row per window (for tree models)
        sliding_window_3d.npz     – 3D numpy arrays (for LSTMs / sequence models)

Why a sliding window?
---------------------
The lag-feature approach in feature_engineering.py is "wide": each row stays
at day T and adds columns for T-1, T-2, … separately. That works for
traditional ML but loses the sequential structure of the data.

A sliding window treats each W-day chunk as one *sample*:

    [day T-W, day T-W+1, …, day T-1]  →  predict mood on day T

This gives us:
  • A 2D flat version  – shape (n_windows, W×F)  – for Random Forest, XGBoost, etc.
  • A 3D tensor        – shape (n_windows, W, F)  – for LSTM, GRU, Transformer

Rules enforced here:
  1. Windows never cross participant boundaries.
  2. A window is only valid if EVERY day in the window AND the target day
     have non-NaN mood values. (Partial imputation within the window for
     non-mood features is fine and handled below.)
  3. Per-participant z-score normalisation is applied BEFORE windowing, using
     statistics from the FULL participant series — not from the window itself.
     This prevents leakage while keeping features on a comparable scale.

Window size W = 7 (one calendar week of history).
Change WINDOW_SIZE below to experiment with other sizes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 7          # days of history per sample
TARGET_COL  = "mood"     # what we're predicting (next day)

# Features used inside each window timestep.
# Selected for being meaningful AND relatively complete across the dataset.
WINDOW_FEATURES = [
    "mood",
    "circumplex.arousal",
    "circumplex.valence",
    "screen",
    "activity",
    "comm_events",          # call + sms (computed below)
    "social_time",          # appCat.social + appCat.communication
    "productive_time",      # appCat.office + appCat.utilities
    "entertainment_time",   # appCat.entertainment + appCat.game
    "day_of_week",
    "is_weekend",
]

# ---------------------------------------------------------------------------
# Load & basic prep
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/dataset_mood_smartphone_clean.csv", parse_dates=["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# Compute derived columns that aren't in the clean file yet
df["comm_events"]       = df[["call", "sms"]].sum(axis=1, min_count=1)
df["social_time"]       = df[["appCat.social", "appCat.communication"]].sum(axis=1, min_count=1)
df["productive_time"]   = df[["appCat.office", "appCat.utilities"]].sum(axis=1, min_count=1)
df["entertainment_time"]= df[["appCat.entertainment", "appCat.game"]].sum(axis=1, min_count=1)
df["day_of_week"]       = df["date"].dt.dayofweek
df["is_weekend"]        = (df["day_of_week"] >= 5).astype(float)

# ---------------------------------------------------------------------------
# Per-participant z-score normalisation
#   Computed from the FULL participant series so the scaler has seen
#   all available data — not just one window. Applied before windowing
#   so every value entering any window is already on the same scale.
#   Temporal features (day_of_week, is_weekend) are left as-is.
# ---------------------------------------------------------------------------
scale_cols = [c for c in WINDOW_FEATURES if c not in ("day_of_week", "is_weekend")]

norm_parts = []
participant_stats = {}   # store means/stds for reference

for pid, grp in df.groupby("id"):
    g = grp.copy()
    stats = {}
    for col in scale_cols:
        if col in g.columns:
            mu, sigma = g[col].mean(), g[col].std()
            stats[col] = (mu, sigma)
            g[col] = (g[col] - mu) / sigma if sigma > 0 else 0.0
    participant_stats[pid] = stats
    norm_parts.append(g)

df_norm = pd.concat(norm_parts, ignore_index=True).sort_values(["id", "date"])

# ---------------------------------------------------------------------------
# Slide the window
#   For each participant we walk through their daily series and extract every
#   valid window of length WINDOW_SIZE followed by a non-NaN target day.
# ---------------------------------------------------------------------------
windows_X   = []   # list of (WINDOW_SIZE, n_features) arrays
windows_y   = []   # list of scalar target values (normalised mood next day)
windows_meta= []   # id, window_start_date, target_date — for traceability

for pid, grp in df_norm.groupby("id"):
    g = grp.sort_values("date").reset_index(drop=True)

    mood_raw  = df[df["id"] == pid].sort_values("date")["mood"].values

    for i in range(len(g) - WINDOW_SIZE):
        window_rows  = g.iloc[i : i + WINDOW_SIZE]
        target_row   = g.iloc[i + WINDOW_SIZE]

        # Validity check: all days in window + target must have original mood
        # (use raw pre-normalised mood to check; NaN in normalised = NaN in raw)
        raw_mood_window = mood_raw[i : i + WINDOW_SIZE]
        raw_mood_target = mood_raw[i + WINDOW_SIZE]

        if np.any(np.isnan(raw_mood_window)) or np.isnan(raw_mood_target):
            continue

        # Extract feature matrix for the window
        feat_matrix = window_rows[WINDOW_FEATURES].values.astype(float)

        # For non-mood features that are NaN within the window, forward-fill
        # column-by-column (time axis = rows), then fill any remaining NaN
        # with 0 (mean of normalised data). This avoids discarding windows
        # just because e.g. screen data is missing on one interior day.
        for col_idx in range(feat_matrix.shape[1]):
            col = feat_matrix[:, col_idx]
            # forward fill
            mask = np.isnan(col)
            for j in range(1, len(col)):
                if mask[j] and not mask[j-1]:
                    col[j] = col[j-1]
                    mask[j] = False
            col[np.isnan(col)] = 0.0   # fill any remaining with 0 (= normalised mean)
            feat_matrix[:, col_idx] = col

        target_val = target_row[TARGET_COL]   # normalised mood on target day

        windows_X.append(feat_matrix)
        windows_y.append(target_val)
        windows_meta.append({
            "id":               pid,
            "window_start":     window_rows.iloc[0]["date"].date(),
            "target_date":      target_row["date"].date(),
            "target_mood_norm": target_val,
        })

X = np.array(windows_X)   # shape: (n_windows, WINDOW_SIZE, n_features)
y = np.array(windows_y)   # shape: (n_windows,)
meta = pd.DataFrame(windows_meta)

print(f"3D tensor X shape : {X.shape}  →  (n_windows, window_size, n_features)")
print(f"Target vector y   : {y.shape}")
print(f"Features per step : {WINDOW_FEATURES}")
print(f"\nWindows per participant:")
print(meta["id"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# 2D flat version  (for scikit-learn compatible models)
#   Column names: mood_t-6, screen_t-6, …, mood_t-0, screen_t-0
#   t-0 = most recent day in the window (day before the target)
# ---------------------------------------------------------------------------
flat_cols = [
    f"{feat}_t-{WINDOW_SIZE - 1 - step}"
    for step in range(WINDOW_SIZE)
    for feat in WINDOW_FEATURES
]

X_flat = X.reshape(X.shape[0], -1)
flat_df = pd.DataFrame(X_flat, columns=flat_cols)
flat_df.insert(0, "id",           meta["id"].values)
flat_df.insert(1, "window_start", meta["window_start"].values)
flat_df.insert(2, "target_date",  meta["target_date"].values)
flat_df["mood_next_day"] = y

flat_df.to_csv("data/processed/sliding_window_flat.csv", index=False)
print(f"\nFlat CSV shape: {flat_df.shape}")
print("Saved → sliding_window_flat.csv")

# ---------------------------------------------------------------------------
# 3D numpy save  (for deep learning)
# ---------------------------------------------------------------------------
np.savez(
    "data/processed/sliding_window_3d.npz",
    X=X,
    y=y,
    feature_names=np.array(WINDOW_FEATURES),
    participant_ids=meta["id"].values,
)
print("Saved → sliding_window_3d.npz")

# ===========================================================================
# VISUALISATIONS
# ===========================================================================

# --- Plot 1: One example window per participant feature ---
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for ax, feat in zip(axes, WINDOW_FEATURES):
    # Pick the first valid window for participant AS14.26
    idx = meta[meta["id"] == "AS14.26"].index[0]
    ax.plot(range(WINDOW_SIZE), X[idx, :, WINDOW_FEATURES.index(feat)],
            marker="o", color="steelblue", linewidth=1.5)
    ax.axvline(WINDOW_SIZE - 0.5, color="salmon", linewidth=1.5,
               linestyle="--", label="target →")
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel("Day in window (t-6 … t-0)")
    ax.set_xticks(range(WINDOW_SIZE))
    ax.set_xticklabels([f"t-{WINDOW_SIZE-1-i}" for i in range(WINDOW_SIZE)],
                       fontsize=7, rotation=30)

for ax in axes[len(WINDOW_FEATURES):]:
    ax.set_visible(False)

axes[0].legend(fontsize=8)
fig.suptitle(f"Plot 1 – One example window (W={WINDOW_SIZE}) for AS14.26\n"
             "Each panel shows one normalised feature across 7 days",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/windows/win_plot1_example_window.png", dpi=150, bbox_inches="tight")
print("Saved win_plot1_example_window.png")

# --- Plot 2: Windows per participant ---
fig, ax = plt.subplots(figsize=(13, 4))
counts = meta["id"].value_counts().sort_index()
ax.bar(counts.index, counts.values, color="steelblue")
ax.set_xlabel("Participant")
ax.set_ylabel(f"Number of valid windows (W={WINDOW_SIZE})")
ax.set_title(f"Plot 2 – Valid Windows per Participant\nTotal: {len(meta)}",
             fontsize=11, fontweight="bold")
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
for i, v in enumerate(counts.values):
    ax.text(i, v + 0.3, str(v), ha="center", fontsize=7)
plt.tight_layout()
plt.savefig("plots/windows/win_plot2_windows_per_participant.png", dpi=150, bbox_inches="tight")
print("Saved win_plot2_windows_per_participant.png")

# --- Plot 3: Feature heatmap across all windows for one participant ---
pid = "AS14.26"
pid_idx = meta[meta["id"] == pid].index
X_pid = X[pid_idx]   # shape: (n_windows_pid, W, F)

# Take mean over window axis to get (n_windows, n_features) — one value per feature per window
X_pid_mean = X_pid.mean(axis=1)   # average feature level per window

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(X_pid_mean.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
               interpolation="nearest")
ax.set_yticks(range(len(WINDOW_FEATURES)))
ax.set_yticklabels(WINDOW_FEATURES, fontsize=9)
ax.set_xlabel("Window index (chronological)")
ax.set_title(f"Plot 3 – Feature heatmap across all windows ({pid})\n"
             "Each column = one 7-day window; colour = mean normalised feature value",
             fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="z-score")
plt.tight_layout()
plt.savefig("plots/windows/win_plot3_feature_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved win_plot3_feature_heatmap.png")

# --- Plot 4: Distribution of window sizes vs target mood (raw, unnormalised) ---
# Recover raw target mood using participant stats
raw_targets = []
for _, row in meta.iterrows():
    pid = row["id"]
    mu, sigma = participant_stats[pid]["mood"]
    raw_targets.append(row["target_mood_norm"] * sigma + mu)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(raw_targets, bins=25, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Next-day mood (original scale)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of target values\nacross all windows")

axes[1].hist(y, bins=25, color="salmon", edgecolor="white")
axes[1].set_xlabel("Next-day mood (normalised)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of target values\n(normalised — input to model)")

fig.suptitle("Plot 4 – Target Variable Distribution Across All Windows",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/windows/win_plot4_target_distribution.png", dpi=150, bbox_inches="tight")
print("Saved win_plot4_target_distribution.png")
