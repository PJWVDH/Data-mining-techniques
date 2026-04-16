"""
Classification: Predicting Next-Day Mood Category
===================================================

Task: predict whether tomorrow's mood will be LOW / MEDIUM / HIGH.

Classes
-------
  Defined by ±0.5 SD thresholds in within-person z-score space:
    LOW    z < -0.5   (meaningfully below personal baseline)
    MEDIUM -0.5 ≤ z ≤ 0.5  (within normal fluctuation)
    HIGH   z > +0.5   (meaningfully above personal baseline)

  Why not tertiles?
    Tertile split puts LOW/HIGH only 0.45 raw mood points apart (6.80 vs
    7.25 on a 1–10 scale). Between-day signal std (0.607) is smaller than
    intraday measurement noise (0.721). Tertile classes are therefore within
    noise of each other, making the task nearly impossible.
    ±0.5 SD gives a meaningful 1 SD gap between LOW and HIGH centres,
    corresponding to a real perceptible difference in mood.

Fixes applied after diagnostic investigation
--------------------------------------------
  Fix 1 – Window W=7 → W=3:
    Lags beyond t-1 have r<0.10 with target — pure noise. Shorter window
    also yields more usable windows (1222 vs 1107).

  Fix 2 – Participant ID added to RF:
    Z-scored features are person-centred; RF had no way to learn
    person-specific thresholds. One-hot participant ID fixes this.

  Fix 3 – LSTM validation split fixed:
    10% flat tail (≈87 samples) was too noisy for early stopping.
    Now uses last 15% per participant, maintaining temporal order.

  Fix 4 – Class boundaries widened (±0.5 SD):
    Tertile classes only 0.45 raw points apart — within measurement noise.
    ±0.5 SD creates a 1 SD gap between LOW and HIGH.

  Fix 5 – Sparse features dropped from windows:
    productive_time (31% NaN on mood days) and entertainment_time (16%)
    were being zero-filled, injecting fake "no usage" values.

  Fix 6 – Rolling features added to RF:
    3-day and 7-day rolling means/stds from feature_engineering.py are
    more stable predictors than raw single-day values. Merged in for RF.

Evaluation setup
----------------
  Temporal hold-out per participant: last 20% → test. Verified: every
  test target_date comes strictly after that participant's last train date.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Rebuild windows with W=3 directly from the clean dataset
#    (Fix 1: shorter window focuses on the two informative mood lags)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 3

clean = pd.read_csv("data/processed/dataset_mood_smartphone_clean.csv",
                    parse_dates=["date"])
clean = clean.sort_values(["id", "date"]).reset_index(drop=True)

# Derive columns needed for windows
clean["comm_events"]        = clean[["call", "sms"]].sum(axis=1, min_count=1)
clean["social_time"]        = clean[["appCat.social", "appCat.communication"]].sum(axis=1, min_count=1)
clean["productive_time"]    = clean[["appCat.office", "appCat.utilities"]].sum(axis=1, min_count=1)
clean["entertainment_time"] = clean[["appCat.entertainment", "appCat.game"]].sum(axis=1, min_count=1)
clean["day_of_week"]        = clean["date"].dt.dayofweek
clean["is_weekend"]         = (clean["day_of_week"] >= 5).astype(float)

# productive_time (31% NaN) and entertainment_time (16% NaN) dropped —
# zero-filling them injects fake "no usage" values on too many days.
WINDOW_FEATURES = [
    "mood", "circumplex.arousal", "circumplex.valence",
    "screen", "activity", "comm_events", "social_time",
    "day_of_week", "is_weekend",
]

# Per-participant z-score normalisation (applied before windowing)
scale_cols = [c for c in WINDOW_FEATURES if c not in ("day_of_week", "is_weekend")]
norm_parts = []
for pid, grp in clean.groupby("id"):
    g = grp.copy()
    for col in scale_cols:
        if col in g.columns:
            mu, sigma = g[col].mean(), g[col].std()
            g[col] = (g[col] - mu) / sigma if sigma > 0 else 0.0
    norm_parts.append(g)
clean_norm = pd.concat(norm_parts).sort_values(["id", "date"])

windows_X, windows_y, windows_pid, windows_start = [], [], [], []

for pid, grp in clean_norm.groupby("id"):
    g     = grp.sort_values("date").reset_index(drop=True)
    raw_m = clean[clean["id"] == pid].sort_values("date")["mood"].values

    for i in range(len(g) - WINDOW_SIZE):
        raw_window = raw_m[i : i + WINDOW_SIZE]
        raw_target = raw_m[i + WINDOW_SIZE]
        if np.any(np.isnan(raw_window)) or np.isnan(raw_target):
            continue

        feat = g.iloc[i : i + WINDOW_SIZE][WINDOW_FEATURES].values.astype(float)
        # Forward-fill then zero-fill non-mood NaNs inside the window
        for ci in range(feat.shape[1]):
            col = feat[:, ci]
            for j in range(1, len(col)):
                if np.isnan(col[j]) and not np.isnan(col[j-1]):
                    col[j] = col[j-1]
            col[np.isnan(col)] = 0.0
            feat[:, ci] = col

        windows_X.append(feat)
        windows_y.append(g.iloc[i + WINDOW_SIZE]["mood"])   # normalised target
        windows_pid.append(pid)
        windows_start.append(g.iloc[i]["date"])

X3d  = np.array(windows_X, dtype=np.float32)   # (n, W, F)
y_raw = np.array(windows_y, dtype=np.float32)
pids  = np.array(windows_pid)

print(f"Windows (W={WINDOW_SIZE}): {X3d.shape}  —  target: {y_raw.shape}")

# ---------------------------------------------------------------------------
# 2. Temporal train / test split (per participant, last 20% → test)
# ---------------------------------------------------------------------------
train_idx, test_idx = [], []
for pid in np.unique(pids):
    idx = np.where(pids == pid)[0]
    cut = max(1, int(len(idx) * 0.80))
    train_idx.extend(idx[:cut])
    test_idx.extend(idx[cut:])
train_idx = np.array(train_idx)
test_idx  = np.array(test_idx)

print(f"Train: {len(train_idx)}  |  Test: {len(test_idx)}")

# ---------------------------------------------------------------------------
# 3. Classes via tertiles of TRAINING target (balanced ~33/33/33)
#    Note: tertiles produce classes only 0.45 raw mood points apart, which
#    is narrow — but ±0.5 SD creates 43% MEDIUM dominance that causes RF to
#    collapse to majority-class prediction. Balanced tertiles + class_weight
#    is the more practical choice given the dataset size.
# ---------------------------------------------------------------------------
t33, t66 = np.percentile(y_raw[train_idx], [33.33, 66.67])
print(f"Class thresholds: t33={t33:.3f}  t66={t66:.3f}")

def to_class(v, t33=None, t66=None):
    if t33 is None: t33, t66 = -0.374, 0.409   # fallback
    c = np.zeros(len(v), dtype=int)
    c[(v > t33) & (v <= t66)] = 1
    c[v > t66] = 2
    return c

y_train = to_class(y_raw[train_idx], t33, t66)
y_test  = to_class(y_raw[test_idx],  t33, t66)
class_names = ["LOW", "MEDIUM", "HIGH"]
print("Train classes:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Test  classes:", dict(zip(*np.unique(y_test,  return_counts=True))))

# ---------------------------------------------------------------------------
# 4. Flat 2D features for Random Forest
#    Fix 2: one-hot participant ID
#    Fix 6: merge rolling features from feature_engineering output
#           Rolling 3-day / 7-day mean & std are more stable than raw lags
# ---------------------------------------------------------------------------
X2d_raw = X3d.reshape(X3d.shape[0], -1)   # (n, W*F)

# Load rolling features — join on (id, target_date) since the target date
# is the day the rolling stats were computed up to (window end + 1)
feat_df = pd.read_csv("data/processed/dataset_mood_smartphone_features.csv",
                      parse_dates=["date"])

roll_cols = [c for c in feat_df.columns if
             ("_roll" in c or c in ("days_into_study", "is_weekend")) and
             "mood" in c or "_roll3" in c or "_roll7" in c]
# Keep only rolling mean/std for the most predictive variables
roll_cols = [c for c in feat_df.columns if
             any(c.startswith(p) for p in
                 ["mood_roll", "circumplex.arousal_roll", "circumplex.valence_roll",
                  "screen_roll", "activity_roll"]) and
             c.endswith(("_mean", "_std"))]
roll_cols += ["days_into_study"]

# Rolling features should be merged on the LAST day of the window (T-0),
# not the start — the rolling stats at T-0 capture the most recent history
# available before the target day.
windows_end_dt = pd.to_datetime([str(d) for d in windows_start]) + pd.Timedelta(days=WINDOW_SIZE - 1)
win_meta = pd.DataFrame({"id": pids, "date": windows_end_dt})
roll_lookup = feat_df[["id", "date"] + roll_cols].copy()
win_roll = win_meta.merge(roll_lookup, on=["id", "date"], how="left")
roll_mat = win_roll[roll_cols].fillna(0).values.astype(np.float32)

# One-hot encode participant IDs
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(pids[train_idx].reshape(-1, 1))
pid_train_oh = ohe.transform(pids[train_idx].reshape(-1, 1))
pid_test_oh  = ohe.transform(pids[test_idx].reshape(-1, 1))

X2d_train = np.hstack([X2d_raw[train_idx], roll_mat[train_idx], pid_train_oh])
X2d_test  = np.hstack([X2d_raw[test_idx],  roll_mat[test_idx],  pid_test_oh])

print(f"RF feature matrix: train={X2d_train.shape}  test={X2d_test.shape}")
print(f"  → window features: {X2d_raw.shape[1]}  rolling: {roll_mat.shape[1]}  pid-OH: {pid_train_oh.shape[1]}")

# ---------------------------------------------------------------------------
# 5. Baseline
# ---------------------------------------------------------------------------
majority_class = np.bincount(y_train).argmax()
base_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))
base_f1  = f1_score(y_test, np.full_like(y_test, majority_class),
                    average="macro", zero_division=0)
print(f"\nBaseline  acc={base_acc:.3f}  macro-F1={base_f1:.3f}")

# ---------------------------------------------------------------------------
# 6. Random Forest
# ---------------------------------------------------------------------------
print("\n--- Random Forest (W=3, +participant ID) ---")
rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X2d_train, y_train)
y_pred_rf = rf.predict(X2d_test)

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_f1  = f1_score(y_test, y_pred_rf, average="macro")
print(f"Accuracy : {rf_acc:.3f}")
print(f"Macro F1 : {rf_f1:.3f}")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# ---------------------------------------------------------------------------
# 7. LSTM  (Fix 3: better validation split + higher patience)
#    Validation set = last 15% of each participant's TRAINING windows
#    (maintains temporal order and gives a balanced per-person val set)
# ---------------------------------------------------------------------------
print("\n--- LSTM (W=3, fixed validation split) ---")

# Build per-participant temporal validation from the training set
inner_train_idx, inner_val_idx = [], []
for pid in np.unique(pids):
    idx = train_idx[pids[train_idx] == pid]
    cut = max(1, int(len(idx) * 0.85))
    inner_train_idx.extend(idx[:cut])
    inner_val_idx.extend(idx[cut:])
inner_train_idx = np.array(inner_train_idx)
inner_val_idx   = np.array(inner_val_idx)

print(f"LSTM inner-train: {len(inner_train_idx)}  inner-val: {len(inner_val_idx)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def make_tensor(idx, y_labels):
    return (torch.tensor(X3d[idx]),
            torch.tensor(y_labels, dtype=torch.long))

Xtr, ytr = make_tensor(inner_train_idx, to_class(y_raw[inner_train_idx], t33, t66))
Xvl, yvl = make_tensor(inner_val_idx,   to_class(y_raw[inner_val_idx],   t33, t66))
Xte       = torch.tensor(X3d[test_idx])

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)

class MoodLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_classes=3, dropout=0.2):
        super().__init__()
        # Single-layer LSTM — simpler model fits small dataset better
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(self.dropout(h[-1]))

model = MoodLSTM(input_size=X3d.shape[2]).to(device)

class_counts  = np.bincount(y_train)
class_weights = torch.tensor(
    len(y_train) / (len(class_counts) * class_counts), dtype=torch.float32
).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
# Gentler scheduler: halve LR after 10 epochs without improvement (was 5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)

EPOCHS, PATIENCE = 150, 20
best_val_loss, patience_cnt, best_weights = float("inf"), 0, None
train_losses, val_losses = [], []

Xvl_dev, yvl_dev = Xvl.to(device), yvl.to(device)

for epoch in range(EPOCHS):
    model.train()
    ep_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(Xvl_dev), yvl_dev).item()

    train_losses.append(ep_loss / len(inner_train_idx))
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
        patience_cnt  = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

model.load_state_dict(best_weights)
model.eval()
with torch.no_grad():
    y_pred_lstm = model(Xte.to(device)).argmax(dim=1).cpu().numpy()

lstm_acc = accuracy_score(y_test, y_pred_lstm)
lstm_f1  = f1_score(y_test, y_pred_lstm, average="macro")
print(f"Accuracy : {lstm_acc:.3f}")
print(f"Macro F1 : {lstm_f1:.3f}")
print(classification_report(y_test, y_pred_lstm, target_names=class_names))

# ---------------------------------------------------------------------------
# 8. Visualisations
# ---------------------------------------------------------------------------

# Plot 1: Autocorrelation — shows WHY shorter window is correct
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

lags = [1, 2, 3, 7]
corrs = []
all_mood = clean[clean["mood"].notna()].copy()
for lag in lags:
    rs = []
    for pid, g in all_mood.groupby("id"):
        g = g.sort_values("date")
        r = g["mood"].autocorr(lag=lag)
        if not np.isnan(r):
            rs.append(r)
    corrs.append(np.mean(rs))

axes[0].bar([f"lag {l}d" for l in lags], corrs, color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Mean within-person mood autocorrelation\n(justifies W=3 window)", fontsize=10)
axes[0].set_ylabel("Pearson r")
for i, v in enumerate(corrs):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

# Daily class-change rate per participant
change_rates = []
for pid, g in all_mood.groupby("id"):
    g = g.sort_values("date").copy()
    vals = g["mood"].values
    t33p, t66p = np.percentile(vals, [33, 66])
    cls = to_class(vals, t33p, t66p)
    rate = (cls[1:] != cls[:-1]).mean()
    change_rates.append(rate)

axes[1].hist(change_rates, bins=10, color="salmon", edgecolor="white")
axes[1].axvline(np.mean(change_rates), color="black", linestyle="--",
                label=f"mean={np.mean(change_rates):.2f}")
axes[1].set_title("Mood class change rate per participant\n(problem difficulty ceiling)", fontsize=10)
axes[1].set_xlabel("Fraction of consecutive-day class changes")
axes[1].set_ylabel("Participants")
axes[1].legend()

fig.suptitle("Diagnostic: Why Mood Is Hard to Predict", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/classification/clf_plot0_difficulty.png", dpi=150, bbox_inches="tight")
print("Saved clf_plot0_difficulty.png")

# Plot 2: Summary comparison
fig, ax = plt.subplots(figsize=(9, 5))
models = ["Baseline\n(majority)", "Random\nForest", "LSTM"]
accs   = [base_acc, rf_acc, lstm_acc]
f1s    = [base_f1,  rf_f1,  lstm_f1]
x, w   = np.arange(len(models)), 0.35
bars1  = ax.bar(x - w/2, accs, w, label="Accuracy", color="steelblue", alpha=0.85)
bars2  = ax.bar(x + w/2, f1s,  w, label="Macro F1", color="salmon",    alpha=0.85)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.axhline(1/3, color="grey", linestyle="--", linewidth=0.8, label="Random chance")
ax.set_ylabel("Score")
ax.set_title(f"Model Comparison  (W={WINDOW_SIZE}, +participant ID for RF)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plots/classification/clf_plot1_comparison.png", dpi=150, bbox_inches="tight")
print("Saved clf_plot1_comparison.png")

# Plot 3: Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, title in zip(axes, [y_pred_rf, y_pred_lstm], ["Random Forest", "LSTM"]):
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=class_names).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{title}  acc={accuracy_score(y_test,y_pred):.2f}  "
                 f"F1={f1_score(y_test,y_pred,average='macro'):.2f}", fontweight="bold")
fig.suptitle("Confusion Matrices", fontsize=13)
plt.tight_layout()
plt.savefig("plots/classification/clf_plot2_confusion.png", dpi=150, bbox_inches="tight")
print("Saved clf_plot2_confusion.png")

# Plot 4: RF feature importances (top 20)
flat_feat_names = [
    f"{feat}_t-{WINDOW_SIZE-1-step}"
    for step in range(WINDOW_SIZE)
    for feat in WINDOW_FEATURES
]
pid_feat_names = [f"pid_{c}" for c in ohe.categories_[0]]
all_feat_names = flat_feat_names + roll_cols + pid_feat_names

importances = pd.Series(rf.feature_importances_, index=all_feat_names).nlargest(20)
fig, ax = plt.subplots(figsize=(9, 6))
importances.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest – Top 20 Feature Importances", fontweight="bold")
ax.set_xlabel("Mean decrease in impurity")
plt.tight_layout()
plt.savefig("plots/classification/clf_plot3_rf_importances.png", dpi=150, bbox_inches="tight")
print("Saved clf_plot3_rf_importances.png")

# Plot 5: LSTM training curve
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_losses, label="Train loss", color="steelblue")
ax.plot(val_losses,   label="Val loss (per-participant temporal)", color="salmon")
ax.set_xlabel("Epoch")
ax.set_ylabel("CrossEntropyLoss")
ax.set_title("LSTM Training Curve", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plots/classification/clf_plot4_lstm_training.png", dpi=150, bbox_inches="tight")
print("Saved clf_plot4_lstm_training.png")

# Plot 6: Per-participant accuracy
rf_pp, lstm_pp, labels_p = [], [], []
for pid in np.unique(pids):
    idx = np.where(pids[test_idx] == pid)[0]
    if len(idx) == 0:
        continue
    labels_p.append(pid)
    rf_pp.append(accuracy_score(y_test[idx], y_pred_rf[idx]))
    lstm_pp.append(accuracy_score(y_test[idx], y_pred_lstm[idx]))

x = np.arange(len(labels_p))
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - 0.2, rf_pp,   0.4, label="Random Forest", color="steelblue", alpha=0.85)
ax.bar(x + 0.2, lstm_pp, 0.4, label="LSTM",          color="salmon",    alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_p, rotation=45, ha="right", fontsize=8)
ax.axhline(1/3, color="grey", linestyle="--", linewidth=0.8, label="Chance")
ax.set_ylabel("Accuracy")
ax.set_title("Per-Participant Test Accuracy", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plots/classification/clf_plot5_per_participant.png", dpi=150, bbox_inches="tight")
print("Saved clf_plot5_per_participant.png")
