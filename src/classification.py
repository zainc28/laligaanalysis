import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------------------------
df = pd.read_csv("../data/matches_cleaned.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Loaded {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

# ------------------------------------------------------------------------------------
# Build Team Game Log
# For each match, record each team's own stats regardless of home/away.
# is_home tracked so we can split home vs away win rates separately.
# ------------------------------------------------------------------------------------
records = []
for _, row in df.iterrows():
    records.append({
        "Date":            row["Date"],
        "team":            row["HomeTeam"],
        "goals_scored":    row["FTHG"],
        "goals_conceded":  row["FTAG"],
        "shots":           row["HS"],
        "shots_on_target": row["HST"],
        "corners":         row["HC"],
        "fouls":           row["HF"],
        "won":             1 if row["FTR"] == "H" else 0,
        "is_home":         1,
    })
    records.append({
        "Date":            row["Date"],
        "team":            row["AwayTeam"],
        "goals_scored":    row["FTAG"],
        "goals_conceded":  row["FTHG"],
        "shots":           row["AS"],
        "shots_on_target": row["AST"],
        "corners":         row["AC"],
        "fouls":           row["AF"],
        "won":             1 if row["FTR"] == "A" else 0,
        "is_home":         0,
    })

game_log = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)

# ------------------------------------------------------------------------------------
# Rolling Average Helper
# Returns avg stats over the last N games before before_date for a given team.
# Returns None if the team has fewer than N prior games.
# ------------------------------------------------------------------------------------
WINDOW = 5

def get_rolling(team, before_date, window=WINDOW):
    past = game_log[(game_log["team"] == team) & (game_log["Date"] < before_date)]
    if len(past) < window:
        return None
    recent = past.tail(window)

    # Separate home/away win rates within the window
    home_games = recent[recent["is_home"] == 1]
    away_games = recent[recent["is_home"] == 0]

    return {
        "goals_scored":    recent["goals_scored"].mean(),
        "goals_conceded":  recent["goals_conceded"].mean(),
        "goal_diff":       (recent["goals_scored"] - recent["goals_conceded"]).mean(),
        "shots":           recent["shots"].mean(),
        "shots_on_target": recent["shots_on_target"].mean(),
        "corners":         recent["corners"].mean(),
        "fouls":           recent["fouls"].mean(),
        "win_rate":        recent["won"].mean(),
        # Home/away win rates — default to overall win rate if no games of that type in window
        "home_win_rate":   home_games["won"].mean() if len(home_games) > 0 else recent["won"].mean(),
        "away_win_rate":   away_games["won"].mean() if len(away_games) > 0 else recent["won"].mean(),
    }

# ------------------------------------------------------------------------------------
# Head-to-Head Helper
# Returns the home team's historical win rate against this specific away team.
# Returns 0.33 (neutral) if they've never played before.
# ------------------------------------------------------------------------------------
def get_h2h(home_team, away_team, before_date):
    past = df[
        (df["HomeTeam"] == home_team) &
        (df["AwayTeam"] == away_team) &
        (df["Date"] < before_date)
    ]
    if len(past) == 0:
        return 0.33  # no history, assume neutral
    return (past["FTR"] == "H").mean()

# ------------------------------------------------------------------------------------
# Feature Engineering
# For each match: rolling stats for home + away, rest days, H2H, implied odds
# Drop rows where either team has fewer than WINDOW prior games
# ------------------------------------------------------------------------------------
feature_rows = []

for _, row in df.iterrows():
    home_stats = get_rolling(row["HomeTeam"], row["Date"])
    away_stats = get_rolling(row["AwayTeam"], row["Date"])

    if home_stats is None or away_stats is None:
        continue

    # Rest days
    home_past = game_log[(game_log["team"] == row["HomeTeam"]) & (game_log["Date"] < row["Date"])]
    away_past = game_log[(game_log["team"] == row["AwayTeam"]) & (game_log["Date"] < row["Date"])]
    home_rest = (row["Date"] - home_past["Date"].max()).days if len(home_past) > 0 else 7
    away_rest = (row["Date"] - away_past["Date"].max()).days if len(away_past) > 0 else 7

    # Implied probability from betting odds (1/odds, normalized to sum to 1)
    # Skip if odds are missing (some older rows may not have them)
    b365h = row.get("B365H", np.nan)
    b365d = row.get("B365D", np.nan)
    b365a = row.get("B365A", np.nan)
    if pd.notna(b365h) and pd.notna(b365d) and pd.notna(b365a) and b365h > 0 and b365d > 0 and b365a > 0:
        raw_h = 1 / b365h
        raw_d = 1 / b365d
        raw_a = 1 / b365a
        total = raw_h + raw_d + raw_a
        implied_h = raw_h / total
        implied_d = raw_d / total
        implied_a = raw_a / total
    else:
        implied_h = implied_d = implied_a = 0.33

    feature_rows.append({
        # Match identifiers (not used as model features, stripped out before training)
        "_date":      row["Date"],
        "_home_team": row["HomeTeam"],
        "_away_team": row["AwayTeam"],
        # Implied win probabilities from bookmaker odds (very strong signal)
        "implied_home_prob": implied_h,
        "implied_draw_prob": implied_d,
        "implied_away_prob": implied_a,
        # Home team rolling stats
        "h_goals_scored":    home_stats["goals_scored"],
        "h_goals_conceded":  home_stats["goals_conceded"],
        "h_goal_diff":       home_stats["goal_diff"],
        "h_shots":           home_stats["shots"],
        "h_shots_on_target": home_stats["shots_on_target"],
        "h_corners":         home_stats["corners"],
        "h_fouls":           home_stats["fouls"],
        "h_win_rate":        home_stats["win_rate"],
        "h_home_win_rate":   home_stats["home_win_rate"],
        "h_rest_days":       home_rest,
        # Away team rolling stats
        "a_goals_scored":    away_stats["goals_scored"],
        "a_goals_conceded":  away_stats["goals_conceded"],
        "a_goal_diff":       away_stats["goal_diff"],
        "a_shots":           away_stats["shots"],
        "a_shots_on_target": away_stats["shots_on_target"],
        "a_corners":         away_stats["corners"],
        "a_fouls":           away_stats["fouls"],
        "a_win_rate":        away_stats["win_rate"],
        "a_away_win_rate":   away_stats["away_win_rate"],
        "a_rest_days":       away_rest,
        # Head-to-head
        "h2h_home_win_rate": get_h2h(row["HomeTeam"], row["AwayTeam"], row["Date"]),
        # Target
        "FTR": row["FTR"],
    })

features_df = pd.DataFrame(feature_rows)
print(f"Feature matrix: {len(features_df)} rows after dropping early-season matches")

# ------------------------------------------------------------------------------------
# Build X and y
# ------------------------------------------------------------------------------------
# Strip identifier columns before building X — they are not model features
meta_cols = ["_date", "_home_team", "_away_team"]
feature_cols = [c for c in features_df.columns if c != "FTR" and c not in meta_cols]
X = features_df[feature_cols].values
y_raw = features_df["FTR"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)  # A=0, D=1, H=2 (alphabetical)
print(f"Classes: {list(le.classes_)}")

# ------------------------------------------------------------------------------------
# Chronological Train/Test Split (80/20)
# Never shuffle — that leaks future results into training
# ------------------------------------------------------------------------------------
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train)} matches | Test: {len(X_test)} matches")

# ------------------------------------------------------------------------------------
# Scale Features
# Fit on train only, apply to both — required for Logistic Regression
# ------------------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------------------------------------------------------------------------------------
# Baseline: always predict the most common class (Home win)
# ------------------------------------------------------------------------------------
most_common = np.bincount(y_train).argmax()
baseline_preds = np.full(len(y_test), most_common)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"\nBaseline accuracy (always predict '{le.classes_[most_common]}'): {baseline_acc:.3f}")

# ------------------------------------------------------------------------------------
# IF-THEN Rules Classifier
# Implements the "Using IF-THEN Rules for Classification" approach from lecture notes.
# Rules are based on implied betting probabilities — the strongest signal in the data.
# Thresholds are fitted by grid search over training data.
# Label encoding (alphabetical): A=0, D=1, H=2
# ------------------------------------------------------------------------------------
class RuleBasedClassifier:
    """
    IF implied_home_prob >= home_thresh  THEN predict H (home win)
    ELIF implied_away_prob >= away_thresh THEN predict A (away win)
    ELSE                                  predict D (draw)
    """
    def __init__(self):
        self.home_thresh = 0.45
        self.away_thresh = 0.38

    def fit(self, X, y):
        # Grid search over threshold pairs to maximise training accuracy
        best_acc, best_h, best_a = -1, self.home_thresh, self.away_thresh
        for h_t in np.arange(0.30, 0.65, 0.05):
            for a_t in np.arange(0.25, 0.55, 0.05):
                preds = self._apply(X, h_t, a_t)
                acc = (preds == y).mean()
                if acc > best_acc:
                    best_acc, best_h, best_a = acc, h_t, a_t
        self.home_thresh = best_h
        self.away_thresh = best_a
        print(f"  IF-THEN Rules fitted: home_thresh={best_h:.2f}, away_thresh={best_a:.2f}")
        return self

    def predict(self, X):
        return self._apply(X, self.home_thresh, self.away_thresh)

    def _apply(self, X, home_thresh, away_thresh):
        implied_home = X[:, 0]   # implied_home_prob
        implied_away = X[:, 2]   # implied_away_prob
        preds = np.where(
            implied_home >= home_thresh, 2,          # H
            np.where(implied_away >= away_thresh, 0, # A
                     1)                               # D
        )
        return preds


# ------------------------------------------------------------------------------------
# Train Models
# class_weight='balanced' makes models try to predict draws instead of ignoring them
# ------------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Naive Bayes":         GaussianNB(),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "IF-THEN Rules":       RuleBasedClassifier(),
}

results = {}
for name, model in models.items():
    uses_scaled = name in ("Logistic Regression", "Naive Bayes")
    X_tr = X_train_scaled if uses_scaled else X_train
    X_te = X_test_scaled  if uses_scaled else X_test

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}

    print(f"\n{'='*50}")
    print(f"  {name}  --  Accuracy: {acc:.3f}")
    print(f"{'='*50}")
    print(classification_report(y_test, preds, target_names=le.classes_))

# ------------------------------------------------------------------------------------
# Identify Best Model
# ------------------------------------------------------------------------------------
best_name = max(results, key=lambda n: results[n]["accuracy"])
best = results[best_name]
print(f"\nBest model: {best_name} ({best['accuracy']:.3f})")

# ------------------------------------------------------------------------------------
# Visualizations — saved to output/figures/
# ------------------------------------------------------------------------------------
os.makedirs("../output/figures", exist_ok=True)

# 1. Confusion Matrix for best model
cm = confusion_matrix(y_test, best["preds"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(le.classes_)))
ax.set_yticks(range(len(le.classes_)))
ax.set_xticklabels(le.classes_)
ax.set_yticklabels(le.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix -- {best_name}")
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("../output/figures/confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")

# 2. Feature Importance from Random Forest
LABEL_MAP = {
    "implied_home_prob":  "Implied Home Prob.",
    "implied_draw_prob":  "Implied Draw Prob.",
    "implied_away_prob":  "Implied Away Prob.",
    "h_goals_scored":     "Home Goals Scored",
    "h_goals_conceded":   "Home Goals Conceded",
    "h_goal_diff":        "Home Goal Diff.",
    "h_shots":            "Home Shots",
    "h_shots_on_target":  "Home Shots on Target",
    "h_corners":          "Home Corners",
    "h_fouls":            "Home Fouls",
    "h_win_rate":         "Home Win Rate",
    "h_home_win_rate":    "Home Win Rate (home)",
    "h_rest_days":        "Home Rest Days",
    "a_goals_scored":     "Away Goals Scored",
    "a_goals_conceded":   "Away Goals Conceded",
    "a_goal_diff":        "Away Goal Diff.",
    "a_shots":            "Away Shots",
    "a_shots_on_target":  "Away Shots on Target",
    "a_corners":          "Away Corners",
    "a_fouls":            "Away Fouls",
    "a_win_rate":         "Away Win Rate",
    "a_away_win_rate":    "Away Win Rate (away)",
    "a_rest_days":        "Away Rest Days",
    "h2h_home_win_rate":  "H2H Win Rate",
}

rf = results["Random Forest"]["model"]
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]  # descending for vertical bar
sorted_labels = [LABEL_MAP.get(feature_cols[i], feature_cols[i]) for i in sorted_idx]
sorted_vals   = importances[sorted_idx]

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(sorted_vals)), sorted_vals, color="#2c5f8a", edgecolor="white")
ax.set_xticks(range(len(sorted_vals)))
ax.set_xticklabels(sorted_labels, rotation=40, ha="right", fontsize=8.5)
ax.set_ylabel("Importance", fontsize=10)
ax.set_title("Random Forest Feature Importances", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("../output/figures/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved feature_importance.png")

# 3. Model Accuracy Comparison
fig, ax = plt.subplots(figsize=(10, 4))
names = list(results.keys())
accs  = [results[n]["accuracy"] for n in names]
colors = ["#4C72B0", "#C44E52", "#DD8452", "#55A868", "#8172B2"]
bars = ax.bar(names, accs, color=colors[:len(names)])
ax.axhline(baseline_acc, color="red", linestyle="--", label=f"Baseline ({baseline_acc:.2f})")
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
ax.legend()
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center")
plt.tight_layout()
plt.savefig("../output/figures/model_comparison.png", dpi=150)
plt.close()
print("Saved model_comparison.png")

# 4. Predicted vs Actual — 2025/26 Season Matchups
# Pull test set rows that belong to the 2025/26 season and show team names,
# predicted outcome, actual outcome, and whether the model was correct.
test_meta = features_df.iloc[split_idx:][meta_cols + ["FTR"]].copy()
test_meta["predicted"] = le.inverse_transform(best["preds"])
test_meta["correct"]   = test_meta["FTR"] == test_meta["predicted"]

# Filter to 2025/26 season only (dates starting Aug 2025)
season_mask = test_meta["_date"] >= "2025-08-01"
season_df = test_meta[season_mask].copy()
season_df = season_df.sort_values("_date").reset_index(drop=True)

if len(season_df) > 0:
    # Show up to 40 matches per page so text stays readable
    per_page = 40
    pages = [season_df.iloc[i:i+per_page] for i in range(0, len(season_df), per_page)]

    for page_num, page_df in enumerate(pages):
        n = len(page_df)
        fig, ax = plt.subplots(figsize=(12, max(4, n * 0.38)))
        ax.axis("off")

        col_labels = ["Date", "Home Team", "Away Team", "Actual", "Predicted", "Correct?"]
        table_data = []
        cell_colors = []

        for _, r in page_df.iterrows():
            correct_str = "Yes" if r["correct"] else "No"
            table_data.append([
                r["_date"].strftime("%Y-%m-%d"),
                r["_home_team"],
                r["_away_team"],
                r["FTR"],
                r["predicted"],
                correct_str,
            ])
            row_color = "#d4edda" if r["correct"] else "#f8d7da"  # green / red
            cell_colors.append([row_color] * 6)

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)

        # Bold header
        for col_idx in range(len(col_labels)):
            tbl[0, col_idx].set_facecolor("#343a40")
            tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

        suffix = f"_p{page_num+1}" if len(pages) > 1 else ""
        title = f"2025/26 Season: Predicted vs Actual ({best_name})"
        if len(pages) > 1:
            title += f"  [{page_num+1}/{len(pages)}]"
        correct_count = page_df["correct"].sum()
        title += f"\nPage accuracy: {correct_count}/{n} correct"
        ax.set_title(title, fontsize=10, pad=12)

        plt.tight_layout()
        plt.savefig(f"../output/figures/matchup_predictions{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

    total_correct = season_df["correct"].sum()
    print(f"Saved matchup_predictions.png  ({total_correct}/{len(season_df)} correct on 2025/26 matches)")
else:
    print("No 2025/26 season matches in test set — adjust split or add more recent data")
