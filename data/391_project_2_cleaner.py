# Running This File Gives Two Output Files:
# matches_cleaned.csv (one row per match) for classifications and
# teams_cleaned.csv (one row per team) for clustering.

import pandas as pd
import numpy as np
# This Is For Normalization Near Bottom, Not Sure If Data Better With Or Without Normalization,
# So Ran Twice One With And One Without.
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------------
# Loading The Data
df = pd.read_csv("SP1.csv", encoding="latin-1")

print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# Get Rid Of Unecessary Columns Like Betting Ones.
keep_cols = [
    # Identifiers
    "Div", "Date", "HomeTeam", "AwayTeam",
    # Match Result
    "FTHG", "FTAG", "FTR",          # Full-time goals (home/away) & result
    "HTHG", "HTAG", "HTR",          # Half-time goals & result (useful context)
    # Stats For Classification
    "HS",  "AS",                     # Shots
    "HST", "AST",                    # Shots on target
    "HC",  "AC",                     # Corners
    "HF",  "AF",                     # Fouls
    "HY",  "AY",                     # Yellow cards
    "HR",  "AR",                     # Red cards
    # Betting Odds (used to compute implied win probabilities in classification)
    "B365H", "B365D", "B365A",
]

keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Print Output After Column Selection.
print(f"After column selection: {df.shape[1]} columns kept")
print(f"Columns: {list(df.columns)}\n")
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# Drop The Fully Empty Rows.
before = len(df)
df.dropna(how="all", inplace=True)
print(f"Dropped {before - len(df)} fully empty rows.")


# ------------------------------------------------------------------------------------
# Fixing and Setting Data Types Like
# Dates Should Be In Proper Format And
# Specific Columns Like Stats and Goals Should Be Numeric.
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

numeric_cols = ["FTHG","FTAG","HTHG","HTAG",
                "HS","AS","HST","AST",
                "HC","AC","HF","AF",
                "HY","AY","HR","AR"]
numeric_cols = [c for c in numeric_cols if c in df.columns]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# Handle The Missing Values in The Data.
# If A Column Has A Missing Value It Will Fill It In With The Columns Median.
# If A Key Indetifier Is Missing Though It Just Drops It Since We Can't Guess/Estimate.
print("\nMissing values per column:")
missing = df[numeric_cols].isnull().sum()
print(missing[missing > 0] if missing.any() else "  None — all stat columns are complete")

# Fill Missing With Median.
for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' NaNs with median = {median_val:.1f}")

# Drop If Missing Key Indentifier.
key_cols = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTR"] if c in df.columns]
before = len(df)
df.dropna(subset=key_cols, inplace=True)
print(f"Dropped {before - len(df)} rows missing key identifiers.")


# ------------------------------------------------------------------------------------
# Remove Duplicate Matches.
before = len(df)
df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], inplace=True)
print(f"Removed {before - len(df)} duplicate rows.")
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Double Check The Data Makes Logical Sense, Like With
# No Stat Being Negative
# Shots On A Certain Area Can't Exceed Total Shots Taken
# FTR or "Full Time Result" Can Only Ever Be Home Win, Draw or Away Win.

for col in numeric_cols:
    neg = (df[col] < 0).sum()
    if neg > 0:
        print(f"WARNING: {neg} negative values in '{col}' — setting to 0.")
        df.loc[df[col] < 0, col] = 0

for on_target, total in [("HST","HS"), ("AST","AS")]:
    if on_target in df.columns and total in df.columns:
        bad = df[on_target] > df[total]
        if bad.any():
            print(f"WARNING: {bad.sum()} rows where {on_target} > {total} — capping.")
            df.loc[bad, on_target] = df.loc[bad, total]

if "FTR" in df.columns:
    bad_ftr = ~df["FTR"].isin({"H","D","A"})
    if bad_ftr.any():
        print(f"WARNING: {bad_ftr.sum()} rows with invalid FTR — dropping.")
        df = df[~bad_ftr]
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# Remove Leading or Trailing Spaces In Team Names.
for col in ["HomeTeam", "AwayTeam"]:
    if col in df.columns:
        df[col] = df[col].str.strip()
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Sort Chronologically Based On Date from Earliest To Most Recent.
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# Create Our Derived Variable Columns For Classification
df["shot_diff"] = df["HS"] - df["AS"]
df["on_target_diff"] = df["HST"] - df["AST"]
df["corner_diff"] = df["HC"] - df["AC"]
df["foul_diff"] = df["HF"] - df["AF"]
df["yellow_diff"] = df["HY"] - df["AY"]
df["red_diff"] = df["HR"] - df["AR"]
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# Save/Create matches_cleaned.csv
df.to_csv("matches_cleaned.csv", index=False)
print("\nCreated matches_cleaned.csv")
# ------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------
# Building The Team Level Aggregates For Clustering.
all_teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
records = []

for team in all_teams:
    home = df[df["HomeTeam"] == team]
    away = df[df["AwayTeam"] == team]
    total_games = len(home) + len(away)

    if total_games == 0:
        continue

    def team_avg(home_col, away_col):
        vals = list(home[home_col]) + list(away[away_col])
        return round(np.mean(vals), 4) if vals else np.nan

    avg_goals_scored   = team_avg("FTHG", "FTAG")
    avg_goals_conceded = team_avg("FTAG", "FTHG")
    avg_shots           = team_avg("HS", "AS")
    avg_shots_on_target = team_avg("HST", "AST")
    avg_corners         = team_avg("HC", "AC")
    avg_fouls           = team_avg("HF", "AF")
    avg_yellow_cards    = team_avg("HY", "AY")
    avg_red_cards       = team_avg("HR", "AR")
    home_wins        = (home["FTR"] == "H").sum()
    away_wins        = (away["FTR"] == "A").sum()
    home_win_rate    = round(home_wins / len(home), 4) if len(home) > 0 else np.nan
    overall_win_rate = round((home_wins + away_wins) / total_games, 4)

    records.append({
        "team":                 team,
        "total_games":          total_games,
        "avg_goals_scored":     avg_goals_scored,
        "avg_goals_conceded":   avg_goals_conceded,
        "avg_shots":            avg_shots,
        "avg_shots_on_target":  avg_shots_on_target,
        "avg_corners":          avg_corners,
        "avg_fouls":            avg_fouls,
        "avg_yellow_cards":     avg_yellow_cards,
        "avg_red_cards":        avg_red_cards,
        "home_win_rate":        home_win_rate,
        "overall_win_rate":     overall_win_rate,
    })
# ------------------------------------------------------------------------------------


team_df = pd.DataFrame(records)

"""
Commented Out Normalization For Now Since Values Look All Messed Up When Used.
# ------------------------------------------------------------------------------------
# Normalization / Scaling For Clustered Data.
# Without Scaling Some Data Would Have Much Smaller Values.
scale_cols = [
    "avg_goals_scored", "avg_goals_conceded", "avg_shots",
    "avg_shots_on_target", "avg_corners", "avg_fouls",
    "avg_yellow_cards", "avg_red_cards", "home_win_rate", "overall_win_rate"
]

scaler = StandardScaler()
team_df_scaled = team_df.copy()
team_df_scaled[scale_cols] = scaler.fit_transform(team_df[scale_cols])
team_df_scaled.to_csv("teams_cleaned_scaled.csv", index=False)
print("Created teams_cleaned_scaled.csv")
# ------------------------------------------------------------------------------------
"""

# ------------------------------------------------------------------------------------
# Save/Create teams_cleaned.csv
team_df.to_csv("teams_cleaned.csv", index=False)
print("Created teams_cleaned.csv")
# ------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------
# Summary Print Out Screen
print("\n--------------------------------------------")
print("              CLEANING SUMMARY")
print("----------------------------------------------")
print(f"  Matches (cleaned):  {len(df)}")
print(f"  Teams:              {len(team_df)}")
if "Date" in df.columns:
    print(f"  Season range:       {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"\n  matches_cleaned.csv  →  {len(df.columns)} columns, {len(df)} rows")
print(f"  teams_cleaned.csv   →  {len(team_df.columns)} columns, {len(team_df)} rows")
print("\n  Teams in dataset:")
for t in all_teams:
    print(f"    - {t}")
print("----------------------------------------------")
# ------------------------------------------------------------------------------------