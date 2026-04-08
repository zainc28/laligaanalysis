## Planning

# What it is 

Scrape data, clean it, and process it

Analyze it - association rules, clustering, or classification methods. 

Draw insight from data including as match prediciton, play style classificaiton, red card prediction 

Write a paper being thorough, organized and professional, should address methodology, findings and conclusion


# Analyzing
Option A: Classification — Predict Match Outcome
Target: FTR (H / D / A)
Features to engineer:

Rolling averages (last 5 games): goals scored, goals conceded, shots, corners
Home/away form: win rate at home vs away
Head-to-head history between teams
Rest days since last match (from Date column)
Betting odds as implied probability: 1 / B365H

Methods: Decision Tree, Random Forest, or Logistic Regression

Option B: Clustering — Group Teams by Style
Features per team (aggregated):

Avg shots per game, avg shots on target
Avg fouls committed, avg cards received
Avg corners, goal difference
Home vs away scoring differential

Methods: K-Means (try k=3–5), then interpret clusters as "attacking," "defensive," "balanced," etc.

Option C: Association Rules — What predicts high-card matches?
Discretize variables:

High fouls (>15), many cards (>3), high shots (>12)
Outcome: upset (favorite lost based on odds)

Use: Apriori algorithm to find rules like "High fouls + Away team → Red card likely"
