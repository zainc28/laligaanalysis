import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv("data/matches_cleaned.csv")

counts = df["FTR"].value_counts()
total = len(df)

labels = ["Home Win", "Draw", "Away Win"]
keys   = ["H", "D", "A"]
values = [counts[k] / total * 100 for k in keys]
colors = ["#2c5f8a", "#888888", "#c0392b"]

fig, ax = plt.subplots(figsize=(5.5, 3.5))

bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.6,
        f"{val:.1f}\\%",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

ax.set_ylabel("Proportion of Matches (\\%)", fontsize=10)
ax.set_title(
    f"La Liga Match Outcome Distribution (2018--2026, $n={total:,}$ matches)",
    fontsize=10
)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.set_ylim(0, 55)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=9)

plt.tight_layout()
plt.savefig("output/figures/outcome_distribution.pdf", bbox_inches="tight")
plt.savefig("output/figures/outcome_distribution.png", dpi=150, bbox_inches="tight")
print("Saved to output/figures/outcome_distribution.pdf and .png")
