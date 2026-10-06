"""
Extensions for Game-Level, Time-Pressure, and Outlier Analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/clean_chess_moves.csv"
PLOTS_DIR = "results/eda/plots"
STATS_DIR = "results/eda/statistics"
TABLES_DIR = "results/eda/tables"

df = pd.read_csv(DATA_PATH)
df_valid = df.dropna(subset=["move_time"]).copy()

# 1. Game-Level Aggregation
print("Computing game-level aggregations...")
game_meta = df_valid.groupby("game_id").agg(
    total_plies=("ply", "count"),
    mean_move_time=("move_time", "mean"),
    std_move_time=("move_time", "std"),
    median_move_time=("move_time", "median"),
    zero_move_pct=("move_time", lambda s: (s == 0.0).mean() * 100),
    speed_category=("speed_category", "first"),
    time_control=("time_control", "first"),
    has_ai=("label", lambda s: (s == "AI").any()),
    has_human=("label", lambda s: (s == "HUMAN").any()),
    min_clock=("clock_after_move", "min"),
    final_clock=("clock_after_move", "last")
).reset_index()

def classify_game_type(row):
    if row["has_ai"] and row["has_human"]:
        return "HUMAN_vs_BOT"
    elif row["has_ai"]:
        return "BOT_vs_BOT"
    else:
        return "HUMAN_vs_HUMAN"

game_meta["game_type"] = game_meta.apply(classify_game_type, axis=1)

game_summary = game_meta.groupby("game_type").agg(
    games=("game_id", "count"),
    avg_plies=("total_plies", "mean"),
    median_plies=("total_plies", "median"),
    avg_game_move_time=("mean_move_time", "mean"),
    avg_game_move_std=("std_move_time", "mean"),
    avg_zero_move_pct=("zero_move_pct", "mean"),
    avg_min_clock=("min_clock", "mean")
).reset_index()

game_summary.to_csv(os.path.join(TABLES_DIR, "game_level_comparison.csv"), index=False)
print("Saved game_level_comparison.csv.")

# 2. Outlier Analysis
print("Analyzing outliers...")
# Outliers in move_time: > 60s, > 120s, > 300s
long_moves = df_valid[df_valid["move_time"] > 60].copy()
outlier_summary = {
    "move_time_gt_60s": {
        "total_count": len(long_moves),
        "human_count": int((long_moves["label"] == "HUMAN").sum()),
        "ai_count": int((long_moves["label"] == "AI").sum()),
        "human_pct_of_total": round(float((long_moves["label"] == "HUMAN").mean() * 100), 2),
        "phases": long_moves["game_phase"].value_counts().to_dict(),
        "speeds": long_moves["speed_category"].value_counts().to_dict(),
        "max_time_human": float(df_valid[df_valid["label"] == "HUMAN"]["move_time"].max()),
        "max_time_ai": float(df_valid[df_valid["label"] == "AI"]["move_time"].max())
    },
    "move_time_gt_120s": {
        "total_count": int((df_valid["move_time"] > 120).sum()),
        "human_count": int(((df_valid["move_time"] > 120) & (df_valid["label"] == "HUMAN")).sum()),
        "ai_count": int(((df_valid["move_time"] > 120) & (df_valid["label"] == "AI")).sum())
    }
}
with open(os.path.join(STATS_DIR, "outliers_summary.json"), "w", encoding="utf-8") as f:
    json.dump(outlier_summary, f, indent=2)
print("Saved outliers_summary.json.")

# 3. Time Pressure Analysis (Remaining Clock < 30s, < 15s, < 10s)
print("Analyzing time pressure behavior...")
clock_bins = [0, 10, 20, 30, 60, 120, 300, 10000]
clock_labels = ["<10s (Extreme)", "10-20s (High)", "20-30s (Moderate)", "30-60s", "1-2m", "2-5m", ">5m"]
df_valid["clock_bin"] = pd.cut(df_valid["clock_after_move"], bins=clock_bins, labels=clock_labels, right=False)

time_pressure = df_valid.groupby(["clock_bin", "label"], observed=False).agg(
    moves=("move_time", "count"),
    mean_move_time=("move_time", "mean"),
    median_move_time=("move_time", "median"),
    std_move_time=("move_time", "std"),
    zero_pct=("move_time", lambda s: (s == 0.0).mean() * 100)
).reset_index()

time_pressure.to_csv(os.path.join(TABLES_DIR, "time_pressure_analysis.csv"), index=False)
print("Saved time_pressure_analysis.csv.")

# Plot 11: Time Pressure Pacing
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.barplot(data=time_pressure, x="clock_bin", y="mean_move_time", hue="label", ax=axes[0], palette=["#1d3557", "#e63946"])
axes[0].set_title("A. Mean Move Time Across Clock Remaining Bins (Time Pressure)")
axes[0].set_xlabel("Remaining Clock Bank")
axes[0].set_ylabel("Mean Move Time (s)")
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(data=time_pressure, x="clock_bin", y="zero_pct", hue="label", ax=axes[1], palette=["#1d3557", "#e63946"])
axes[1].set_title("B. Pre-Move / Zero-Move Rate Across Clock Remaining Bins")
axes[1].set_xlabel("Remaining Clock Bank")
axes[1].set_ylabel("Zero-Move Rate (%)")
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_time_pressure_behavior.png"), dpi=300)
plt.close()
print("Saved Plot 11.")

print("Deep EDA extensions completed.")
