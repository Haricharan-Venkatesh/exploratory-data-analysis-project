"""
Comprehensive Exploratory Data Analysis (EDA) Script
=====================================================
Analyzes clean_chess_moves.csv to discover genuine behavioral differences
between HUMAN and AI chess players. Generates statistical tables, tests,
and publication-grade figures.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_PATH = "data/processed/clean_chess_moves.csv"
PLOTS_DIR = "results/eda/plots"
STATS_DIR = "results/eda/statistics"
TABLES_DIR = "results/eda/tables"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

def run_eda():
    print("=== Loading Clean Chess Move Dataset ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,d} rows and {df.shape[1]} columns.")
    
    # 1. Basic Verification
    verification_stats = {
        "total_moves": len(df),
        "total_columns": df.shape[1],
        "columns": list(df.columns),
        "human_moves": int((df["label"] == "HUMAN").sum()),
        "ai_moves": int((df["label"] == "AI").sum()),
        "human_move_pct": round(float((df["label"] == "HUMAN").mean() * 100), 2),
        "ai_move_pct": round(float((df["label"] == "AI").mean() * 100), 2),
        "unique_games": int(df["game_id"].nunique()),
        "unique_players": int(df["player_id"].nunique()),
        "missing_move_times": int(df["move_time"].isnull().sum()),
        "missing_move_time_pct": round(float(df["move_time"].isnull().mean() * 100), 4)
    }
    with open(os.path.join(STATS_DIR, "data_verification.json"), "w", encoding="utf-8") as f:
        json.dump(verification_stats, f, indent=2)
    print("Saved verification stats.")
    
    # Non-null subset for move_time analyses
    df_valid = df.dropna(subset=["move_time"]).copy()
    
    # 2. Univariate Numerical Statistics
    num_cols = ["move_time", "player_rating", "move_number", "clock_after_move", "opponent_rating"]
    num_stats_list = []
    for col in num_cols:
        s = df_valid[col]
        num_stats_list.append({
            "feature": col,
            "count": len(s),
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "p05": round(float(s.quantile(0.05)), 2),
            "p10": round(float(s.quantile(0.10)), 2),
            "q25": round(float(s.quantile(0.25)), 2),
            "median": round(float(s.median()), 2),
            "q75": round(float(s.quantile(0.75)), 2),
            "p90": round(float(s.quantile(0.90)), 2),
            "p95": round(float(s.quantile(0.95)), 2),
            "p99": round(float(s.quantile(0.99)), 2),
            "max": round(float(s.max()), 2),
            "skew": round(float(s.skew()), 2),
            "kurtosis": round(float(s.kurtosis()), 2)
        })
    df_num_stats = pd.DataFrame(num_stats_list)
    df_num_stats.to_csv(os.path.join(STATS_DIR, "univariate_numerical.csv"), index=False)
    print("Saved univariate numerical statistics.")
    
    # 3. Univariate Categorical Statistics
    cat_cols = ["game_phase", "speed_category", "game_result", "player_color"]
    cat_stats_list = []
    for col in cat_cols:
        counts = df[col].value_counts()
        for val, count in counts.items():
            cat_stats_list.append({
                "feature": col,
                "category": val,
                "frequency": int(count),
                "percentage": round(float(count / len(df) * 100), 2)
            })
    df_cat_stats = pd.DataFrame(cat_stats_list)
    df_cat_stats.to_csv(os.path.join(STATS_DIR, "univariate_categorical.csv"), index=False)
    print("Saved univariate categorical statistics.")
    
    # Plot 1: Univariate Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Move Time (log scale due to right skew)
    sns.histplot(df_valid["move_time"], bins=60, kde=False, ax=axes[0, 0], color="#2b5c8f")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("A. Move Time Distribution (Log Count Scale)")
    axes[0, 0].set_xlabel("Move Time (seconds)")
    axes[0, 0].set_ylabel("Log(Count)")
    
    # Player Rating
    sns.histplot(df["player_rating"], bins=40, kde=True, ax=axes[0, 1], color="#3f8f6b")
    axes[0, 1].set_title("B. Player Elo Rating Distribution")
    axes[0, 1].set_xlabel("Elo Rating")
    axes[0, 1].set_ylabel("Move Count")
    
    # Game Phase
    phase_order = ["Opening", "Middlegame", "Endgame"]
    phase_counts = df["game_phase"].value_counts()[phase_order]
    axes[1, 0].bar(phase_order, phase_counts, color=["#e07a5f", "#3d5a80", "#81b29a"])
    for i, v in enumerate(phase_counts):
        axes[1, 0].text(i, v + 3000, f"{v:,}\n({v/len(df)*100:.1f}%)", ha='center', fontsize=9)
    axes[1, 0].set_title("C. Game Phase Distribution")
    axes[1, 0].set_ylabel("Move Count")
    
    # Speed Category
    speed_order = ["Bullet", "Blitz", "Rapid", "Classical"]
    speed_counts = df["speed_category"].value_counts().reindex(speed_order, fill_value=0)
    axes[1, 1].bar(speed_order, speed_counts, color=["#d62828", "#f77f00", "#003049", "#669bbc"])
    for i, v in enumerate(speed_counts):
        axes[1, 1].text(i, v + 3000, f"{v:,}\n({v/len(df)*100:.1f}%)", ha='center', fontsize=9)
    axes[1, 1].set_title("D. Speed Category Distribution")
    axes[1, 1].set_ylabel("Move Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_univariate_distributions.png"), dpi=300)
    plt.close()
    print("Saved Plot 01.")

    # 4. Human vs AI Comparison Table
    human_mask = (df_valid["label"] == "HUMAN")
    ai_mask = (df_valid["label"] == "AI")
    
    comparison_metrics = []
    comp_features = ["move_time", "player_rating", "clock_after_move", "move_number", "opponent_rating"]
    for col in comp_features:
        h_vals = df_valid.loc[human_mask, col]
        a_vals = df_valid.loc[ai_mask, col]
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(h_vals, a_vals, alternative='two-sided')
        # Effect size: Rank-Biserial correlation r_rb = 1 - (2*U)/(n1*n2)
        n1, n2 = len(h_vals), len(a_vals)
        r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)
        # Cohen's d
        s_pooled = np.sqrt(((n1 - 1) * h_vals.var() + (n2 - 1) * a_vals.var()) / (n1 + n2 - 2))
        cohens_d = (h_vals.mean() - a_vals.mean()) / s_pooled if s_pooled > 0 else 0.0
        
        comparison_metrics.append({
            "feature": col,
            "human_mean": round(float(h_vals.mean()), 2),
            "human_median": round(float(h_vals.median()), 2),
            "human_std": round(float(h_vals.std()), 2),
            "human_iqr": round(float(h_vals.quantile(0.75) - h_vals.quantile(0.25)), 2),
            "ai_mean": round(float(a_vals.mean()), 2),
            "ai_median": round(float(a_vals.median()), 2),
            "ai_std": round(float(a_vals.std()), 2),
            "ai_iqr": round(float(a_vals.quantile(0.75) - a_vals.quantile(0.25)), 2),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": round(float(r_rb), 4),
            "cohens_d": round(float(cohens_d), 4)
        })
    df_comp = pd.DataFrame(comparison_metrics)
    df_comp.to_csv(os.path.join(TABLES_DIR, "human_vs_ai_comparison.csv"), index=False)
    print("Saved Human vs AI comparison table.")

    # 5. Move-Time Detailed Analysis
    h_mt = df_valid.loc[human_mask, "move_time"]
    a_mt = df_valid.loc[ai_mask, "move_time"]
    
    h_zero_pct = float((h_mt == 0.0).mean() * 100)
    a_zero_pct = float((a_mt == 0.0).mean() * 100)
    h_gt10_pct = float((h_mt > 10.0).mean() * 100)
    a_gt10_pct = float((a_mt > 10.0).mean() * 100)
    h_gt20_pct = float((h_mt > 20.0).mean() * 100)
    a_gt20_pct = float((a_mt > 20.0).mean() * 100)
    h_gt30_pct = float((h_mt > 30.0).mean() * 100)
    a_gt30_pct = float((a_mt > 30.0).mean() * 100)
    
    move_time_deep_dive = {
        "human": {
            "count": len(h_mt),
            "mean": round(float(h_mt.mean()), 3),
            "median": round(float(h_mt.median()), 3),
            "std": round(float(h_mt.std()), 3),
            "cv": round(float(h_mt.std() / h_mt.mean()), 3),
            "zero_move_pct": round(h_zero_pct, 2),
            "gt_10s_pct": round(h_gt10_pct, 2),
            "gt_20s_pct": round(h_gt20_pct, 2),
            "gt_30s_pct": round(h_gt30_pct, 2),
            "p25": round(float(h_mt.quantile(0.25)), 2),
            "p75": round(float(h_mt.quantile(0.75)), 2),
            "p95": round(float(h_mt.quantile(0.95)), 2),
            "p99": round(float(h_mt.quantile(0.99)), 2),
            "max": round(float(h_mt.max()), 2)
        },
        "ai": {
            "count": len(a_mt),
            "mean": round(float(a_mt.mean()), 3),
            "median": round(float(a_mt.median()), 3),
            "std": round(float(a_mt.std()), 3),
            "cv": round(float(a_mt.std() / a_mt.mean()), 3),
            "zero_move_pct": round(a_zero_pct, 2),
            "gt_10s_pct": round(a_gt10_pct, 2),
            "gt_20s_pct": round(a_gt20_pct, 2),
            "gt_30s_pct": round(a_gt30_pct, 2),
            "p25": round(float(a_mt.quantile(0.25)), 2),
            "p75": round(float(a_mt.quantile(0.75)), 2),
            "p95": round(float(a_mt.quantile(0.95)), 2),
            "p99": round(float(a_mt.quantile(0.99)), 2),
            "max": round(float(a_mt.max()), 2)
        }
    }
    with open(os.path.join(STATS_DIR, "move_time_deep_dive.json"), "w", encoding="utf-8") as f:
        json.dump(move_time_deep_dive, f, indent=2)
    print("Saved move time deep dive JSON.")
    
    # Plot 2: Move Time Distribution (KDE and Boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Boxplot clamped to 30s for visibility
    sns.boxplot(x="label", y="move_time", data=df_valid[df_valid["move_time"] <= 25], ax=axes[0], palette=["#1d3557", "#e63946"])
    axes[0].set_title("A. Move Time Boxplot (Clamped to <= 25s)")
    axes[0].set_ylabel("Move Time (s)")
    axes[0].set_xlabel("Player Label")
    
    # Log-transformed density
    df_valid["log_move_time"] = np.log1p(df_valid["move_time"])
    sns.kdeplot(data=df_valid, x="log_move_time", hue="label", common_norm=False, fill=True, alpha=0.3, palette=["#1d3557", "#e63946"], ax=axes[1])
    axes[1].set_title("B. Density of Log(Move Time + 1)")
    axes[1].set_xlabel("log(1 + move_time)")
    axes[1].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_human_vs_ai_move_time_dist.png"), dpi=300)
    plt.close()
    print("Saved Plot 02.")
    
    # Plot 3: Zero-Move Pre-move Rates
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["HUMAN", "AI"], [h_zero_pct, a_zero_pct], color=["#1d3557", "#e63946"], width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel("Percentage of Moves with 0.0s Elapsed (%)")
    ax.set_title("Zero-Time / Instant Pre-Move Rate: Human vs AI")
    ax.set_ylim(0, max(h_zero_pct, a_zero_pct) + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_human_vs_ai_zero_moves.png"), dpi=300)
    plt.close()
    print("Saved Plot 03.")
    
    # 6. Progression Across Game (Move Number)
    # Aggregate move time by move number (up to move 50)
    move_prog = df_valid[df_valid["move_number"] <= 50].groupby(["move_number", "label"])["move_time"].agg(["mean", "median"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(data=move_prog, x="move_number", y="mean", hue="label", marker="o", ax=axes[0], palette=["#1d3557", "#e63946"])
    axes[0].set_title("A. Mean Move Time Across Game Progression (Moves 1-50)")
    axes[0].set_xlabel("Move Number")
    axes[0].set_ylabel("Mean Move Time (s)")
    
    sns.lineplot(data=move_prog, x="move_number", y="median", hue="label", marker="s", ax=axes[1], palette=["#1d3557", "#e63946"])
    axes[1].set_title("B. Median Move Time Across Game Progression (Moves 1-50)")
    axes[1].set_xlabel("Move Number")
    axes[1].set_ylabel("Median Move Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_move_time_progression.png"), dpi=300)
    plt.close()
    print("Saved Plot 04.")

    # 7. Player-Level Aggregation & Variability (CV)
    print("Calculating player-level aggregations...")
    player_groups = df_valid.groupby(["player_id", "label"]).agg(
        total_moves=("move_time", "count"),
        mean_move_time=("move_time", "mean"),
        median_move_time=("move_time", "median"),
        std_move_time=("move_time", "std"),
        zero_move_rate=("move_time", lambda s: (s == 0.0).mean() * 100),
        avg_rating=("player_rating", "mean")
    ).reset_index()
    # Require at least 15 moves for stable variance estimates
    player_active = player_groups[player_groups["total_moves"] >= 15].copy()
    player_active["cv"] = player_active["std_move_time"] / player_active["mean_move_time"]
    
    player_active.to_csv(os.path.join(TABLES_DIR, "player_level_behavior.csv"), index=False)
    
    # Plot 5: Player Variability
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x="label", y="cv", data=player_active, ax=axes[0], palette=["#1d3557", "#e63946"])
    axes[0].set_title("A. Move-Time Coefficient of Variation (CV) per Player (>=15 moves)")
    axes[0].set_ylabel("CV = Std / Mean")
    axes[0].set_xlabel("Player Label")
    
    sns.boxplot(x="label", y="zero_move_rate", data=player_active, ax=axes[1], palette=["#1d3557", "#e63946"])
    axes[1].set_title("B. Zero-Move Rate (%) per Player")
    axes[1].set_ylabel("Zero-Move %")
    axes[1].set_xlabel("Player Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_player_variability_cv.png"), dpi=300)
    plt.close()
    print("Saved Plot 05.")

    # 8. Game Phase Behavior
    phase_stats = df_valid.groupby(["game_phase", "label"]).agg(
        move_count=("move_time", "count"),
        mean_time=("move_time", "mean"),
        median_time=("move_time", "median"),
        std_time=("move_time", "std"),
        zero_pct=("move_time", lambda s: (s == 0.0).mean() * 100),
        mean_clock=("clock_after_move", "mean")
    ).reset_index()
    phase_stats.to_csv(os.path.join(TABLES_DIR, "game_phase_behavior.csv"), index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=phase_stats, x="game_phase", y="mean_time", hue="label", ax=axes[0], palette=["#1d3557", "#e63946"], order=["Opening", "Middlegame", "Endgame"])
    axes[0].set_title("A. Mean Move Time by Game Phase")
    axes[0].set_ylabel("Mean Move Time (s)")
    axes[0].set_xlabel("Game Phase")
    
    sns.barplot(data=phase_stats, x="game_phase", y="zero_pct", hue="label", ax=axes[1], palette=["#1d3557", "#e63946"], order=["Opening", "Middlegame", "Endgame"])
    axes[1].set_title("B. Zero-Move / Pre-Move Rate by Game Phase")
    axes[1].set_ylabel("Zero-Move Rate (%)")
    axes[1].set_xlabel("Game Phase")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_game_phase_behavior.png"), dpi=300)
    plt.close()
    print("Saved Plot 06.")

    # 9. Speed Category Analysis
    speed_stats = df_valid.groupby(["speed_category", "label"]).agg(
        move_count=("move_time", "count"),
        mean_time=("move_time", "mean"),
        median_time=("move_time", "median"),
        std_time=("move_time", "std"),
        zero_pct=("move_time", lambda s: (s == 0.0).mean() * 100),
        mean_rating=("player_rating", "mean")
    ).reset_index()
    speed_stats.to_csv(os.path.join(TABLES_DIR, "speed_category_behavior.csv"), index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    order_sp = ["Bullet", "Blitz", "Rapid", "Classical"]
    sp_present = [s for s in order_sp if s in df_valid["speed_category"].unique()]
    sns.barplot(data=speed_stats, x="speed_category", y="mean_time", hue="label", ax=axes[0], palette=["#1d3557", "#e63946"], order=sp_present)
    axes[0].set_title("A. Mean Move Time by Speed Category")
    axes[0].set_ylabel("Mean Move Time (s)")
    axes[0].set_xlabel("Speed Category")
    
    sns.barplot(data=speed_stats, x="speed_category", y="median_time", hue="label", ax=axes[1], palette=["#1d3557", "#e63946"], order=sp_present)
    axes[1].set_title("B. Median Move Time by Speed Category")
    axes[1].set_ylabel("Median Move Time (s)")
    axes[1].set_xlabel("Speed Category")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_speed_category_comparison.png"), dpi=300)
    plt.close()
    print("Saved Plot 07.")

    # 10. Time Control Normalization
    # Compute normalized pacing features
    # Available budget per move = remaining clock / estimated remaining moves (~30)
    df_valid["time_spent_ratio"] = df_valid["move_time"] / np.maximum(df_valid["clock_after_move"] + df_valid["move_time"], 1.0)
    # Parse base time from time_control string
    df_valid["base_time"] = df_valid["time_control"].apply(lambda tc: int(tc.split("+")[0]) if "+" in tc else 180)
    df_valid["relative_move_time"] = df_valid["move_time"] / np.maximum(df_valid["base_time"] / 40.0, 1.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x="label", y="time_spent_ratio", data=df_valid[df_valid["time_spent_ratio"] <= 0.25], ax=axes[0], palette=["#1d3557", "#e63946"])
    axes[0].set_title("A. Ratio of Available Clock Spent per Move (Clamped <= 0.25)")
    axes[0].set_ylabel("Move Time / (Clock Before Move)")
    axes[0].set_xlabel("Player Label")
    
    sns.boxplot(x="label", y="relative_move_time", data=df_valid[df_valid["relative_move_time"] <= 5.0], ax=axes[1], palette=["#1d3557", "#e63946"])
    axes[1].set_title("B. Relative Move Time (Move Time / [Base Time / 40])")
    axes[1].set_ylabel("Normalized Move Time Units")
    axes[1].set_xlabel("Player Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_time_control_normalization.png"), dpi=300)
    plt.close()
    print("Saved Plot 08.")

    # 11. Paired Human vs Bot Matches (68 games)
    # Identify games where both HUMAN and AI played
    game_labels = df_valid.groupby("game_id")["label"].unique()
    mixed_game_ids = [gid for gid, lbls in game_labels.items() if len(lbls) > 1]
    print(f"Number of paired HUMAN-vs-BOT games: {len(mixed_game_ids)}")
    
    df_mixed = df_valid[df_valid["game_id"].isin(mixed_game_ids)].copy()
    paired_comparison = df_mixed.groupby("label").agg(
        moves=("move_time", "count"),
        mean_move_time=("move_time", "mean"),
        median_move_time=("move_time", "median"),
        std_move_time=("move_time", "std"),
        zero_pct=("move_time", lambda s: (s == 0.0).mean() * 100),
        mean_rating=("player_rating", "mean"),
        mean_clock=("clock_after_move", "mean")
    ).reset_index()
    paired_comparison.to_csv(os.path.join(TABLES_DIR, "paired_human_bot_matches.csv"), index=False)
    
    # Paired game move progression
    paired_prog = df_mixed[df_mixed["move_number"] <= 40].groupby(["move_number", "label"])["move_time"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=paired_prog, x="move_number", y="move_time", hue="label", marker="o", ax=ax, palette=["#1d3557", "#e63946"])
    ax.set_title("Move-Time Pacing in Identical Paired Matches (68 Human-vs-Bot Games)")
    ax.set_xlabel("Move Number")
    ax.set_ylabel("Mean Move Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "09_paired_human_vs_bot_matches.png"), dpi=300)
    plt.close()
    print("Saved Plot 09.")

    # 12. Correlation Analysis
    corr_features = ["move_time", "player_rating", "opponent_rating", "move_number", "clock_after_move", "time_spent_ratio"]
    corr_pearson = df_valid[corr_features].corr(method="pearson")
    corr_spearman = df_valid[corr_features].corr(method="spearman")
    corr_pearson.to_csv(os.path.join(TABLES_DIR, "correlation_pearson.csv"))
    corr_spearman.to_csv(os.path.join(TABLES_DIR, "correlation_spearman.csv"))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("A. Pearson Correlation Matrix")
    
    sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("B. Spearman Rank Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "10_correlation_matrix.png"), dpi=300)
    plt.close()
    print("Saved Plot 10.")

    # 13. Statistical Summary & Effect Sizes Table
    tests_summary = []
    # Test 1: Move time
    h_time = df_valid.loc[human_mask, "move_time"]
    a_time = df_valid.loc[ai_mask, "move_time"]
    u_time, p_time = stats.mannwhitneyu(h_time, a_time)
    ks_time, p_ks = stats.ks_2samp(h_time, a_time)
    r_rb_time = 1.0 - (2.0 * u_time) / (len(h_time) * len(a_time))
    
    tests_summary.append({
        "comparison": "Overall Move Time (s)",
        "human_stat": f"mean={h_time.mean():.2f}, med={h_time.median():.2f}",
        "ai_stat": f"mean={a_time.mean():.2f}, med={a_time.median():.2f}",
        "test_name": "Mann-Whitney U & KS test",
        "test_stat": f"U={u_time:.2e}, KS={ks_time:.3f}",
        "p_value": f"{p_time:.2e}",
        "effect_size": f"r_rb={r_rb_time:.3f}",
        "practical_significance": "Statistically significant; bots have substantially lower median time and fewer long pauses"
    })
    
    # Test 2: Zero-move rate (Chi-square test of proportions)
    h_zeros = (h_time == 0.0).sum()
    a_zeros = (a_time == 0.0).sum()
    table = [[h_zeros, len(h_time) - h_zeros], [a_zeros, len(a_time) - a_zeros]]
    chi2, p_chi2, _, _ = stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (len(df_valid) * 1))
    
    tests_summary.append({
        "comparison": "Zero-Move / Pre-Move Rate",
        "human_stat": f"{h_zero_pct:.2f}%",
        "ai_stat": f"{a_zero_pct:.2f}%",
        "test_name": "Chi-Square Test of Independence",
        "test_stat": f"Chi2={chi2:.2f}",
        "p_value": f"{p_chi2:.2e}",
        "effect_size": f"Cramer's V={cramers_v:.4f}",
        "practical_significance": "Significant difference in instantaneous moves; human players premove at a much higher frequency in online play"
    })
    
    # Test 3: Player rating
    h_rat = df_valid.loc[human_mask, "player_rating"]
    a_rat = df_valid.loc[ai_mask, "player_rating"]
    u_rat, p_rat = stats.mannwhitneyu(h_rat, a_rat)
    r_rb_rat = 1.0 - (2.0 * u_rat) / (len(h_rat) * len(a_rat))
    tests_summary.append({
        "comparison": "Player Rating (Elo)",
        "human_stat": f"mean={h_rat.mean():.1f}, med={h_rat.median():.1f}",
        "ai_stat": f"mean={a_rat.mean():.1f}, med={a_rat.median():.1f}",
        "test_name": "Mann-Whitney U Test",
        "test_stat": f"U={u_rat:.2e}",
        "p_value": f"{p_rat:.2e}",
        "effect_size": f"r_rb={r_rb_rat:.3f}",
        "practical_significance": "Bots in our sample span active ratings with higher mean rating than casual pool"
    })
    
    df_tests = pd.DataFrame(tests_summary)
    df_tests.to_csv(os.path.join(TABLES_DIR, "statistical_tests_summary.csv"), index=False)
    print("Saved Statistical Tests Summary.")
    print("All EDA analyses and outputs generated successfully.")

if __name__ == "__main__":
    run_eda()
