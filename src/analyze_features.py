"""
Feature Analysis, Statistical Comparison, and Correlation Audit
===============================================================
Computes:
- results/features/feature_analysis.csv
- results/features/feature_correlation.csv
- results/features/redundant_pairs.csv
- results/features/plots/feature_comparisons.png
- results/features/plots/feature_correlation_heatmap.png
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

FEATURE_PATH = "data/processed/chess_behavioral_features.csv"
RESULTS_DIR = "results/features"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

def analyze():
    print("=== Loading Engineered Feature Dataset ===")
    df = pd.read_csv(FEATURE_PATH)
    print(f"Loaded {len(df):,d} rows and {df.shape[1]} columns.")
    
    predictive_features = [
        "relative_move_time", "time_spent_ratio", "is_premove", "tank_move_indicator",
        "clock_remaining_ratio", "player_rating_normalized", "opponent_rating_difference",
        "move_number_normalized", "move_time_mean_10", "move_time_std_10", "move_time_cv_10",
        "premove_rate_10", "tank_move_count_10", "move_time_skewness_10", "consecutive_fast_moves",
        "in_time_pressure", "time_pressure_jitter", "cumulative_mean_move_time",
        "cumulative_move_time_std", "cumulative_premove_rate", "phase_deliberation_ratio",
        "is_endgame", "current_endgame_clock_ratio",
        "game_phase_opening", "game_phase_middlegame", "game_phase_endgame",
        "speed_is_bullet", "speed_is_blitz", "speed_is_rapid", "speed_is_classical"
    ]
    
    human_mask = (df["label"] == "HUMAN")
    ai_mask = (df["label"] == "AI")
    
    # -------------------------------------------------------------
    # 1. Feature Analysis: Human vs AI Separation
    # -------------------------------------------------------------
    analysis_records = []
    for col in predictive_features:
        s = df[col]
        h_vals = df.loc[human_mask, col]
        a_vals = df.loc[ai_mask, col]
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(h_vals, a_vals, alternative='two-sided')
        n1, n2 = len(h_vals), len(a_vals)
        r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)
        
        # Cohen's d
        s_pooled = np.sqrt(((n1 - 1) * h_vals.var() + (n2 - 1) * a_vals.var()) / (n1 + n2 - 2))
        cohens_d = (h_vals.mean() - a_vals.mean()) / s_pooled if s_pooled > 0 else 0.0
        
        analysis_records.append({
            "feature": col,
            "overall_mean": round(float(s.mean()), 4),
            "overall_median": round(float(s.median()), 4),
            "overall_std": round(float(s.std()), 4),
            "human_mean": round(float(h_vals.mean()), 4),
            "human_median": round(float(h_vals.median()), 4),
            "human_std": round(float(h_vals.std()), 4),
            "ai_mean": round(float(a_vals.mean()), 4),
            "ai_median": round(float(a_vals.median()), 4),
            "ai_std": round(float(a_vals.std()), 4),
            "abs_cohens_d": round(abs(float(cohens_d)), 4),
            "cohens_d": round(float(cohens_d), 4),
            "rank_biserial_r": round(float(r_rb), 4),
            "p_value": float(p_val)
        })
        
    df_analysis = pd.DataFrame(analysis_records)
    # Sort by absolute separation effect size
    df_analysis.sort_values(by="abs_cohens_d", ascending=False, inplace=True)
    analysis_path = os.path.join(RESULTS_DIR, "feature_analysis.csv")
    df_analysis.to_csv(analysis_path, index=False)
    print(f"Saved feature analysis to {analysis_path}")
    
    print("\nTop 10 Features by Cohen's d separation:")
    for idx, row in df_analysis.head(10).iterrows():
        print(f"  {row['feature']:30s}: d={row['cohens_d']:+.4f} | Human={row['human_mean']:.3f} | AI={row['ai_mean']:.3f}")
        
    # -------------------------------------------------------------
    # 2. Correlation Analysis & Redundancy Detection
    # -------------------------------------------------------------
    corr_matrix = df[predictive_features].corr(method="spearman")
    corr_path = os.path.join(RESULTS_DIR, "feature_correlation.csv")
    corr_matrix.to_csv(corr_path)
    print(f"Saved feature correlation matrix to {corr_path}")
    
    # Identify redundant pairs (|r| > 0.70)
    redundant_pairs = []
    cols = list(predictive_features)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            r = corr_matrix.loc[c1, c2]
            if abs(r) >= 0.70:
                redundant_pairs.append({
                    "feature_a": c1,
                    "feature_b": c2,
                    "spearman_r": round(float(r), 4),
                    "interpretation": f"High collinearity between {c1} and {c2}",
                    "recommendation": "Retain feature with higher Cohen's d separation; evaluate tree feature importance before final pruning"
                })
                
    df_red = pd.DataFrame(redundant_pairs)
    df_red.sort_values(by="spearman_r", key=abs, ascending=False, inplace=True)
    red_path = os.path.join(RESULTS_DIR, "redundant_pairs.csv")
    df_red.to_csv(red_path, index=False)
    print(f"Saved redundant pairs ({len(df_red)} pairs) to {red_path}")
    
    # -------------------------------------------------------------
    # 3. Visualization: Correlation Heatmap & Feature Separation Plots
    # -------------------------------------------------------------
    print("Generating figures...")
    # Plot A: Correlation Heatmap of top 15 features
    top_15_features = list(df_analysis["feature"].head(15))
    top_corr = df[top_15_features].corr(method="spearman")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(top_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Spearman Correlation'})
    ax.set_title("Spearman Correlation Matrix of Top 15 Behavioral Features", fontsize=13, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, "feature_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved {heatmap_path}")
    
    # Plot B: Feature Comparison Boxplots for Top 6 Non-Rating Behavioral Features
    # Exclude rating features to focus purely on behavioral timing
    behavioral_top = [f for f in df_analysis["feature"] if "rating" not in f][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    
    for i, feat in enumerate(behavioral_top):
        # Clip top 1% for cleaner visualization
        q99 = df[feat].quantile(0.99)
        plot_data = df[df[feat] <= q99] if q99 > 0 else df
        sns.boxplot(x="label", y=feat, data=plot_data, ax=axes[i], palette=["#1d3557", "#e63946"], hue="label", legend=False)
        d_val = df_analysis[df_analysis["feature"] == feat]["cohens_d"].values[0]
        axes[i].set_title(f"{feat}\n(Cohen's d = {d_val:+.3f})", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Feature Value")
        
    plt.suptitle("Human vs. AI Separation: Top Behavioral Timing Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    comp_plot_path = os.path.join(PLOTS_DIR, "feature_comparisons.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved {comp_plot_path}")
    print("Feature analysis completed successfully.")

if __name__ == "__main__":
    analyze()
