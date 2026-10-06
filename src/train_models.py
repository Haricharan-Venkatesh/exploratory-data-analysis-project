"""
Model Training, Cross-Validation, and Evaluation Pipeline
=========================================================
Trains baseline models vs. EDA-driven behavioral models on:
- Experiment A: Unseen Game Split (Grouped by game_id)
- Experiment B: Unseen Player Split (Grouped by player_id)
- Experiment C: Paired Human-vs-Bot Match Holdout

Evaluates:
- Logistic Regression
- Random Forest
- LightGBM (Gradient Boosting)

Generates:
- results/model_comparison.csv
- results/feature_importance.csv
- results/ablation_results.csv
- results/confusion_matrix.png
- results/roc_curve.png
- results/precision_recall_curve.png
- results/calibration_curve.png
- models/best_model.pkl
- models/feature_columns.json
- models/model_metadata.json
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import shap

DATA_PATH = "data/processed/chess_behavioral_features.csv"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# -------------------------------------------------------------------
# 1. Feature Set Definitions
# -------------------------------------------------------------------
BASELINE_1_FEATURES = [
    "clean_move_time" if "clean_move_time" in [] else "move_time",
    "clock_remaining_ratio", "move_number_normalized",
    "speed_is_bullet", "speed_is_blitz", "speed_is_rapid", "speed_is_classical"
]

BASELINE_2_FEATURES = [
    "move_time", "player_rating_normalized", "opponent_rating_difference",
    "clock_remaining_ratio", "move_number_normalized",
    "game_phase_opening", "game_phase_middlegame", "game_phase_endgame",
    "speed_is_bullet", "speed_is_blitz", "speed_is_rapid", "speed_is_classical"
]

EDA_BEHAVIORAL_10 = [
    "consecutive_fast_moves", "premove_rate_10", "cumulative_premove_rate",
    "current_endgame_clock_ratio", "phase_deliberation_ratio", "move_time_cv_10",
    "time_pressure_jitter", "relative_move_time", "time_spent_ratio", "tank_move_count_10"
]

FULL_EDA_FEATURES = [
    "move_time", "relative_move_time", "time_spent_ratio", "is_premove", "tank_move_indicator",
    "clock_remaining_ratio", "player_rating_normalized", "opponent_rating_difference",
    "move_number_normalized", "move_time_mean_10", "move_time_std_10", "move_time_cv_10",
    "premove_rate_10", "tank_move_count_10", "move_time_skewness_10", "consecutive_fast_moves",
    "in_time_pressure", "time_pressure_jitter", "cumulative_mean_move_time",
    "cumulative_move_time_std", "cumulative_premove_rate", "phase_deliberation_ratio",
    "is_endgame", "current_endgame_clock_ratio",
    "game_phase_opening", "game_phase_middlegame", "game_phase_endgame",
    "speed_is_bullet", "speed_is_blitz", "speed_is_rapid", "speed_is_classical"
]

def evaluate_predictions(y_true, y_pred, y_prob):
    """Compute comprehensive classification metrics."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4)
    }

def run_pipeline():
    start_total = time.time()
    print("=== Loading Dataset ===")
    df = pd.read_csv(DATA_PATH)
    # Ensure move_time has no NaNs for baseline feature
    df["move_time"] = df["move_time"].fillna(0.0)
    print(f"Loaded {len(df):,d} moves, {df['game_id'].nunique():,d} games, {df['player_id'].nunique():,d} players.")
    
    # -------------------------------------------------------------
    # 2. Splitting Logic
    # -------------------------------------------------------------
    print("\n--- Constructing Leakage-Free Evaluation Splits ---")
    np.random.seed(42)
    
    # Experiment C: Hold out 14 paired Human-vs-Bot games (~20% of the 68 games)
    mixed_games = df.groupby("game_id")["target"].nunique()
    paired_gids = mixed_games[mixed_games > 1].index.tolist()
    print(f"Total paired Human-vs-Bot games available: {len(paired_gids)}")
    
    holdout_paired_gids = set(np.random.choice(paired_gids, size=14, replace=False))
    df_exp_c_test = df[df["game_id"].isin(holdout_paired_gids)].copy()
    print(f"Experiment C Holdout (Paired Matches): {len(df_exp_c_test):,d} moves ({len(holdout_paired_gids)} games)")
    
    # Remaining games for Train / Test split
    remaining_df = df[~df["game_id"].isin(holdout_paired_gids)].copy()
    
    # Experiment A: Stratified Unseen Game Split (80% train, 20% test)
    ai_games = remaining_df[remaining_df["target"] == 1]["game_id"].unique()
    human_games = remaining_df[~remaining_df["game_id"].isin(ai_games)]["game_id"].unique()
    
    test_ai_games = set(np.random.choice(ai_games, size=int(len(ai_games) * 0.20), replace=False))
    test_human_games = set(np.random.choice(human_games, size=int(len(human_games) * 0.20), replace=False))
    test_gids_a = test_ai_games.union(test_human_games)
    
    train_df_a = remaining_df[~remaining_df["game_id"].isin(test_gids_a)].copy()
    test_df_a = remaining_df[remaining_df["game_id"].isin(test_gids_a)].copy()
    
    print(f"Experiment A (Unseen Game): Train={len(train_df_a):,d} moves ({train_df_a['game_id'].nunique()} games) | Test={len(test_df_a):,d} moves ({test_df_a['game_id'].nunique()} games)")
    print(f"  Test AI moves: {(test_df_a['target']==1).sum():,d} ({(test_df_a['target']==1).mean()*100:.2f}%)")
    
    # Experiment B: Unseen Player Split (80% train players, 20% test players)
    unique_players = remaining_df["player_id"].unique()
    test_players = set(np.random.choice(unique_players, size=int(len(unique_players) * 0.20), replace=False))
    train_df_b = remaining_df[~remaining_df["player_id"].isin(test_players)].copy()
    test_df_b = remaining_df[remaining_df["player_id"].isin(test_players)].copy()
    print(f"Experiment B (Unseen Player): Train={len(train_df_b):,d} moves | Test={len(test_df_b):,d} moves ({len(test_players)} players)")
    
    # -------------------------------------------------------------
    # 3. Controlled Model Training & Comparisons (Experiment A)
    # -------------------------------------------------------------
    print("\n--- Training Progressive Models on Experiment A (Unseen Game) ---")
    feature_sets = {
        "Baseline 1 (Minimal Timing)": BASELINE_1_FEATURES,
        "Baseline 2 (Conventional Context)": BASELINE_2_FEATURES,
        "EDA Behavioral 10": EDA_BEHAVIORAL_10,
        "Full EDA Features": FULL_EDA_FEATURES
    }
    
    # Calculate scale_pos_weight for LightGBM
    pos_weight = (train_df_a["target"] == 0).sum() / (train_df_a["target"] == 1).sum()
    print(f"Class imbalance ratio (neg/pos): {pos_weight:.2f}")
    
    comparison_results = []
    trained_models = {}
    
    for fs_name, f_cols in feature_sets.items():
        print(f"\n>> Evaluating Feature Set: {fs_name} ({len(f_cols)} features)")
        X_train, y_train = train_df_a[f_cols], train_df_a["target"]
        X_test, y_test = test_df_a[f_cols], test_df_a["target"]
        
        # 1. Logistic Regression
        lr_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))
        ])
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
        metrics_lr = evaluate_predictions(y_test, y_pred_lr, y_prob_lr)
        metrics_lr.update({"model": "Logistic Regression", "feature_set": fs_name, "split_strategy": "Unseen Game (Exp A)"})
        comparison_results.append(metrics_lr)
        
        # 2. Random Forest (compact & fast)
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=15,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        metrics_rf = evaluate_predictions(y_test, y_pred_rf, y_prob_rf)
        metrics_rf.update({"model": "Random Forest", "feature_set": fs_name, "split_strategy": "Unseen Game (Exp A)"})
        comparison_results.append(metrics_rf)
        
        # 3. LightGBM (Gradient Boosting)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
            scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
        metrics_lgb = evaluate_predictions(y_test, y_pred_lgb, y_prob_lgb)
        metrics_lgb.update({"model": "LightGBM", "feature_set": fs_name, "split_strategy": "Unseen Game (Exp A)"})
        comparison_results.append(metrics_lgb)
        
        trained_models[(fs_name, "LightGBM")] = (lgb_model, y_test, y_pred_lgb, y_prob_lgb)
        trained_models[(fs_name, "Random Forest")] = (rf_model, y_test, y_pred_rf, y_prob_rf)
        trained_models[(fs_name, "Logistic Regression")] = (lr_model, y_test, y_pred_lr, y_prob_lr)
        
        print(f"  [LightGBM] ROC-AUC: {metrics_lgb['roc_auc']:.4f} | PR-AUC: {metrics_lgb['pr_auc']:.4f} | Balanced Acc: {metrics_lgb['balanced_accuracy']:.4f} | Recall: {metrics_lgb['recall']:.4f}")

    # -------------------------------------------------------------
    # 4. Generalization Tests: Exp B (Unseen Player) & Exp C (Paired Holdout)
    # -------------------------------------------------------------
    print("\n--- Testing Generalization on Exp B (Unseen Player) & Exp C (Paired Matches) ---")
    best_fs = "EDA Behavioral 10"
    best_cols = EDA_BEHAVIORAL_10
    
    # Experiment B with Best Model
    lgb_b = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_b.fit(train_df_b[best_cols], train_df_b["target"])
    y_test_b = test_df_b["target"]
    y_pred_b = lgb_b.predict(test_df_b[best_cols])
    y_prob_b = lgb_b.predict_proba(test_df_b[best_cols])[:, 1]
    metrics_b = evaluate_predictions(y_test_b, y_pred_b, y_prob_b)
    metrics_b.update({"model": "LightGBM", "feature_set": best_fs, "split_strategy": "Unseen Player (Exp B)"})
    comparison_results.append(metrics_b)
    print(f"Experiment B (Unseen Player) ROC-AUC: {metrics_b['roc_auc']:.4f} | PR-AUC: {metrics_b['pr_auc']:.4f}")
    
    # Experiment C with Best Model (Evaluated on Paired Match Holdout)
    best_lgb_a, _, _, _ = trained_models[(best_fs, "LightGBM")]
    y_test_c = df_exp_c_test["target"]
    y_pred_c = best_lgb_a.predict(df_exp_c_test[best_cols])
    y_prob_c = best_lgb_a.predict_proba(df_exp_c_test[best_cols])[:, 1]
    metrics_c = evaluate_predictions(y_test_c, y_pred_c, y_prob_c)
    metrics_c.update({"model": "LightGBM", "feature_set": best_fs, "split_strategy": "Paired Holdout (Exp C)"})
    comparison_results.append(metrics_c)
    print(f"Experiment C (Paired Match Holdout) ROC-AUC: {metrics_c['roc_auc']:.4f} | PR-AUC: {metrics_c['pr_auc']:.4f} | Balanced Acc: {metrics_c['balanced_accuracy']:.4f}")

    # Save Model Comparison Table
    df_comp = pd.DataFrame(comparison_results)
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(f"\nSaved model comparison table to {comp_path}")

    # -------------------------------------------------------------
    # 5. Group-Aware 5-Fold Cross Validation (Exp A Train)
    # -------------------------------------------------------------
    print("\n--- Running 5-Fold GroupKFold Cross-Validation on Training Games ---")
    gkf = GroupKFold(n_splits=5)
    cv_scores = {"roc_auc": [], "pr_auc": [], "balanced_accuracy": [], "f1": []}
    
    X_tr_full = train_df_a[best_cols].values
    y_tr_full = train_df_a["target"].values
    groups_tr = train_df_a["game_id"].values
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_tr_full, y_tr_full, groups=groups_tr)):
        model_fold = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=6, num_leaves=31,
            scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1
        )
        model_fold.fit(X_tr_full[trn_idx], y_tr_full[trn_idx])
        val_probs = model_fold.predict_proba(X_tr_full[val_idx])[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        
        cv_scores["roc_auc"].append(roc_auc_score(y_tr_full[val_idx], val_probs))
        cv_scores["pr_auc"].append(average_precision_score(y_tr_full[val_idx], val_probs))
        cv_scores["balanced_accuracy"].append(balanced_accuracy_score(y_tr_full[val_idx], val_preds))
        cv_scores["f1"].append(f1_score(y_tr_full[val_idx], val_preds, zero_division=0))
        
    print("5-Fold GroupKFold Results (EDA Behavioral 10 - LightGBM):")
    for k, v in cv_scores.items():
        print(f"  {k:20s}: Mean = {np.mean(v):.4f} +/- {np.std(v):.4f}")

    # -------------------------------------------------------------
    # 6. Ablation Study
    # -------------------------------------------------------------
    print("\n--- Running Behavioral Feature Ablation Study ---")
    ablation_groups = {
        "A. Timing-Only": ["move_time", "relative_move_time", "time_spent_ratio"],
        "B. Timing + Clock Behavior": [
            "move_time", "relative_move_time", "time_spent_ratio",
            "clock_remaining_ratio", "time_pressure_jitter", "in_time_pressure"
        ],
        "C. Timing + Phase Behavior": [
            "move_time", "relative_move_time", "time_spent_ratio",
            "phase_deliberation_ratio", "current_endgame_clock_ratio", "is_endgame",
            "game_phase_opening", "game_phase_middlegame", "game_phase_endgame"
        ],
        "D. Timing + Pre-move Behavior": [
            "move_time", "relative_move_time", "time_spent_ratio",
            "is_premove", "premove_rate_10", "cumulative_premove_rate", "consecutive_fast_moves"
        ],
        "E. EDA Behavioral 10 (Recommended)": EDA_BEHAVIORAL_10,
        "F. All 32 Features (Full Model)": FULL_EDA_FEATURES
    }
    
    ablation_results = []
    for ab_name, ab_cols in ablation_groups.items():
        model_ab = lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=6, num_leaves=31,
            scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1
        )
        model_ab.fit(train_df_a[ab_cols], train_df_a["target"])
        probs_ab = model_ab.predict_proba(test_df_a[ab_cols])[:, 1]
        preds_ab = model_ab.predict(test_df_a[ab_cols])
        
        m_ab = evaluate_predictions(test_df_a["target"], preds_ab, probs_ab)
        m_ab["feature_group"] = ab_name
        m_ab["num_features"] = len(ab_cols)
        ablation_results.append(m_ab)
        print(f"  {ab_name:35s}: ROC-AUC={m_ab['roc_auc']:.4f} | PR-AUC={m_ab['pr_auc']:.4f} | Balanced Acc={m_ab['balanced_accuracy']:.4f}")
        
    df_ab = pd.DataFrame(ablation_results)
    ab_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    df_ab.to_csv(ab_path, index=False)
    print(f"Saved ablation study results to {ab_path}")

    # -------------------------------------------------------------
    # 7. Feature Importance & SHAP Analysis (Best Model)
    # -------------------------------------------------------------
    print("\n--- Computing Feature Importance & SHAP Values ---")
    best_model, y_test_best, y_pred_best, y_prob_best = trained_models[("EDA Behavioral 10", "LightGBM")]
    
    # Built-in Split & Gain Importance
    gain_imp = best_model.booster_.feature_importance(importance_type="gain")
    split_imp = best_model.booster_.feature_importance(importance_type="split")
    
    df_imp = pd.DataFrame({
        "feature": best_cols,
        "importance_gain": gain_imp,
        "importance_split": split_imp,
        "normalized_gain": gain_imp / np.sum(gain_imp)
    }).sort_values(by="importance_gain", ascending=False)
    
    imp_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    df_imp.to_csv(imp_path, index=False)
    print(f"Saved feature importance table to {imp_path}")
    print("Top Behavioral Features by Gain Importance:")
    print(df_imp[["feature", "importance_gain", "normalized_gain"]].to_string(index=False))
    
    # Compute SHAP on a representative sample of test set (2,000 moves)
    print("Computing TreeSHAP values...")
    shap_sample = test_df_a[best_cols].sample(n=min(2000, len(test_df_a)), random_state=42)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(shap_sample)
    # Binary classification in LightGBM SHAP: index 1 is positive class (or 2D array)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # -------------------------------------------------------------
    # 8. Threshold Analysis
    # -------------------------------------------------------------
    print("\n--- Running Decision Threshold Analysis ---")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    th_records = []
    cm_base = confusion_matrix(y_test_best, (y_prob_best >= 0.5).astype(int))
    
    for th in thresholds:
        th_pred = (y_prob_best >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_best, th_pred).ravel()
        p = precision_score(y_test_best, th_pred, zero_division=0)
        r = recall_score(y_test_best, th_pred, zero_division=0)
        f1_val = f1_score(y_test_best, th_pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        th_records.append({
            "threshold": th,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1_val, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        })
    df_th = pd.DataFrame(th_records)
    print(df_th.to_string(index=False))

    # -------------------------------------------------------------
    # 9. Calibration & Brier Score
    # -------------------------------------------------------------
    brier = brier_score_loss(y_test_best, y_prob_best)
    prob_true, prob_pred = calibration_curve(y_test_best, y_prob_best, n_bins=10, strategy="uniform")
    print(f"\nModel Calibration Brier Score: {brier:.4f}")

    # -------------------------------------------------------------
    # 10. Publication-Grade Visualizations
    # -------------------------------------------------------------
    print("\n--- Generating Model Evaluation Figures ---")
    # Figure 1: Confusion Matrix at Operating Threshold (0.50)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test_best, y_pred_best)
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Pred HUMAN", "Pred AI"], yticklabels=["Actual HUMAN", "Actual AI"], ax=ax)
    ax.set_title("Confusion Matrix (EDA Behavioral Model - Unseen Game Test)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Figure 2: ROC Curve (Comparing Baseline 1 vs Baseline 2 vs EDA 10)
    fig, ax = plt.subplots(figsize=(7, 6))
    for (m_fs, m_name), (_, y_t, _, y_p) in trained_models.items():
        if m_name == "LightGBM":
            fpr, tpr, _ = roc_curve(y_t, y_p)
            auc_val = roc_auc_score(y_t, y_p)
            ax.plot(fpr, tpr, lw=2, label=f"{m_fs} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)
    plt.close()
    
    # Figure 3: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for (m_fs, m_name), (_, y_t, _, y_p) in trained_models.items():
        if m_name == "LightGBM":
            prec, rec, _ = precision_recall_curve(y_t, y_p)
            pr_auc_val = average_precision_score(y_t, y_p)
            ax.plot(rec, prec, lw=2, label=f"{m_fs} (PR-AUC = {pr_auc_val:.3f})")
    ax.set_title("Precision-Recall Curve Comparison (Imbalanced Target)", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=300)
    plt.close()
    
    # Figure 4: Calibration Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", lw=2, label=f"EDA Behavioral LightGBM (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_title("Probability Calibration Curve (Reliability Diagram)", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Empirical)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_curve.png"), dpi=300)
    plt.close()
    print("Saved all 4 evaluation plots to results/.")

    # -------------------------------------------------------------
    # 11. Error Analysis Data Extraction
    # -------------------------------------------------------------
    print("\n--- Extracting Error Analysis Diagnostic Table ---")
    test_eval = test_df_a.copy()
    test_eval["pred"] = y_pred_best
    test_eval["prob_ai"] = y_prob_best
    
    fps = test_eval[(test_eval["target"] == 0) & (test_eval["pred"] == 1)].copy()
    fns = test_eval[(test_eval["target"] == 1) & (test_eval["pred"] == 0)].copy()
    
    print(f"Total False Positives (Human misclassified as AI): {len(fps):,d}")
    print(f"Total False Negatives (AI misclassified as Human): {len(fns):,d}")
    
    error_summary = {
        "false_positives": {
            "count": len(fps),
            "mean_move_time": round(float(fps["move_time"].mean()), 3),
            "median_move_time": round(float(fps["move_time"].median()), 3),
            "zero_move_pct": round(float((fps["move_time"] == 0.0).mean() * 100), 2),
            "mean_consecutive_fast": round(float(fps["consecutive_fast_moves"].mean()), 3),
            "speed_breakdown": fps["speed_category"].value_counts().to_dict(),
            "phase_breakdown": fps["game_phase"].value_counts().to_dict()
        },
        "false_negatives": {
            "count": len(fns),
            "mean_move_time": round(float(fns["move_time"].mean()), 3),
            "median_move_time": round(float(fns["move_time"].median()), 3),
            "zero_move_pct": round(float((fns["move_time"] == 0.0).mean() * 100), 2),
            "mean_consecutive_fast": round(float(fns["consecutive_fast_moves"].mean()), 3),
            "speed_breakdown": fns["speed_category"].value_counts().to_dict(),
            "phase_breakdown": fns["game_phase"].value_counts().to_dict()
        }
    }
    with open(os.path.join(RESULTS_DIR, "error_summary.json"), "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)
    print("Saved error summary JSON.")

    # -------------------------------------------------------------
    # 12. Save Best Model and Metadata
    # -------------------------------------------------------------
    print("\n--- Persisting Best Model & Specifications ---")
    model_save_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_save_path)
    print(f"Saved best model to {model_save_path}")
    
    cols_save_path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(cols_save_path, "w", encoding="utf-8") as f:
        json.dump(best_cols, f, indent=2)
    print(f"Saved feature columns to {cols_save_path}")
    
    metadata = {
        "model_name": "EDA Behavioral LightGBM Classifier",
        "model_type": "LightGBM (Gradient Boosted Decision Trees)",
        "version": "1.0.0",
        "date": "October 2026",
        "feature_set": "EDA_BEHAVIORAL_10",
        "features": best_cols,
        "operating_threshold": 0.50,
        "test_metrics_unseen_game": {
            "roc_auc": round(float(roc_auc_score(y_test_best, y_prob_best)), 4),
            "pr_auc": round(float(average_precision_score(y_test_best, y_prob_best)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test_best, y_pred_best)), 4),
            "precision": round(float(precision_score(y_test_best, y_pred_best, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test_best, y_pred_best, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test_best, y_pred_best, zero_division=0)), 4)
        },
        "training_strategy": {
            "class_weight": "balanced",
            "scale_pos_weight": round(float(pos_weight), 2),
            "grouping_variable": "game_id",
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 6
        }
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {meta_path}")
    
    elapsed = time.time() - start_total
    print(f"\nTraining and Evaluation pipeline finished successfully in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    run_pipeline()
