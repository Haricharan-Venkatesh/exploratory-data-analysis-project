"""
chess_eda.ml.v5_experiments
============================
Controlled experimental evaluation and model training for Version 5:
Position-Complexity-Aware Features for Engine-Assistance Detection.

Key Capabilities:
-----------------
1. Extracts scientifically motivated position-complexity features:
   - `unstable_top1_match_rate`: Engine agreement on non-obvious moves (shallow SF1 != deep SF15)
   - `stable_top1_match_rate`: Engine agreement on shallow-obvious moves (shallow SF1 == deep SF15)
   - `depth_drop_rate`: Accuracy drop from simple to difficult moves (stable - unstable)
   - `critical_top1_match_rate`: Engine agreement in close/tense positions (|eval| <= 200 cp)
   - `critical_mean_cp_loss`: Error magnitude in close/tense positions (|eval| <= 200 cp)
2. Executes strict 5-fold StratifiedGroupKFold cross-validation grouped by physical match.
3. Compares baseline (13 features) against complexity-augmented models (18 features).
4. Saves experimental pipelines separately to models/experimental/ without replacing baseline production models.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chess_eda.ml.dataset import DEFAULT_DATA_PATH, load_chess_fraud_dataset
from chess_eda.ml.features import (
    ML_FEATURE_COLUMNS,
    V5_COMPLEXITY_FEATURE_COLUMNS,
    V5_ALL_FEATURE_COLUMNS,
    extract_player_features_from_df,
)


DEFAULT_EXP_MODEL_DIR = os.path.join("models", "experimental")


def evaluate_oof_cv(
    pipeline_factory,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run StratifiedGroupKFold cross-validation and compute out-of-fold metrics."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        pipe = pipeline_factory()
        pipe.fit(X[train_idx], y[train_idx])

        val_preds = pipe.predict(X[val_idx])
        val_probs = pipe.predict_proba(X[val_idx])[:, 1]

        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs

        f_acc = float(accuracy_score(y[val_idx], val_preds))
        f_prec = float(precision_score(y[val_idx], val_preds, zero_division=0))
        f_rec = float(recall_score(y[val_idx], val_preds, zero_division=0))
        f_f1 = float(f1_score(y[val_idx], val_preds, zero_division=0))
        f_roc = float(roc_auc_score(y[val_idx], val_probs))
        f_pr = float(average_precision_score(y[val_idx], val_probs))

        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": round(f_acc, 4),
            "precision": round(f_prec, 4),
            "recall": round(f_rec, 4),
            "f1_score": round(f_f1, 4),
            "roc_auc": round(f_roc, 4),
            "pr_auc": round(f_pr, 4),
        })

    acc = float(accuracy_score(y, oof_preds))
    prec = float(precision_score(y, oof_preds, zero_division=0))
    rec = float(recall_score(y, oof_preds, zero_division=0))
    f1 = float(f1_score(y, oof_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y, oof_probs))
    pr_auc = float(average_precision_score(y, oof_probs))
    cm = confusion_matrix(y, oof_preds).tolist()

    fold_accs = [m["accuracy"] for m in fold_metrics]
    summary = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fold_acc_std": round(float(np.std(fold_accs)), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["Fair (0)", "Assisted (1)"],
    }
    return summary, oof_preds, oof_probs, fold_metrics


def run_v5_complexity_experiment(
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_EXP_MODEL_DIR,
    random_state: int = 42,
    n_splits: int = 5,
    save_models: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute the controlled Version 5 position-complexity experiment.

    Compares baseline 13-feature models against 18-feature complexity-augmented models
    using identical StratifiedGroupKFold splits grouped by physical_game_id.
    """
    if verbose:
        print("=" * 70)
        print("VERSION 5: POSITION-COMPLEXITY EXPERIMENTAL EVALUATION")
        print("=" * 70)

    # 1. Load benchmark dataset
    df = load_chess_fraud_dataset(data_path)
    total_plies = len(df)

    # 2. Extract features
    pg_df = extract_player_features_from_df(df, include_complexity=True)
    total_samples = len(pg_df)
    unique_matches = pg_df["physical_game_id"].nunique()

    y = pg_df["is_cheating"].values
    groups = pg_df["physical_game_id"].values

    base_cols = list(ML_FEATURE_COLUMNS)
    comp_cols = list(V5_COMPLEXITY_FEATURE_COLUMNS)
    all_cols = list(V5_ALL_FEATURE_COLUMNS)

    # Feature subsets for ablation
    sub_depth = base_cols + ["unstable_top1_match_rate", "stable_top1_match_rate", "depth_drop_rate"]
    sub_crit = base_cols + ["critical_top1_match_rate", "critical_mean_cp_loss"]

    # Pipeline factories
    def make_lr_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])

    def make_rf_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)),
        ])

    # 3. Evaluate Baseline (13 features)
    X_base = pg_df[base_cols].values
    lr_base_metrics, _, _, lr_base_folds = evaluate_oof_cv(make_lr_pipe, X_base, y, groups, n_splits, random_state)
    rf_base_metrics, _, _, rf_base_folds = evaluate_oof_cv(make_rf_pipe, X_base, y, groups, n_splits, random_state)

    # 4. Evaluate Augmented (18 features)
    X_all = pg_df[all_cols].values
    lr_exp_metrics, _, _, lr_exp_folds = evaluate_oof_cv(make_lr_pipe, X_all, y, groups, n_splits, random_state)
    rf_exp_metrics, _, _, rf_exp_folds = evaluate_oof_cv(make_rf_pipe, X_all, y, groups, n_splits, random_state)

    # 5. Evaluate Ablation Subsets with RF
    X_depth = pg_df[sub_depth].values
    rf_depth_metrics, _, _, _ = evaluate_oof_cv(make_rf_pipe, X_depth, y, groups, n_splits, random_state)

    X_crit = pg_df[sub_crit].values
    rf_crit_metrics, _, _, _ = evaluate_oof_cv(make_rf_pipe, X_crit, y, groups, n_splits, random_state)

    # 6. Fit Final Experimental Production Pipelines (on all 1,010 samples)
    final_lr_exp = make_lr_pipe().fit(X_all, y)
    final_rf_exp = make_rf_pipe().fit(X_all, y)

    # Extract feature importance for RF
    rf_clf = final_rf_exp.named_steps["clf"]
    importances = {col: round(float(imp), 4) for col, imp in zip(all_cols, rf_clf.feature_importances_)}
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    # Extract LR coefficients
    lr_clf = final_lr_exp.named_steps["clf"]
    coefficients = {col: round(float(c), 4) for col, c in zip(all_cols, lr_clf.coef_[0])}

    # 7. Serialize models if requested
    saved_files = []
    if save_models:
        os.makedirs(output_dir, exist_ok=True)
        rf_path = os.path.join(output_dir, "chess_fraud_rf_v5_exp.joblib")
        lr_path = os.path.join(output_dir, "chess_fraud_lr_v5_exp.joblib")
        meta_path = os.path.join(output_dir, "v5_experiment_metadata.json")

        joblib.dump(final_rf_exp, rf_path)
        joblib.dump(final_lr_exp, lr_path)
        saved_files.extend([rf_path, lr_path])

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "ChessFraud Controlled-Tournament Benchmark",
        "total_plies": total_plies,
        "total_player_games": total_samples,
        "total_physical_matches": unique_matches,
        "n_splits": n_splits,
        "random_state": random_state,
        "baseline_feature_columns": base_cols,
        "complexity_feature_columns": comp_cols,
        "all_feature_columns": all_cols,
        "comparisons": {
            "logistic_regression": {
                "baseline": lr_base_metrics,
                "augmented": lr_exp_metrics,
                "delta": {
                    "accuracy": round(lr_exp_metrics["accuracy"] - lr_base_metrics["accuracy"], 4),
                    "precision": round(lr_exp_metrics["precision"] - lr_base_metrics["precision"], 4),
                    "recall": round(lr_exp_metrics["recall"] - lr_base_metrics["recall"], 4),
                    "f1_score": round(lr_exp_metrics["f1_score"] - lr_base_metrics["f1_score"], 4),
                    "roc_auc": round(lr_exp_metrics["roc_auc"] - lr_base_metrics["roc_auc"], 4),
                    "pr_auc": round(lr_exp_metrics["pr_auc"] - lr_base_metrics["pr_auc"], 4),
                },
                "fold_metrics_augmented": lr_exp_folds,
                "feature_coefficients": coefficients,
            },
            "random_forest": {
                "baseline": rf_base_metrics,
                "augmented": rf_exp_metrics,
                "delta": {
                    "accuracy": round(rf_exp_metrics["accuracy"] - rf_base_metrics["accuracy"], 4),
                    "precision": round(rf_exp_metrics["precision"] - rf_base_metrics["precision"], 4),
                    "recall": round(rf_exp_metrics["recall"] - rf_base_metrics["recall"], 4),
                    "f1_score": round(rf_exp_metrics["f1_score"] - rf_base_metrics["f1_score"], 4),
                    "roc_auc": round(rf_exp_metrics["roc_auc"] - rf_base_metrics["roc_auc"], 4),
                    "pr_auc": round(rf_exp_metrics["pr_auc"] - rf_base_metrics["pr_auc"], 4),
                },
                "fold_metrics_augmented": rf_exp_folds,
                "feature_importances": sorted_importances,
            },
        },
        "ablations_rf": {
            "baseline_13": rf_base_metrics,
            "depth_stability_16": rf_depth_metrics,
            "critical_eval_15": rf_crit_metrics,
            "all_augmented_18": rf_exp_metrics,
        },
        "saved_artifacts": saved_files,
    }

    if save_models:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        saved_files.append(meta_path)

    if verbose:
        print(f"Total player-games: {total_samples} across {unique_matches} physical matches.")
        print("-" * 70)
        print("MODEL COMPARISON SUMMARY (Out-of-Fold 5-Fold StratifiedGroupKFold):")
        print("-" * 70)
        print(f"{'Model & Configuration':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7} | {'PR-AUC':<6}")
        print("-" * 70)
        print(f"{'Logistic Regression (Base 13)':<28} | {lr_base_metrics['accuracy']:.4f}   | {lr_base_metrics['precision']:.4f}    | {lr_base_metrics['recall']:.4f} | {lr_base_metrics['f1_score']:.4f} | {lr_base_metrics['roc_auc']:.4f}  | {lr_base_metrics['pr_auc']:.4f}")
        print(f"{'Logistic Regression (Exp 18)':<28} | {lr_exp_metrics['accuracy']:.4f}   | {lr_exp_metrics['precision']:.4f}    | {lr_exp_metrics['recall']:.4f} | {lr_exp_metrics['f1_score']:.4f} | {lr_exp_metrics['roc_auc']:.4f}  | {lr_exp_metrics['pr_auc']:.4f}")
        print("-" * 70)
        print(f"{'Random Forest (Base 13)':<28} | {rf_base_metrics['accuracy']:.4f}   | {rf_base_metrics['precision']:.4f}    | {rf_base_metrics['recall']:.4f} | {rf_base_metrics['f1_score']:.4f} | {rf_base_metrics['roc_auc']:.4f}  | {rf_base_metrics['pr_auc']:.4f}")
        print(f"{'Random Forest (Exp 18)':<28} | {rf_exp_metrics['accuracy']:.4f}   | {rf_exp_metrics['precision']:.4f}    | {rf_exp_metrics['recall']:.4f} | {rf_exp_metrics['f1_score']:.4f} | {rf_exp_metrics['roc_auc']:.4f}  | {rf_exp_metrics['pr_auc']:.4f}")
        print("-" * 70)
        print("RANDOM FOREST ABLATION COMPARISON:")
        print(f"  * Baseline (13 feats):             Accuracy = {rf_base_metrics['accuracy']:.4f}, ROC-AUC = {rf_base_metrics['roc_auc']:.4f}, PR-AUC = {rf_base_metrics['pr_auc']:.4f}")
        print(f"  * Depth Stability Only (16 feats): Accuracy = {rf_depth_metrics['accuracy']:.4f}, ROC-AUC = {rf_depth_metrics['roc_auc']:.4f}, PR-AUC = {rf_depth_metrics['pr_auc']:.4f}")
        print(f"  * Critical Positions Only (15 feats): Accuracy = {rf_crit_metrics['accuracy']:.4f}, ROC-AUC = {rf_crit_metrics['roc_auc']:.4f}, PR-AUC = {rf_crit_metrics['pr_auc']:.4f}")
        print(f"  * All 5 Augmented (18 feats):      Accuracy = {rf_exp_metrics['accuracy']:.4f}, ROC-AUC = {rf_exp_metrics['roc_auc']:.4f}, PR-AUC = {rf_exp_metrics['pr_auc']:.4f}")
        print("=" * 70)

    return report
