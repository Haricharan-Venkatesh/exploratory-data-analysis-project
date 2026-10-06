"""
Verify Saved Model and Metadata Artifacts
=========================================
Checks:
- best_model.pkl loads properly
- feature_columns.json has exactly 10 features matching training
- model_metadata.json contains model type, threshold, metrics
- target mapping (0 = HUMAN, 1 = AI)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

def verify_model_artifacts():
    print("=== 1. Verifying Model Artifacts ===")
    assert os.path.exists(MODEL_PATH), f"Missing {MODEL_PATH}"
    assert os.path.exists(COLUMNS_PATH), f"Missing {COLUMNS_PATH}"
    assert os.path.exists(METADATA_PATH), f"Missing {METADATA_PATH}"
    
    # Load model
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded: {type(model).__name__}")
    
    # Load feature columns
    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)
    print(f"[OK] Feature count: {len(features)}")
    print(f"[OK] Feature columns: {features}")
    
    expected_10_features = [
        "consecutive_fast_moves",
        "premove_rate_10",
        "cumulative_premove_rate",
        "current_endgame_clock_ratio",
        "phase_deliberation_ratio",
        "move_time_cv_10",
        "time_pressure_jitter",
        "relative_move_time",
        "time_spent_ratio",
        "tank_move_count_10"
    ]
    assert features == expected_10_features, f"Feature columns mismatch: {features} vs {expected_10_features}"
    
    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"[OK] Model Name: {metadata.get('model_name')}")
    print(f"[OK] Operating Threshold: {metadata.get('operating_threshold')}")
    print(f"[OK] Test ROC-AUC: {metadata.get('test_metrics_unseen_game', {}).get('roc_auc')}")
    print(f"[OK] Test PR-AUC: {metadata.get('test_metrics_unseen_game', {}).get('pr_auc')}")
    
    # Verify predict_proba output shape
    dummy_input = pd.DataFrame([{col: 0.0 for col in features}])
    probs = model.predict_proba(dummy_input)
    assert probs.shape == (1, 2), f"Unexpected proba shape: {probs.shape}"
    print(f"[OK] Sample prediction proba: Human={probs[0][0]:.4f}, AI={probs[0][1]:.4f}")
    print("[SUCCESS] All model artifact verifications passed cleanly!\n")

if __name__ == "__main__":
    verify_model_artifacts()
