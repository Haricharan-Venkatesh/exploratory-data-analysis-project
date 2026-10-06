"""
Prediction & Inference Module for Human-AI Chess Move Detection
===============================================================
Provides clean prediction functions for scoring:
1. Individual feature vectors: predict_player(features, threshold)
2. Full PGN games move-by-move: analyze_pgn(pgn_text, threshold)

Uses strictly the validated 10-feature EDA Behavioral LightGBM model.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.inference_features import parse_pgn_game, FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

_MODEL = None
_FEATURE_COLS = None
_METADATA = None

DISCLAIMER_TEXT = (
    "This prediction is a statistical behavioral classification and is "
    "not proof of cheating or engine assistance."
)

def _load_model_artifacts():
    global _MODEL, _FEATURE_COLS, _METADATA
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        _MODEL = joblib.load(MODEL_PATH)
        
        with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
            _FEATURE_COLS = json.load(f)
            
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _METADATA = json.load(f)

def get_model_metadata() -> Dict[str, Any]:
    _load_model_artifacts()
    return _METADATA

def get_feature_columns() -> List[str]:
    _load_model_artifacts()
    return _FEATURE_COLS


def explain_behavioral_signals(features: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Generate grounded, empirical behavioral signal summaries based on
    feature values and EDA benchmarks. Never claims cognitive causality.
    """
    signals = []
    
    # 1. Consecutive fast moves
    fast_streak = features.get("consecutive_fast_moves", 0)
    if fast_streak >= 4:
        signals.append({
            "feature": "consecutive_fast_moves",
            "name": "Fast Move Streak",
            "value": f"{int(fast_streak)} moves",
            "indication": "AI-leaning",
            "observation": f"Sustained sequence of {int(fast_streak)} moves under 0.5s with zero hesitation."
        })
    elif fast_streak == 0:
        signals.append({
            "feature": "consecutive_fast_moves",
            "name": "Deliberate Pacing",
            "value": "0 fast moves",
            "indication": "Human-leaning",
            "observation": "Move executed with deliberate human thinking delay (>0.5s)."
        })
        
    # 2. Pre-move rate
    pmr = features.get("premove_rate_10", 0.0)
    if pmr >= 0.40:
        signals.append({
            "feature": "premove_rate_10",
            "name": "Pre-move Frequency",
            "value": f"{pmr * 100:.1f}%",
            "indication": "AI-leaning" if pmr >= 0.60 else "Human Bullet Pacing",
            "observation": f"High rate of zero-second execution ({pmr * 100:.0f}% of recent moves)."
        })
        
    # 3. Move-time variability (CV)
    cv = features.get("move_time_cv_10", 0.0)
    if cv < 0.25 and features.get("consecutive_fast_moves", 0) < 3:
        signals.append({
            "feature": "move_time_cv_10",
            "name": "Low Timing Variance",
            "value": f"CV {cv:.2f}",
            "indication": "AI-leaning",
            "observation": "Unusually uniform deliberation time across consecutive tactical decisions."
        })
    elif cv > 1.2:
        signals.append({
            "feature": "move_time_cv_10",
            "name": "High Timing Volatility",
            "value": f"CV {cv:.2f}",
            "indication": "Human-leaning",
            "observation": "Marked variation between intuitive moves and tactical deliberation."
        })
        
    # 4. Phase Deliberation
    pdr = features.get("phase_deliberation_ratio", 1.0)
    if pdr < 1.1:
        signals.append({
            "feature": "phase_deliberation_ratio",
            "name": "Flat Deliberation Curve",
            "value": f"{pdr:.2f}x",
            "indication": "AI-leaning",
            "observation": "Middlegame thinking duration matches opening book speed rather than surging."
        })
    elif pdr > 2.5:
        signals.append({
            "feature": "phase_deliberation_ratio",
            "name": "Middlegame Thinking Surge",
            "value": f"{pdr:.2f}x",
            "indication": "Human-leaning",
            "observation": "Characteristic human behavior: deep deliberation in complex middlegame positions."
        })
        
    # 5. Tank moves
    tanks = features.get("tank_move_count_10", 0)
    if tanks >= 1:
        signals.append({
            "feature": "tank_move_count_10",
            "name": "Deep Think (>30s)",
            "value": f"{int(tanks)} tank moves",
            "indication": "Human-leaning",
            "observation": "Extended pause on critical juncture, typical of human tactical calculation."
        })
        
    # 6. Relative move time
    rel_time = features.get("relative_move_time", 0.0)
    if rel_time > 3.0:
        signals.append({
            "feature": "relative_move_time",
            "name": "High Time Investment",
            "value": f"{rel_time:.1f}x budget",
            "indication": "Human-leaning",
            "observation": "Substantial fraction of time budget invested on single turn."
        })
        
    return signals


def predict_player(
    features: Union[Dict[str, float], pd.DataFrame],
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Score a single move or feature vector using the EDA Behavioral LightGBM model.
    """
    _load_model_artifacts()
    
    op_threshold = float(threshold if threshold is not None else _METADATA.get("operating_threshold", 0.50))
    
    if isinstance(features, dict):
        df_input = pd.DataFrame([features])
        feat_dict = features
    elif isinstance(features, pd.DataFrame):
        df_input = features.copy()
        feat_dict = df_input.iloc[0].to_dict()
    else:
        raise ValueError("features must be a dict or pandas DataFrame")
        
    # Ensure all required columns are present
    missing = [c for c in _FEATURE_COLS if c not in df_input.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
        
    X = df_input[_FEATURE_COLS]
    probs = _MODEL.predict_proba(X)[:, 1]
    ai_prob = float(probs[0])
    human_prob = 1.0 - ai_prob
    
    is_ai = ai_prob >= op_threshold
    prediction_label = "AI-like behavioral pattern" if is_ai else "Predicted HUMAN"
    short_class = "AI" if is_ai else "HUMAN"
    
    behavioral_signals = explain_behavioral_signals(feat_dict)
    
    return {
        "prediction": prediction_label,
        "predicted_class": short_class,
        "ai_probability": round(ai_prob, 4),
        "human_probability": round(human_prob, 4),
        "ai_probability_pct": round(ai_prob * 100.0, 1),
        "human_probability_pct": round(human_prob * 100.0, 1),
        "operating_threshold": op_threshold,
        "model_name": _METADATA.get("model_name", "EDA Behavioral LightGBM Classifier"),
        "model_version": _METADATA.get("version", "1.0.0"),
        "features": {k: round(float(v), 4) for k, v in feat_dict.items() if k in _FEATURE_COLS},
        "behavioral_signals": behavioral_signals,
        "disclaimer": DISCLAIMER_TEXT
    }


def analyze_pgn(
    pgn_content: str,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    End-to-end PGN analysis pipeline:
    1. Parses PGN and moves without lookahead
    2. Computes behavioral features per move
    3. Runs model.predict_proba() for each move
    4. Aggregates player-level profiles and trajectories
    """
    _load_model_artifacts()
    op_threshold = float(threshold if threshold is not None else _METADATA.get("operating_threshold", 0.50))
    
    parsed = parse_pgn_game(pgn_content)
    moves = parsed.get("moves", [])
    
    if not moves:
        return {
            "metadata": parsed.get("metadata", {}),
            "has_clock_data": parsed.get("has_clock_data", False),
            "warning": parsed.get("warning", "No moves found in game."),
            "summary": {
                "white": {"prediction": "INSUFFICIENT_DATA", "ai_prob_mean": 0.0},
                "black": {"prediction": "INSUFFICIENT_DATA", "ai_prob_mean": 0.0}
            },
            "moves": []
        }
        
    evaluated_moves = []
    white_ai_probs = []
    black_ai_probs = []
    has_clock = parsed.get("has_clock_data", False)
    
    for m in moves:
        m_eval = dict(m)
        f10 = m["features_10"]
        if has_clock and f10 is not None:
            pred_res = predict_player(f10, threshold=op_threshold)
            m_eval["prediction"] = pred_res["prediction"]
            m_eval["predicted_class"] = pred_res["predicted_class"]
            m_eval["ai_probability"] = pred_res["ai_probability"]
            m_eval["human_probability"] = pred_res["human_probability"]
            m_eval["behavioral_signals"] = pred_res["behavioral_signals"]
            
            if m["player"] == "White":
                white_ai_probs.append(pred_res["ai_probability"])
            else:
                black_ai_probs.append(pred_res["ai_probability"])
        else:
            m_eval["prediction"] = "UNAVAILABLE (No Clock Data)"
            m_eval["predicted_class"] = "UNAVAILABLE"
            m_eval["ai_probability"] = None
            m_eval["human_probability"] = None
            m_eval["behavioral_signals"] = [{
                "feature": "clock_missing",
                "name": "Move Clock Data Unavailable",
                "value": "N/A",
                "indication": "Neutral",
                "observation": "Move clock data unavailable. Timing-based behavioral prediction may be unreliable without [%clk] comments."
            }]
            
        evaluated_moves.append(m_eval)
        
    # Aggregate player profiles
    def summarize_player(probs: List[float], player_name: str, total_moves: int) -> Dict[str, Any]:
        if not has_clock:
            return {
                "name": player_name,
                "moves_count": total_moves,
                "mean_ai_probability": None,
                "mean_human_probability": None,
                "mean_ai_probability_pct": None,
                "max_ai_probability_pct": None,
                "ai_flagged_move_ratio": None,
                "ai_flagged_move_pct": None,
                "overall_prediction": "UNAVAILABLE (No Clock Data)",
                "prediction": "UNAVAILABLE (No Clock Data)",
                "predicted_class": "UNAVAILABLE",
                "operating_threshold": op_threshold,
                "warning": "Move clock data unavailable. Timing-based behavioral prediction may be unreliable."
            }
        if not probs:
            return {"name": player_name, "moves_count": 0, "prediction": "NO_MOVES"}
        mean_p = float(np.mean(probs))
        max_p = float(np.max(probs))
        ai_flagged_ratio = float(sum(1 for p in probs if p >= op_threshold) / len(probs))
        is_overall_ai = mean_p >= op_threshold or ai_flagged_ratio >= 0.50
        
        pred_label = "AI-like behavioral pattern" if is_overall_ai else "Predicted HUMAN"
        short_class = "AI" if is_overall_ai else "HUMAN"
        
        return {
            "name": player_name,
            "moves_count": len(probs),
            "mean_ai_probability": round(mean_p, 4),
            "mean_human_probability": round(1.0 - mean_p, 4),
            "mean_ai_probability_pct": round(mean_p * 100.0, 1),
            "mean_human_probability_pct": round((1.0 - mean_p) * 100.0, 1),
            "max_ai_probability_pct": round(max_p * 100.0, 1),
            "ai_flagged_move_ratio": round(ai_flagged_ratio, 4),
            "ai_flagged_move_pct": round(ai_flagged_ratio * 100.0, 1),
            "overall_prediction": pred_label,
            "prediction": pred_label,
            "predicted_class": short_class,
            "operating_threshold": op_threshold
        }
        
    meta = parsed["metadata"]
    white_moves_cnt = sum(1 for m in moves if m["player"] == "White")
    black_moves_cnt = sum(1 for m in moves if m["player"] == "Black")
    summary_white = summarize_player(white_ai_probs, meta.get("white", "White"), white_moves_cnt)
    summary_black = summarize_player(black_ai_probs, meta.get("black", "Black"), black_moves_cnt)
    
    return {
        "metadata": meta,
        "has_clock_data": parsed.get("has_clock_data", False),
        "warning": parsed.get("warning"),
        "summary": {
            "white": summary_white,
            "black": summary_black
        },
        "operating_threshold": op_threshold,
        "disclaimer": DISCLAIMER_TEXT,
        "moves": evaluated_moves
    }


if __name__ == "__main__":
    # Test on single feature dict
    test_feat = {
        "consecutive_fast_moves": 5.0,
        "premove_rate_10": 0.70,
        "cumulative_premove_rate": 0.65,
        "current_endgame_clock_ratio": 0.0,
        "phase_deliberation_ratio": 0.95,
        "move_time_cv_10": 0.12,
        "time_pressure_jitter": 0.0,
        "relative_move_time": 0.10,
        "time_spent_ratio": 0.01,
        "tank_move_count_10": 0.0
    }
    res = predict_player(test_feat)
    print("=== Single Feature Vector Test ===")
    print(json.dumps(res, indent=2))
