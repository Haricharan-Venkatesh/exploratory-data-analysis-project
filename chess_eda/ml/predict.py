"""
chess_eda.ml.predict
====================
Local inference pipeline for PGN engine-assistance detection.

Extracts Stockfish-enriched features from a completed PGN game and generates:
1. White: human-like vs. engine-assisted likelihood
2. Black: human-like vs. engine-assisted likelihood
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd

from chess_eda.engine_analysis import analyse_game, validate_stockfish_path
from chess_eda.feature_extraction import extract_move_rows
from chess_eda.ml.features import (
    ML_FEATURE_COLUMNS,
    extract_player_features_from_engine_rows,
)
from chess_eda.move_features import extract_move_features
from chess_eda.pgn_parser import load_first_game, PGNParseError


DEFAULT_RF_MODEL_PATH = os.path.join("models", "chess_fraud_rf.joblib")
DEFAULT_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", r"C:\stockfish\stockfish.exe")


def predict_game(
    pgn_path: str,
    stockfish_path: str = DEFAULT_STOCKFISH_PATH,
    model_path: str = DEFAULT_RF_MODEL_PATH,
    depth: int = 10,
    verbose: bool = True,
    include_v5: bool = False,
    require_v5_depth15: bool = True,
) -> dict[str, Any]:
    """Run end-to-end ML prediction on a completed PGN game.

    Parameters
    ----------
    pgn_path:
        Path to PGN file.
    stockfish_path:
        Path to official Stockfish executable.
    model_path:
        Path to fitted scikit-learn Pipeline (.joblib).
    depth:
        Stockfish search depth.
    verbose:
        Whether to print user-friendly output.
    include_v5:
        If True, extracts experimental Version 5 position-complexity features
        and evaluates experimental models if available in models/experimental/.
        Defaults to False (pure Version 4 baseline production analysis).
    require_v5_depth15:
        If True and include_v5 is True, requires depth=15 for Version 5 experimental
        model inference to maintain mathematical consistency with the training benchmark
        (move_stockfish_1 vs move_stockfish_15).

    Returns
    -------
    dict[str, Any]
        Prediction report containing player estimates,
        computed features, and scientific disclaimers.
    """
    if not os.path.isfile(pgn_path):
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Fitted model not found at '{model_path}'.\n"
            "Please run:  python main.py --train-model"
        )

    resolved_sf = validate_stockfish_path(stockfish_path)

    # 1. Parse game
    game = load_first_game(pgn_path)
    headers = game.headers
    white_name = headers.get("White", "White Player")
    black_name = headers.get("Black", "Black Player")
    result = headers.get("Result", "*")

    # 2. Extract base features and check non-empty
    base_rows = extract_move_features(game, game_id=1)
    if not base_rows:
        raise ValueError(
            f"Game in '{pgn_path}' contains no mainline moves. Cannot compute features."
        )

    # 3. Run Stockfish analysis
    if verbose:
        print(f"Analysing '{pgn_path}' with Stockfish (depth={depth})...", flush=True)

    engine_rows, meta = analyse_game(game, stockfish_path=resolved_sf, depth=depth, include_depth1=include_v5)

    # Merge base features with engine analysis
    merged_rows: list[dict[str, Any]] = []
    for b, e in zip(base_rows, engine_rows):
        merged_rows.append({**b, **e})

    # 4. Extract aligned feature vectors for White and Black
    white_feat_dict = extract_player_features_from_engine_rows(merged_rows, "White", include_complexity=include_v5)
    black_feat_dict = extract_player_features_from_engine_rows(merged_rows, "Black", include_complexity=include_v5)

    df_white = pd.DataFrame([white_feat_dict])[ML_FEATURE_COLUMNS]
    df_black = pd.DataFrame([black_feat_dict])[ML_FEATURE_COLUMNS]

    # 5. Load model pipeline and infer
    pipeline = joblib.load(model_path)

    # Dynamically resolve positive class index (class 1) from pipeline classes_
    pipeline_classes = list(
        getattr(pipeline, "classes_", getattr(pipeline.steps[-1][1], "classes_", [0, 1]))
    )
    pos_idx = pipeline_classes.index(1) if 1 in pipeline_classes else (
        pipeline_classes.index(True) if True in pipeline_classes else 1
    )

    p_assisted_white = float(pipeline.predict_proba(df_white)[:, pos_idx][0])
    p_human_white = float(1.0 - p_assisted_white)
    white_label = "engine-assisted" if p_assisted_white >= 0.5 else "human-like"

    p_assisted_black = float(pipeline.predict_proba(df_black)[:, pos_idx][0])
    p_human_black = float(1.0 - p_assisted_black)
    black_label = "engine-assisted" if p_assisted_black >= 0.5 else "human-like"

    # 6. Optional Version 5 experimental inference (only when include_v5 is True)
    v5_exp = None
    if include_v5:
        exp_model_name = os.path.basename(model_path).replace(".joblib", "_v5_exp.joblib")
        exp_dir = os.path.join(os.path.dirname(model_path), "experimental")
        exp_path = os.path.join(exp_dir, exp_model_name)
        if not os.path.isfile(exp_path):
            exp_path = os.path.join("models", "experimental", exp_model_name)

        if os.path.isfile(exp_path):
            is_compatible = (depth == 15)
            if not is_compatible and require_v5_depth15:
                v5_exp = {
                    "model_name": os.path.basename(exp_path),
                    "is_depth_compatible": False,
                    "status": "incompatible_depth",
                    "depth_used": depth,
                    "required_depth": 15,
                    "message": (
                        f"Version 5 experimental models require depth-15 analysis to match "
                        f"the training benchmark schema (move_stockfish_1 vs move_stockfish_15). "
                        f"Current analysis depth is {depth}."
                    ),
                    "white_score": None,
                    "black_score": None,
                    "white_classification": None,
                    "black_classification": None,
                    "features_white": None,
                    "features_black": None,
                }
            else:
                try:
                    from chess_eda.ml.features import V5_ALL_FEATURE_COLUMNS
                    exp_pipeline = joblib.load(exp_path)
                    df_white_v5 = pd.DataFrame([white_feat_dict])[V5_ALL_FEATURE_COLUMNS]
                    df_black_v5 = pd.DataFrame([black_feat_dict])[V5_ALL_FEATURE_COLUMNS]

                    p_w_v5 = float(exp_pipeline.predict_proba(df_white_v5)[:, 1][0])
                    p_b_v5 = float(exp_pipeline.predict_proba(df_black_v5)[:, 1][0])

                    v5_exp = {
                        "model_name": os.path.basename(exp_path),
                        "is_depth_compatible": is_compatible,
                        "status": "compatible" if is_compatible else "unverified_depth",
                        "depth_used": depth,
                        "required_depth": 15,
                        "white_score": round(p_w_v5, 4),
                        "black_score": round(p_b_v5, 4),
                        "white_classification": "engine-assisted" if p_w_v5 >= 0.5 else "human-like",
                        "black_classification": "engine-assisted" if p_b_v5 >= 0.5 else "human-like",
                        "features_white": {k: white_feat_dict.get(k) for k in [
                            "unstable_top1_match_rate", "stable_top1_match_rate", "depth_drop_rate",
                            "critical_top1_match_rate", "critical_mean_cp_loss"
                        ]},
                        "features_black": {k: black_feat_dict.get(k) for k in [
                            "unstable_top1_match_rate", "stable_top1_match_rate", "depth_drop_rate",
                            "critical_top1_match_rate", "critical_mean_cp_loss"
                        ]},
                    }
                except Exception:
                    v5_exp = None

    report: dict[str, Any] = {
        "game_info": {
            "pgn_path": pgn_path,
            "white": white_name,
            "black": black_name,
            "result": result,
            "total_plies": len(merged_rows),
            "engine_used": meta.get("engine_name", "Stockfish"),
            "depth": depth,
        },
        "model_used": os.path.basename(model_path),
        "white_player": {
            "name": white_name,
            "classification": white_label,
            "engine_assisted_likelihood": round(p_assisted_white, 4),
            "human_like_likelihood": round(p_human_white, 4),
            "engine_assistance_score": round(p_assisted_white, 4),
            "complementary_human_like_score": round(p_human_white, 4),
            "human_like_score": round(p_human_white, 4),
            "key_features": {
                "mean_centipawn_loss": white_feat_dict.get("mean_centipawn_loss"),
                "top1_match_rate": white_feat_dict.get("top1_match_rate"),
                "blunder_rate": white_feat_dict.get("blunder_rate"),
                "move_count": white_feat_dict.get("move_count"),
            },
        },
        "black_player": {
            "name": black_name,
            "classification": black_label,
            "engine_assisted_likelihood": round(p_assisted_black, 4),
            "human_like_likelihood": round(p_human_like_black := p_human_black, 4),
            "engine_assistance_score": round(p_assisted_black, 4),
            "complementary_human_like_score": round(p_human_black, 4),
            "human_like_score": round(p_human_black, 4),
            "key_features": {
                "mean_centipawn_loss": black_feat_dict.get("mean_centipawn_loss"),
                "top1_match_rate": black_feat_dict.get("top1_match_rate"),
                "blunder_rate": black_feat_dict.get("blunder_rate"),
                "move_count": black_feat_dict.get("move_count"),
            },
        },
        "disclaimer": (
            "These uncalibrated scores reflect behavioral concordance with the controlled ChessFraud "
            "benchmark (~2000 Elo players with Stockfish 15 hints). The complementary human-like score is "
            "calculated as 100% minus the engine-assistance score; neither score is a calibrated real-world "
            "probability. The 50% classification threshold is an operational software boundary, not a "
            "scientifically established threshold for cheating. These scores DO NOT constitute definitive "
            "proof of computer assistance or unethical conduct."
        ),
        "white_features": white_feat_dict,
        "black_features": black_feat_dict,
        "move_analysis": merged_rows,
        "v5_experimental": v5_exp,
    }

    if verbose:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        w_cp = white_feat_dict.get("mean_centipawn_loss")
        w_cp_str = f"{w_cp:.2f} cp" if isinstance(w_cp, (int, float)) and not (isinstance(w_cp, float) and pd.isna(w_cp)) else "N/A"
        w_top1 = white_feat_dict.get("top1_match_rate")
        w_top1_str = f"{w_top1 * 100:.1f}%" if isinstance(w_top1, (int, float)) and not (isinstance(w_top1, float) and pd.isna(w_top1)) else "N/A"
        w_blund = white_feat_dict.get("blunder_rate")
        w_blund_str = f"{w_blund * 100:.1f}%" if isinstance(w_blund, (int, float)) and not (isinstance(w_blund, float) and pd.isna(w_blund)) else "N/A"

        b_cp = black_feat_dict.get("mean_centipawn_loss")
        b_cp_str = f"{b_cp:.2f} cp" if isinstance(b_cp, (int, float)) and not (isinstance(b_cp, float) and pd.isna(b_cp)) else "N/A"
        b_top1 = black_feat_dict.get("top1_match_rate")
        b_top1_str = f"{b_top1 * 100:.1f}%" if isinstance(b_top1, (int, float)) and not (isinstance(b_top1, float) and pd.isna(b_top1)) else "N/A"
        b_blund = black_feat_dict.get("blunder_rate")
        b_blund_str = f"{b_blund * 100:.1f}%" if isinstance(b_blund, (int, float)) and not (isinstance(b_blund, float) and pd.isna(b_blund)) else "N/A"

        print("\n=================================================================")
        print("VERSION 3: ENGINE-ASSISTANCE PREDICTION REPORT")
        print("=================================================================")
        print(f"Game       : {white_name} vs. {black_name} ({result})")
        print(f"Moves      : {len(merged_rows)} plies | Analyzed at depth {depth}")
        print(f"Model      : {report['model_used']}")
        print("-----------------------------------------------------------------")
        print("PLAYER-LEVEL ENGINE-ASSISTANCE ESTIMATES & SCORES\n")
        print(f"Player 1 — {white_name} (White)")
        print(f"  Engine-assistance score (uncalibrated):        {p_assisted_white * 100:.1f}%")
        print(f"  Complementary human-like score (uncalibrated): {p_human_white * 100:.1f}%")
        print(f"  Classification (50% software threshold):       {white_label.upper()}")
        print(f"  Key Metrics:                                  Mean CP Loss = {w_cp_str}, Top1 Rate = {w_top1_str}, Blunder Rate = {w_blund_str}\n")

        print(f"Player 2 — {black_name} (Black)")
        print(f"  Engine-assistance score (uncalibrated):        {p_assisted_black * 100:.1f}%")
        print(f"  Complementary human-like score (uncalibrated): {p_human_black * 100:.1f}%")
        print(f"  Classification (50% software threshold):       {black_label.upper()}")
        print(f"  Key Metrics:                                  Mean CP Loss = {b_cp_str}, Top1 Rate = {b_top1_str}, Blunder Rate = {b_blund_str}")
        print("-----------------------------------------------------------------")
        print("METHODOLOGICAL NOTE & SCIENTIFIC DISCLAIMER:")
        print(report["disclaimer"])
        print("=================================================================\n")

    return report
