"""
chess_eda/dashboard.py
======================
Reusable data transformation, validation, and presentation helper functions
for the Version 4 Streamlit dashboard.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Any

import chess.pgn
import pandas as pd


DEFAULT_METADATA_PATH = os.path.join("models", "model_metadata.json")
DEFAULT_STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", r"C:\stockfish\stockfish.exe")


def load_model_metadata(metadata_path: str = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """Load and parse the training and evaluation metadata for baseline models."""
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Model metadata not found at '{metadata_path}'. "
            "Please train the models first using 'python main.py --train-model'."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric_definitions() -> dict[str, dict[str, str]]:
    """Return beginner-friendly definitions of ML evaluation metrics."""
    return {
        "Accuracy": {
            "short": "Overall percentage of correct classifications.",
            "detailed": (
                "Proportion of total games correctly classified as either fair or engine-assisted. "
                "While intuitive, accuracy alone can be misleading if the dataset is imbalanced. "
                "Our benchmark majority-class baseline is 61.51%."
            ),
            "benchmark_context": "LR: 69.87% | RF: 69.04% | Majority Baseline: 61.51%",
        },
        "Precision": {
            "short": "Reliability when flagging engine assistance (few false alarms).",
            "detailed": (
                "Out of all games the model flagged as engine-assisted, what percentage were "
                "actually assisted? High precision minimizes false accusations against innocent players."
            ),
            "benchmark_context": "LR: 60.42% | RF: 60.23% (Avoids excessive false accusations)",
        },
        "Recall": {
            "short": "Sensitivity in detecting engine assistance (few missed cheaters).",
            "detailed": (
                "Out of all games that were genuinely engine-assisted in the benchmark, what "
                "fraction did the model successfully identify? High recall ensures few assisted games slip through."
            ),
            "benchmark_context": "LR: 63.04% | RF: 57.61%",
        },
        "F1-Score": {
            "short": "Harmonic mean balancing Precision and Recall.",
            "detailed": (
                "The balance between false alarms (precision) and missed cheaters (recall). "
                "Calculated as 2 * (Precision * Recall) / (Precision + Recall). A well-rounded performance score."
            ),
            "benchmark_context": "LR: 61.70% | RF: 58.89%",
        },
        "ROC-AUC": {
            "short": "Discriminative ability across all decision thresholds.",
            "detailed": (
                "Area Under the Receiver Operating Characteristic curve. Measures the model's ability "
                "to rank engine-assisted games higher than human games regardless of the 50% cutoff threshold. "
                "0.50 represents random guessing; 1.0 represents perfect ranking."
            ),
            "benchmark_context": "LR: 0.7575 | RF: 0.7241 (Strong discriminative power)",
        },
        "PR-AUC": {
            "short": "Precision-Recall Area under curve for the assisted class.",
            "detailed": (
                "Area under the Precision-Recall curve. Especially informative under class imbalance. "
                "Measures how well high precision is maintained as recall increases."
            ),
            "benchmark_context": "LR: 0.6983 | RF: 0.6304 | Baseline: 0.3849",
        },
    }


def get_feature_descriptions() -> dict[str, str]:
    """Return educational descriptions for all 13 behavioral ML features."""
    return {
        "mean_centipawn_loss": "Average loss in centipawns across all moves. Lower = more accurate play.",
        "std_centipawn_loss": "Standard deviation of centipawn loss. Measures tactical consistency vs volatility.",
        "median_centipawn_loss": "Median centipawn loss. Resilient to isolated catastrophic blunders.",
        "mean_norm_cp_loss": "Win probability drop per move via Lichess win-percentage curve (0.0 to 1.0).",
        "std_norm_cp_loss": "Standard deviation of win-probability drop.",
        "blunder_rate": "Fraction of moves with centipawn loss > 200 cp (severe errors).",
        "mistake_rate": "Fraction of moves with 100 < centipawn loss <= 200 cp.",
        "inaccuracy_rate": "Fraction of moves with 50 < centipawn loss <= 100 cp.",
        "excellent_rate": "Fraction of moves with centipawn loss <= 25 cp (near-engine moves).",
        "top1_match_rate": "Fraction of post-opening moves matching Stockfish's #1 candidate move.",
        "used_mean_cp_loss": "Average centipawn loss on post-opening plies (half_move >= 21).",
        "used_blunder_rate": "Fraction of post-opening plies with centipawn loss > 100 cp.",
        "move_count": "Total mainline moves played by the player in the game.",
    }


def get_preset_games() -> dict[str, dict[str, str]]:
    """Return verified preset games available locally for testing and demonstration."""
    return {
        "Sample: Immortal Game (Anderssen vs. Kieseritzky, 1851)": {
            "path": os.path.join("data", "sample.pgn"),
            "description": "Historical romantic master game. White sacrifices bishop, rooks, and queen in a forced mate.",
            "type": "Historical Human Game",
        },
        "Morphy Opera Game (Morphy vs. Duke of Brunswick, 1858)": {
            "path": os.path.join("data", "raw", "human_games", "opera_game_1858.pgn"),
            "description": "Classic masterpiece showcasing rapid development and devastating tactical sacrifices.",
            "type": "Historical Human Game",
        },
        "TCEC Engine Game (Stockfish vs. Leela Chess Zero, 2021)": {
            "path": os.path.join("data", "raw", "engine_generated_games", "tcec_engine_game.pgn"),
            "description": "Top-level engine championship match between Stockfish and neural net LCZero.",
            "type": "Computer vs Computer (Smoke Test)",
        },
        "Casual Practice Game (Synthetic dev game, 12 plies)": {
            "path": os.path.join("data", "raw", "unlabeled", "casual_game.pgn"),
            "description": "Short programmatic test fixture with opening blunders.",
            "type": "Synthetic Development Fixture",
        },
        "Audited Game: Engaliawad vs TheGreatIMU (Chess.com Rapid, 22 moves / 44 plies)": {
            "path": os.path.join("data", "raw", "unlabeled", "thegreatimu_game.pgn"),
            "description": "Rapid game (962 vs 758 Elo, 22 moves / 44 plies). White made early blunders leading to Black's tactical win.",
            "type": "Audited Online Rapid Game",
        },
    }


def validate_pgn_input(pgn_content: str) -> tuple[bool, str, chess.pgn.Game | None]:
    """
    Validate a PGN string.
    Returns:
        (is_valid, error_or_success_message, parsed_game_or_None)
    """
    if not pgn_content or not pgn_content.strip():
        return False, "PGN content is empty. Please provide a valid PGN string or file.", None

    try:
        s_io = io.StringIO(pgn_content.strip())
        game = chess.pgn.read_game(s_io)
    except Exception as e:
        return False, f"Failed to parse PGN syntax: {e}", None

    if game is None:
        return False, "No valid chess game found in the provided PGN text.", None

    # Verify game contains at least one move
    first_move = game.next()
    if first_move is None:
        return (
            False,
            "The PGN game contains headers but no mainline moves. Cannot compute features.",
            None,
        )

    return True, "Valid PGN game with mainline moves.", game


def save_uploaded_pgn_to_temp(uploaded_file_or_content: bytes | str) -> str:
    """Save PGN bytes or string into a temporary .pgn file and return its absolute path."""
    if isinstance(uploaded_file_or_content, bytes):
        content = uploaded_file_or_content.decode("utf-8", errors="replace")
    else:
        content = str(uploaded_file_or_content)

    temp_fd, temp_path = tempfile.mkstemp(suffix=".pgn", prefix="chess_eda_")
    with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
        f.write(content)

    return temp_path


def extract_eda_chart_data(report: dict[str, Any]) -> dict[str, Any]:
    """
    Transform report data into formats ready for Streamlit charts and dataframes.
    Extracts:
    - Move-level DataFrame with ply, player, cp_loss, top1 match, san
    - Move quality breakdown for White and Black
    - 13-feature comparison table
    """
    move_rows = report.get("move_analysis", [])
    df_moves = pd.DataFrame(move_rows)

    if not df_moves.empty:
        if "centipawn_loss" in df_moves.columns:
            df_moves["centipawn_loss"] = pd.to_numeric(df_moves["centipawn_loss"], errors="coerce")
        # Ensure canonical schema columns and aliases for robust consumption
        if "player_color" in df_moves.columns and "player" not in df_moves.columns:
            df_moves["player"] = df_moves["player_color"]
        elif "player" in df_moves.columns and "player_color" not in df_moves.columns:
            df_moves["player_color"] = df_moves["player"]

        if "ply_number" in df_moves.columns and "ply" not in df_moves.columns:
            df_moves["ply"] = df_moves["ply_number"]
        elif "ply" in df_moves.columns and "ply_number" not in df_moves.columns:
            df_moves["ply_number"] = df_moves["ply"]

    w_feat = report.get("white_features", {})
    b_feat = report.get("black_features", {})
    descriptions = get_feature_descriptions()

    # Build 13-feature comparison DataFrame
    comparison_records = []
    from chess_eda.ml.features import ML_FEATURE_COLUMNS

    for col in ML_FEATURE_COLUMNS:
        w_val = w_feat.get(col)
        b_val = b_feat.get(col)

        def _fmt(v: Any) -> str:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "N/A"
            if isinstance(v, float):
                if "rate" in col:
                    return f"{v * 100:.1f}%"
                return f"{v:.2f}"
            return str(v)

        comparison_records.append({
            "Feature": col,
            "Description": descriptions.get(col, ""),
            "White": _fmt(w_val),
            "Black": _fmt(b_val),
            "White_raw": w_val if isinstance(w_val, (int, float)) else None,
            "Black_raw": b_val if isinstance(b_val, (int, float)) else None,
        })

    df_comparison = pd.DataFrame(comparison_records)

    # Move quality category counts
    # Excellent (<= 25), Inaccuracy (50-100], Mistake (100-200], Blunder (> 200)
    w_breakdown = {
        "Top-1 Engine Match": w_feat.get("top1_match_rate", 0.0) * 100 if w_feat.get("top1_match_rate") else 0.0,
        "Excellent (<=25 cp)": w_feat.get("excellent_rate", 0.0) * 100 if w_feat.get("excellent_rate") else 0.0,
        "Inaccuracy (50-100 cp)": w_feat.get("inaccuracy_rate", 0.0) * 100 if w_feat.get("inaccuracy_rate") else 0.0,
        "Mistake (100-200 cp)": w_feat.get("mistake_rate", 0.0) * 100 if w_feat.get("mistake_rate") else 0.0,
        "Blunder (>200 cp)": w_feat.get("blunder_rate", 0.0) * 100 if w_feat.get("blunder_rate") else 0.0,
    }

    b_breakdown = {
        "Top-1 Engine Match": b_feat.get("top1_match_rate", 0.0) * 100 if b_feat.get("top1_match_rate") else 0.0,
        "Excellent (<=25 cp)": b_feat.get("excellent_rate", 0.0) * 100 if b_feat.get("excellent_rate") else 0.0,
        "Inaccuracy (50-100 cp)": b_feat.get("inaccuracy_rate", 0.0) * 100 if b_feat.get("inaccuracy_rate") else 0.0,
        "Mistake (100-200 cp)": b_feat.get("mistake_rate", 0.0) * 100 if b_feat.get("mistake_rate") else 0.0,
        "Blunder (>200 cp)": b_feat.get("blunder_rate", 0.0) * 100 if b_feat.get("blunder_rate") else 0.0,
    }

    df_quality = pd.DataFrame([
        {"Category": cat, "White (%)": round(w_breakdown[cat], 1), "Black (%)": round(b_breakdown[cat], 1)}
        for cat in w_breakdown
    ])

    return {
        "df_moves": df_moves,
        "df_comparison": df_comparison,
        "df_quality": df_quality,
        "w_breakdown": w_breakdown,
        "b_breakdown": b_breakdown,
    }


def get_player_card_presentation_data(player_data: dict[str, Any]) -> dict[str, Any]:
    """Extract and format presentation values for a player card based on model classification.

    Determines whether Engine-assistance score or Human-like score receives primary visual
    emphasis in the dashboard UI:
    - Case A: Classification is 'engine-assisted':
      Primary: 'Engine-assistance score' (large hero score, progress bar, amber badge).
      Secondary: 'Human-like score' (smaller secondary text).
    - Case B: Classification is 'human-like':
      Primary: 'Human-like score' (large hero score, progress bar, green badge).
      Secondary: 'Engine-assistance score' (smaller secondary text).

    Both scores remain visible, retaining full transparency.
    """
    classification = str(player_data.get("classification", "human-like")).lower()
    is_assisted = (classification == "engine-assisted")

    assist_prob = float(player_data.get("engine_assisted_likelihood", 0.0))
    human_prob = float(player_data.get("human_like_likelihood", 1.0 - assist_prob))

    assist_pct = assist_prob * 100.0
    human_pct = human_prob * 100.0

    if is_assisted:
        badge_text = "ENGINE-ASSISTED"
        badge_class = "badge-assisted"
        primary_label = "Engine-assistance score"
        primary_pct = assist_pct
        primary_prob = assist_prob
        primary_color_class = "score-assisted"
        secondary_label = "Human-like score"
        secondary_pct = human_pct
        secondary_prob = human_prob
    else:
        badge_text = "HUMAN-LIKE"
        badge_class = "badge-human"
        primary_label = "Human-like score"
        primary_pct = human_pct
        primary_prob = human_prob
        primary_color_class = "score-human"
        secondary_label = "Engine-assistance score"
        secondary_pct = assist_pct
        secondary_prob = assist_prob

    return {
        "classification": classification,
        "is_assisted": is_assisted,
        "badge_text": badge_text,
        "badge_class": badge_class,
        "primary_label": primary_label,
        "primary_pct": primary_pct,
        "primary_prob": primary_prob,
        "primary_color_class": primary_color_class,
        "secondary_label": secondary_label,
        "secondary_pct": secondary_pct,
        "secondary_prob": secondary_prob,
        "assist_pct": assist_pct,
        "human_pct": human_pct,
    }

