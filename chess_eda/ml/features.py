"""
chess_eda.ml.features
=====================
Feature extraction, normalization, and schema alignment for Version 3 ML baseline.

Features are designed to be strictly compatible between:
1. The ChessFraud benchmark dataset (38,510 plies / 1,010 player games)
2. Local PGN files analyzed with Stockfish via the Version 1/2 pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


ML_FEATURE_COLUMNS: list[str] = [
    "mean_centipawn_loss",
    "std_centipawn_loss",
    "median_centipawn_loss",
    "mean_norm_cp_loss",
    "std_norm_cp_loss",
    "blunder_rate",       # centipawn_loss > 200
    "mistake_rate",       # 100 < centipawn_loss <= 200
    "inaccuracy_rate",    # 50 < centipawn_loss <= 100
    "excellent_rate",     # centipawn_loss <= 25
    "top1_match_rate",    # engine top-1 match rate on post-opening moves
    "used_mean_cp_loss",  # mean centipawn loss on post-opening moves
    "used_blunder_rate",  # blunder rate on post-opening moves
    "move_count",         # total moves played by this player
]

V5_COMPLEXITY_FEATURE_COLUMNS: list[str] = [
    "unstable_top1_match_rate",  # Engine match on moves where shallow depth 1 != depth 15 (deep/complex moves)
    "stable_top1_match_rate",    # Engine match on moves where shallow depth 1 == depth 15 (shallow/obvious moves)
    "depth_drop_rate",          # Difference between stable and unstable match rate (measures calculate-intensity drop)
    "critical_top1_match_rate", # Engine match in balanced/critical positions (|eval_before| <= 200 cp)
    "critical_mean_cp_loss",    # Mean centipawn loss in balanced/critical positions (|eval_before| <= 200 cp)
]

V5_ALL_FEATURE_COLUMNS: list[str] = ML_FEATURE_COLUMNS + V5_COMPLEXITY_FEATURE_COLUMNS


def compute_normalized_cp_loss(eval_before: float, eval_after: float) -> float:
    """Compute normalized centipawn loss via the Lichess Win% model.

    Formula:
    --------
    p(e) = 1 / (1 + exp(-k * e)), where k = 0.00368208
    normalized_centipawn_loss = max(0, p(eval_before) - p(eval_after))
    """
    e_bef = max(-1000.0, min(1000.0, float(eval_before)))
    e_aft = max(-1000.0, min(1000.0, float(eval_after)))
    k = 0.00368208
    p_bef = 1.0 / (1.0 + math.exp(-k * e_bef))
    p_aft = 1.0 / (1.0 + math.exp(-k * e_aft))
    return float(max(0.0, p_bef - p_aft))


def extract_physical_match_id(game_id: str | int) -> str:
    """Normalize game_id to base physical match identifier.

    In the ChessFraud benchmark, each physical match generates two player-game
    records ending in '_w' (White perspective) and '_b' (Black perspective).
    Stripping this suffix groups the paired records into the exact same physical match.
    """
    s = str(game_id)
    if s.endswith(("_w", "_b")):
        return s[:-2]
    return s


def extract_player_features_from_df(
    full_df: pd.DataFrame,
    include_complexity: bool = False,
) -> pd.DataFrame:
    """Extract player-game feature vectors from the ChessFraud DataFrame.

    Parameters
    ----------
    full_df:
        The raw ChessFraud DataFrame (38,510 rows).
    include_complexity:
        If True, include Version 5 position-complexity features
        (V5_COMPLEXITY_FEATURE_COLUMNS). Defaults to False for baseline backward-compatibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 1,010 rows (one per player-game) and feature columns
        plus identifiers (`game_id`, `physical_game_id`, `player_id`, `is_cheating`).
    """
    records: list[dict[str, Any]] = []

    for game_id, group in full_df.groupby("game_id"):
        used_group = group[group["is_used"] == True]

        is_cheat = int(bool(group["is_cheating_player_game"].iloc[0]))
        player_id = group["player_id"].iloc[0]
        color = group["player_color"].iloc[0]
        physical_id = extract_physical_match_id(game_id)

        cp_losses = group["centipawn_loss"].dropna().values.astype(float)
        mean_cp = float(np.mean(cp_losses)) if len(cp_losses) > 0 else np.nan
        std_cp = float(np.std(cp_losses)) if len(cp_losses) > 1 else (0.0 if len(cp_losses) == 1 else np.nan)
        med_cp = float(np.median(cp_losses)) if len(cp_losses) > 0 else np.nan

        norm_cp = group["normalized_centipawn_loss"].dropna().values.astype(float)
        mean_norm_cp = float(np.mean(norm_cp)) if len(norm_cp) > 0 else np.nan
        std_norm_cp = float(np.std(norm_cp)) if len(norm_cp) > 1 else (0.0 if len(norm_cp) == 1 else np.nan)

        blunder_rate = float(np.mean(cp_losses > 200)) if len(cp_losses) > 0 else np.nan
        mistake_rate = float(np.mean((cp_losses > 100) & (cp_losses <= 200))) if len(cp_losses) > 0 else np.nan
        inaccuracy_rate = float(np.mean((cp_losses > 50) & (cp_losses <= 100))) if len(cp_losses) > 0 else np.nan
        excellent_rate = float(np.mean(cp_losses <= 25)) if len(cp_losses) > 0 else np.nan

        # Top-1 match on post-opening moves (SF15 match in benchmark)
        if len(used_group) > 0:
            top1_matches = (used_group["move_player"] == used_group["move_stockfish_15"]).values
            top1_rate = float(np.mean(top1_matches))
            used_cps = used_group["centipawn_loss"].values.astype(float)
            used_mean_cp = float(np.mean(used_cps))
            used_blunder_rate = float(np.mean(used_cps > 100))
        else:
            # Fallback if no moves >= 21
            top1_rate = np.nan
            used_mean_cp = np.nan
            used_blunder_rate = np.nan

        rec: dict[str, Any] = {
            "game_id": game_id,
            "physical_game_id": physical_id,
            "player_id": player_id,
            "player_color": color,
            "is_cheating": is_cheat,
            "mean_centipawn_loss": mean_cp,
            "std_centipawn_loss": std_cp,
            "median_centipawn_loss": med_cp,
            "mean_norm_cp_loss": mean_norm_cp,
            "std_norm_cp_loss": std_norm_cp,
            "blunder_rate": blunder_rate,
            "mistake_rate": mistake_rate,
            "inaccuracy_rate": inaccuracy_rate,
            "excellent_rate": excellent_rate,
            "top1_match_rate": top1_rate,
            "used_mean_cp_loss": used_mean_cp,
            "used_blunder_rate": used_blunder_rate,
            "move_count": len(group),
        }

        if include_complexity:
            # 1. Depth-stability: moves where shallow depth 1 disagrees with depth 15
            if len(used_group) > 0 and "move_stockfish_1" in used_group.columns and "move_stockfish_15" in used_group.columns:
                is_unstable = (used_group["move_stockfish_1"] != used_group["move_stockfish_15"]).values
                unstable_moves = used_group[is_unstable]
                top1_unstable = (
                    float(np.mean(unstable_moves["move_player"] == unstable_moves["move_stockfish_15"]))
                    if len(unstable_moves) > 0 else np.nan
                )

                stable_moves = used_group[~is_unstable]
                top1_stable = (
                    float(np.mean(stable_moves["move_player"] == stable_moves["move_stockfish_15"]))
                    if len(stable_moves) > 0 else np.nan
                )

                depth_drop = (
                    float(top1_stable - top1_unstable)
                    if (not np.isnan(top1_stable) and not np.isnan(top1_unstable)) else np.nan
                )
            else:
                top1_unstable = np.nan
                top1_stable = np.nan
                depth_drop = np.nan

            # 2. Critical/balanced positions: |eval_before| <= 200 cp
            if len(used_group) > 0 and "eval_before" in used_group.columns:
                crit_moves = used_group[used_group["eval_before"].abs() <= 200]
                top1_crit = (
                    float(np.mean(crit_moves["move_player"] == crit_moves["move_stockfish_15"]))
                    if len(crit_moves) > 0 else np.nan
                )
                crit_cps = crit_moves["centipawn_loss"].dropna().values.astype(float)
                mean_cp_crit = float(np.mean(crit_cps)) if len(crit_cps) > 0 else np.nan
            else:
                top1_crit = np.nan
                mean_cp_crit = np.nan

            rec["unstable_top1_match_rate"] = top1_unstable
            rec["stable_top1_match_rate"] = top1_stable
            rec["depth_drop_rate"] = depth_drop
            rec["critical_top1_match_rate"] = top1_crit
            rec["critical_mean_cp_loss"] = mean_cp_crit

        records.append(rec)

    return pd.DataFrame(records)


def extract_player_features_from_engine_rows(
    merged_rows: list[dict[str, Any]],
    player_color: str,
    include_complexity: bool = False,
) -> dict[str, Any]:
    """Extract the aligned feature vector for one player in a game evaluated with Stockfish.

    Parameters
    ----------
    merged_rows:
        Rows enriched with move features and Stockfish analysis from `analyse_game`.
    player_color:
        'White' or 'Black'.
    include_complexity:
        If True, include Version 5 position-complexity features.

    Returns
    -------
    dict[str, Any]
        Dictionary matching ML_FEATURE_COLUMNS or V5_ALL_FEATURE_COLUMNS.
    """
    color_lower = player_color.lower()
    # Filter plies made by this player
    plies = [r for r in merged_rows if str(r.get("player_color", "")).lower() == color_lower]

    target_cols = V5_ALL_FEATURE_COLUMNS if include_complexity else ML_FEATURE_COLUMNS
    if not plies:
        return {col: np.nan for col in target_cols}

    cp_losses: list[float] = []
    norm_cp_losses: list[float] = []
    used_top1_matches: list[int] = []
    used_cp_losses: list[float] = []

    crit_top1_matches: list[int] = []
    crit_cp_losses: list[float] = []
    unstable_top1_matches: list[int] = []
    stable_top1_matches: list[int] = []

    for r in plies:
        # Centipawn loss
        raw_cp = r.get("centipawn_loss")
        has_cp = False
        cp_val = 0.0
        if raw_cp not in ("", None) and not str(raw_cp).startswith("#"):
            try:
                cp_val = float(raw_cp)
                cp_losses.append(cp_val)
                has_cp = True
            except (ValueError, TypeError):
                pass

        # Normalized centipawn loss from score_before / score_after or engine_score_before_cp / engine_score_after_cp
        s_bef = r.get("score_before") if r.get("score_before") not in ("", None) else r.get("engine_score_before_cp")
        s_aft = r.get("score_after") if r.get("score_after") not in ("", None) else r.get("engine_score_after_cp")
        if s_bef not in ("", None) and s_aft not in ("", None):
            if not str(s_bef).startswith("#") and not str(s_aft).startswith("#"):
                try:
                    # score_before/after in engine_analysis are White POV.
                    # Convert to player POV:
                    sb = float(s_bef) if color_lower == "white" else -float(s_bef)
                    sa = float(s_aft) if color_lower == "white" else -float(s_aft)
                    norm_val = compute_normalized_cp_loss(sb, sa)
                    norm_cp_losses.append(norm_val)
                except (ValueError, TypeError):
                    pass

        # Decision point: half_move >= 21 (plies >= 21)
        ply_num = int(r.get("ply_number", 0) or 0)
        is_used = ply_num >= 21

        is_top1 = int(r.get("is_engine_top1") or 0)
        if is_used:
            used_top1_matches.append(is_top1)
            if has_cp:
                used_cp_losses.append(cp_val)

            # Complexity checks
            if include_complexity:
                if s_bef not in ("", None) and not str(s_bef).startswith("#"):
                    try:
                        if abs(float(s_bef)) <= 200:
                            crit_top1_matches.append(is_top1)
                            if has_cp:
                                crit_cp_losses.append(cp_val)
                    except (ValueError, TypeError):
                        pass

                d1_move = r.get("engine_best_move_depth1") or r.get("move_depth1")
                best_move = r.get("engine_best_move_uci")
                played_move = r.get("uci")
                if d1_move and best_move and played_move:
                    if d1_move != best_move:
                        unstable_top1_matches.append(int(played_move == best_move))
                    else:
                        stable_top1_matches.append(int(played_move == best_move))

    # If game ended early and no moves >= 21, use all moves as decision points
    if not used_top1_matches and plies:
        for r in plies:
            used_top1_matches.append(int(r.get("is_engine_top1") or 0))
            raw_cp = r.get("centipawn_loss")
            if raw_cp not in ("", None) and not str(raw_cp).startswith("#"):
                try:
                    used_cp_losses.append(float(raw_cp))
                except (ValueError, TypeError):
                    pass

    # Aggregations
    cp_arr = np.array(cp_losses) if cp_losses else np.array([])
    norm_arr = np.array(norm_cp_losses) if norm_cp_losses else np.array([])

    mean_cp = float(np.mean(cp_arr)) if len(cp_arr) > 0 else np.nan
    std_cp = float(np.std(cp_arr)) if len(cp_arr) > 1 else (0.0 if len(cp_arr) == 1 else np.nan)
    med_cp = float(np.median(cp_arr)) if len(cp_arr) > 0 else np.nan

    mean_norm = float(np.mean(norm_arr)) if len(norm_arr) > 0 else np.nan
    std_norm = float(np.std(norm_arr)) if len(norm_arr) > 1 else (0.0 if len(norm_arr) == 1 else np.nan)

    blunder_rate = float(np.mean(cp_arr > 200)) if len(cp_arr) > 0 else np.nan
    mistake_rate = float(np.mean((cp_arr > 100) & (cp_arr <= 200))) if len(cp_arr) > 0 else np.nan
    inacc_rate = float(np.mean((cp_arr > 50) & (cp_arr <= 100))) if len(cp_arr) > 0 else np.nan
    ex_rate = float(np.mean(cp_arr <= 25)) if len(cp_arr) > 0 else np.nan

    top1_rate = float(np.mean(used_top1_matches)) if used_top1_matches else np.nan
    used_cp_arr = np.array(used_cp_losses) if used_cp_losses else np.array([])
    used_mean_cp = float(np.mean(used_cp_arr)) if len(used_cp_arr) > 0 else np.nan
    used_blunder = float(np.mean(used_cp_arr > 100)) if len(used_cp_arr) > 0 else np.nan

    res: dict[str, Any] = {
        "mean_centipawn_loss": mean_cp,
        "std_centipawn_loss": std_cp,
        "median_centipawn_loss": med_cp,
        "mean_norm_cp_loss": mean_norm,
        "std_norm_cp_loss": std_norm,
        "blunder_rate": blunder_rate,
        "mistake_rate": mistake_rate,
        "inaccuracy_rate": inacc_rate,
        "excellent_rate": ex_rate,
        "top1_match_rate": top1_rate,
        "used_mean_cp_loss": used_mean_cp,
        "used_blunder_rate": used_blunder,
        "move_count": len(plies),
    }

    if include_complexity:
        top1_unst = float(np.mean(unstable_top1_matches)) if unstable_top1_matches else np.nan
        top1_st = float(np.mean(stable_top1_matches)) if stable_top1_matches else np.nan
        depth_drop = (
            float(top1_st - top1_unst)
            if (not np.isnan(top1_st) and not np.isnan(top1_unst))
            else np.nan
        )
        res["unstable_top1_match_rate"] = top1_unst
        res["stable_top1_match_rate"] = top1_st
        res["depth_drop_rate"] = depth_drop
        res["critical_top1_match_rate"] = float(np.mean(crit_top1_matches)) if crit_top1_matches else np.nan
        res["critical_mean_cp_loss"] = float(np.mean(crit_cp_losses)) if crit_cp_losses else np.nan

    return res
