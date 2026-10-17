"""
chess_eda.aggregation
=====================
Aggregates move-level features to player-game-level behavioral profiles.

Each row in the resulting dataset represents one player's performance
in a single game (two rows per completed game: White and Black).

Mathematical & Denominator Rules:
---------------------------------
* ``scored_moves`` : Moves evaluated by Stockfish with a valid numeric
  centipawn loss (excludes unanalyzed plies, book positions with no evaluation,
  and forced checkmate sequences where centipawn diff is undefined).
* ``top1_match_rate``, ``top3_match_rate``, ``blunder_rate`` :
  Calculated strictly with ``scored_moves`` as denominator.
  If ``scored_moves == 0``, the value is an honest empty string (``""``).
* ``missing_eval_rate`` :
  Calculated with ``total_moves`` as denominator.
* Centipawn loss statistics (mean, median, std) are computed only across
  ``scored_moves``.

Notice:
These metrics are purely descriptive behavioral summaries. They do NOT
constitute a cheating verdict or machine learning classification.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def _compute_stats(values: list[int | float]) -> tuple[Any, Any, Any, Any, Any]:
    """Compute mean, median, sample std, min, max for a list of numbers.

    Returns (mean, median, std, min, max) rounded to 2 decimals,
    or empty strings if values is empty.
    """
    if not values:
        return "", "", "", "", ""

    mean_val = round(float(statistics.mean(values)), 2)
    med_val = round(float(statistics.median(values)), 2)
    min_val = min(values)
    max_val = max(values)

    if len(values) >= 2:
        std_val = round(float(statistics.stdev(values)), 2)
    elif len(values) == 1:
        std_val = 0.0
    else:
        std_val = ""

    return mean_val, med_val, std_val, min_val, max_val


def _rate(numerator: int, denominator: int) -> Any:
    """Safe rate calculation rounded to 4 decimals, or empty string."""
    if denominator <= 0:
        return ""
    return round(numerator / denominator, 4)


def aggregate_player_game_features(
    move_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate a list of move rows into player-game-level summaries.

    Parameters
    ----------
    move_rows:
        Enriched move-level rows containing both base features and engine analysis.

    Returns
    -------
    list[dict[str, Any]]
        One dict per player per game (ordered by game_id and player_color).
    """
    if not move_rows:
        return []

    # Group moves by (game_id, player_color) preserving first-seen order
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in move_rows:
        key = (row.get("game_id", 1), row.get("player_color", "White"))
        groups[key].append(row)

    aggregated: list[dict[str, Any]] = []

    for (game_id, color), plies in groups.items():
        sample = plies[0]
        source_file = sample.get("source_file", "")
        source_cat = sample.get("source_category", "unlabeled")
        event = sample.get("event", "")
        site = sample.get("site", "")
        date = sample.get("date", "")
        round_ = sample.get("round", "")
        player_name = sample.get("player_name", "")
        player_elo = sample.get("player_elo", "")
        opp_name = sample.get("opponent_name", "")
        opp_elo = sample.get("opponent_elo", "")
        result = sample.get("result", "")

        # Outcome from this player's perspective
        if result == "1-0":
            player_result = "win" if color == "White" else "loss"
        elif result == "0-1":
            player_result = "loss" if color == "White" else "win"
        elif result == "1/2-1/2":
            player_result = "draw"
        else:
            player_result = "unknown"

        total_moves = len(plies)

        # Separate scored moves from unscored/mate
        scored_losses: list[int] = []
        top1_count = 0
        top3_count = 0
        inacc_count = 0
        mistake_count = 0
        blunder_count = 0
        mate_count = 0
        missing_count = 0

        # Phase breakdowns
        phase_moves = {"Opening": 0, "Middlegame": 0, "Endgame": 0}
        phase_scored_losses: dict[str, list[int]] = {
            "Opening": [],
            "Middlegame": [],
            "Endgame": [],
        }
        phase_top1 = {"Opening": 0, "Middlegame": 0, "Endgame": 0}

        captures = 0
        checks = 0
        promotions = 0

        for ply in plies:
            phase = ply.get("game_phase", "Middlegame")
            if phase in phase_moves:
                phase_moves[phase] += 1

            captures += int(ply.get("is_capture") or 0)
            checks += int(ply.get("is_check") or 0)
            promotions += int(ply.get("is_promotion") or 0)

            is_missing = ply.get("is_engine_eval_missing") == 1
            if is_missing:
                missing_count += 1

            is_mate = ply.get("is_forced_mate") == 1
            if is_mate:
                mate_count += 1

            cp_loss = ply.get("centipawn_loss", "")
            if cp_loss != "" and cp_loss is not None:
                try:
                    cp_int = int(cp_loss)
                    scored_losses.append(cp_int)
                    if phase in phase_scored_losses:
                        phase_scored_losses[phase].append(cp_int)

                    if cp_int > 200:
                        blunder_count += 1
                    elif cp_int > 100:
                        mistake_count += 1
                    elif cp_int > 50:
                        inacc_count += 1

                except (ValueError, TypeError):
                    pass

            # Top candidate matches (evaluated over scored moves)
            if ply.get("is_engine_top1") == 1:
                top1_count += 1
                if phase in phase_top1:
                    phase_top1[phase] += 1
            if ply.get("is_engine_top3") == 1:
                top3_count += 1

        scored_count = len(scored_losses)
        mean_cp, med_cp, std_cp, min_cp, max_cp = _compute_stats(scored_losses)

        # Missing eval rate
        missing_rate = _rate(missing_count, total_moves)

        # Engine match rates (denominator = scored_moves)
        top1_rate = _rate(top1_count, scored_count)
        top3_rate = _rate(top3_count, scored_count)
        inacc_rate = _rate(inacc_count, scored_count)
        mistake_rate = _rate(mistake_count, scored_count)
        blunder_rate = _rate(blunder_count, scored_count)

        # Phase statistics
        op_scored = len(phase_scored_losses["Opening"])
        mg_scored = len(phase_scored_losses["Middlegame"])
        eg_scored = len(phase_scored_losses["Endgame"])

        op_mean = round(float(statistics.mean(phase_scored_losses["Opening"])), 2) if op_scored > 0 else ""
        mg_mean = round(float(statistics.mean(phase_scored_losses["Middlegame"])), 2) if mg_scored > 0 else ""
        eg_mean = round(float(statistics.mean(phase_scored_losses["Endgame"])), 2) if eg_scored > 0 else ""

        op_top1_rate = _rate(phase_top1["Opening"], op_scored)
        mg_top1_rate = _rate(phase_top1["Middlegame"], mg_scored)
        eg_top1_rate = _rate(phase_top1["Endgame"], eg_scored)

        aggregated.append({
            "game_id": game_id,
            "source_file": source_file,
            "source_category": source_cat,
            "event": event,
            "site": site,
            "date": date,
            "round": round_,
            "player_color": color,
            "player_name": player_name,
            "player_elo": player_elo,
            "opponent_name": opp_name,
            "opponent_elo": opp_elo,
            "result": result,
            "player_result": player_result,
            "total_moves": total_moves,
            "scored_moves": scored_count,
            "missing_eval_count": missing_count,
            "missing_eval_rate": missing_rate if missing_rate != "" else 0.0,
            "mate_eval_moves": mate_count,
            "mean_centipawn_loss": mean_cp,
            "median_centipawn_loss": med_cp,
            "std_centipawn_loss": std_cp,
            "min_centipawn_loss": min_cp,
            "max_centipawn_loss": max_cp,
            "top1_match_count": top1_count,
            "top1_match_rate": top1_rate,
            "top3_match_count": top3_count,
            "top3_match_rate": top3_rate,
            "inaccuracy_count": inacc_count,
            "inaccuracy_rate": inacc_rate,
            "mistake_count": mistake_count,
            "mistake_rate": mistake_rate,
            "blunder_count": blunder_count,
            "blunder_rate": blunder_rate,
            "opening_moves": phase_moves["Opening"],
            "opening_scored_moves": op_scored,
            "opening_mean_cp_loss": op_mean,
            "opening_top1_rate": op_top1_rate,
            "middlegame_moves": phase_moves["Middlegame"],
            "middlegame_scored_moves": mg_scored,
            "middlegame_mean_cp_loss": mg_mean,
            "middlegame_top1_rate": mg_top1_rate,
            "endgame_moves": phase_moves["Endgame"],
            "endgame_scored_moves": eg_scored,
            "endgame_mean_cp_loss": eg_mean,
            "endgame_top1_rate": eg_top1_rate,
            "captures_count": captures,
            "checks_count": checks,
            "promotions_count": promotions,
        })

    return aggregated


PLAYER_GAME_COLUMNS = [
    "game_id",
    "source_file",
    "source_category",
    "event",
    "site",
    "date",
    "round",
    "player_color",
    "player_name",
    "player_elo",
    "opponent_name",
    "opponent_elo",
    "result",
    "player_result",
    "total_moves",
    "scored_moves",
    "missing_eval_count",
    "missing_eval_rate",
    "mate_eval_moves",
    "mean_centipawn_loss",
    "median_centipawn_loss",
    "std_centipawn_loss",
    "min_centipawn_loss",
    "max_centipawn_loss",
    "top1_match_count",
    "top1_match_rate",
    "top3_match_count",
    "top3_match_rate",
    "inaccuracy_count",
    "inaccuracy_rate",
    "mistake_count",
    "mistake_rate",
    "blunder_count",
    "blunder_rate",
    "opening_moves",
    "opening_scored_moves",
    "opening_mean_cp_loss",
    "opening_top1_rate",
    "middlegame_moves",
    "middlegame_scored_moves",
    "middlegame_mean_cp_loss",
    "middlegame_top1_rate",
    "endgame_moves",
    "endgame_scored_moves",
    "endgame_mean_cp_loss",
    "endgame_top1_rate",
    "captures_count",
    "checks_count",
    "promotions_count",
]
