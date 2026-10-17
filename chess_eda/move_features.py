"""
chess_eda.move_features
=======================
Extracts rich move-level behavioral features from chess games, positions,
and Stockfish engine evaluations.

Version 2 extensions
--------------------
- Board-derived context BEFORE move: legal move count, capture, check, promotion.
- Clock / thinking time extraction (strictly when real annotations exist).
- Quality classifications (Best, Excellent, Good, Inaccuracy, Mistake, Blunder).
- Forced mate indicators and missing-evaluation flags.
- Source tracking (source file, category).
"""

from __future__ import annotations

from typing import Any, Optional

import chess
import chess.pgn

from chess_eda.feature_extraction import (
    _header,
    _game_phase,
)


def _classify_move_quality(
    centipawn_loss: Any,
    is_top1: Any,
    is_forced_mate: Any,
    is_missing: bool,
) -> str:
    """Classify the quality of a move according to standard centipawn loss tiers.

    Tiers:
      - "Forced Mate": move is part of a forced checkmate sequence
      - "Best"       : engine top choice or 0 centipawn loss
      - "Excellent"  : 1 to 25 centipawns lost
      - "Good"       : 26 to 50 centipawns lost
      - "Inaccuracy" : 51 to 100 centipawns lost
      - "Mistake"    : 101 to 200 centipawns lost
      - "Blunder"    : > 200 centipawns lost
      - ""           : unanalyzed or missing engine evaluation
    """
    if is_missing or centipawn_loss == "" or centipawn_loss is None:
        if is_forced_mate == 1:
            return "Forced Mate"
        return ""

    try:
        cp = int(centipawn_loss)
    except (ValueError, TypeError):
        return ""

    if cp == 0 or is_top1 == 1:
        return "Best"
    if cp <= 25:
        return "Excellent"
    if cp <= 50:
        return "Good"
    if cp <= 100:
        return "Inaccuracy"
    if cp <= 200:
        return "Mistake"
    return "Blunder"


def extract_move_features(
    game: chess.pgn.Game,
    game_id: int | str = 1,
    source_file: str = "",
    source_category: str = "unlabeled",
) -> list[dict[str, Any]]:
    """Extract move-level features for every mainline ply in *game*.

    All board-derived features (legal moves count, capture, check, promotion)
    are strictly computed on the position BEFORE the move is executed.

    Parameters
    ----------
    game:
        A parsed ``chess.pgn.Game``.
    game_id:
        Identifier for the game.
    source_file:
        Originating PGN filename or relative path.
    source_category:
        Provenance category: 'human_games', 'engine_generated_games', or 'unlabeled'.

    Returns
    -------
    list[dict[str, Any]]
        One dict per ply containing base move and board features.
    """
    event = _header(game, "Event")
    site = _header(game, "Site")
    date = _header(game, "Date")
    round_ = _header(game, "Round")
    white_name = _header(game, "White")
    black_name = _header(game, "Black")
    white_elo = _header(game, "WhiteElo")
    black_elo = _header(game, "BlackElo")
    result = _header(game, "Result")

    rows: list[dict[str, Any]] = []
    board = game.board()
    node: Optional[chess.pgn.GameNode] = game.next()
    ply_number = 0

    while node is not None:
        move: chess.Move = node.move  # type: ignore[assignment]
        ply_number += 1

        # Position snapshot BEFORE move
        fen_before = board.fen()
        turn = board.turn
        fullmove_num = board.fullmove_number
        phase = _game_phase(board)

        # Tactical / board-derived features BEFORE push
        legal_moves_count = len(list(board.legal_moves))
        is_capture = int(board.is_capture(move))
        is_check = int(board.gives_check(move))
        is_promotion = int(bool(move.promotion))

        # Real clock annotation if present
        clock_val = getattr(node, "clock", lambda: None)()
        clock_seconds: str | float = round(float(clock_val), 1) if clock_val is not None else ""

        # Player perspective fields
        if turn == chess.WHITE:
            player_color = "White"
            player_name = white_name
            player_elo = white_elo
            opponent_name = black_name
            opponent_elo = black_elo
        else:
            player_color = "Black"
            player_name = black_name
            player_elo = black_elo
            opponent_name = white_name
            opponent_elo = white_elo

        san = board.san(move)
        uci = move.uci()

        rows.append({
            "game_id": game_id,
            "source_file": source_file,
            "source_category": source_category,
            "event": event,
            "site": site,
            "date": date,
            "round": round_,
            "white_player": white_name,
            "black_player": black_name,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": result,
            "fullmove_number": fullmove_num,
            "ply_number": ply_number,
            "player_color": player_color,
            "player_name": player_name,
            "player_elo": player_elo,
            "opponent_name": opponent_name,
            "opponent_elo": opponent_elo,
            "san": san,
            "uci": uci,
            "fen_before": fen_before,
            "game_phase": phase,
            "legal_moves_count": legal_moves_count,
            "is_capture": is_capture,
            "is_check": is_check,
            "is_promotion": is_promotion,
            "clock_seconds": clock_seconds,
        })

        board.push(move)
        node = node.next()

    return rows


def enrich_moves_with_engine(
    move_rows: list[dict[str, Any]],
    engine_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge move rows with Stockfish engine evaluations and derived quality features.

    Parameters
    ----------
    move_rows:
        Rows from ``extract_move_features()``.
    engine_rows:
        Rows from ``analyse_game()``, or None if engine analysis was skipped.

    Returns
    -------
    list[dict[str, Any]]
        Enriched move rows containing all 37 move-level feature columns.
    """
    enriched: list[dict[str, Any]] = []

    for i, base in enumerate(move_rows):
        eng = engine_rows[i] if (engine_rows and i < len(engine_rows)) else None

        if eng:
            engine_best_uci = eng.get("engine_best_move_uci", "")
            engine_best_san = eng.get("engine_best_move_san", "")
            rank = eng.get("played_move_rank_top5", "")
            top1 = eng.get("is_engine_top1", "")
            top3 = eng.get("is_engine_top3", "")
            cp_loss = eng.get("centipawn_loss", "")
            gap = eng.get("best_vs_second_cp_gap", "")
            s_before = eng.get("engine_score_before_cp", "")
            s_after = eng.get("engine_score_after_cp", "")

            # Check for mate sentinels (#+N, #-N)
            is_mate = int(
                (isinstance(s_before, str) and s_before.startswith("#"))
                or (isinstance(s_after, str) and s_after.startswith("#"))
            )

            # Missing evaluation indicator
            is_missing = int(
                s_before == "" and s_after == "" and cp_loss == ""
            )

            quality = _classify_move_quality(cp_loss, top1, is_mate, bool(is_missing))
            d1_best = eng.get("engine_best_move_depth1", "")

        else:
            engine_best_uci = ""
            engine_best_san = ""
            d1_best = ""
            rank = ""
            top1 = ""
            top3 = ""
            cp_loss = ""
            gap = ""
            s_before = ""
            s_after = ""
            is_mate = ""
            is_missing = 1
            quality = ""

        row = dict(base)
        row.update({
            "engine_best_move_uci": engine_best_uci,
            "engine_best_move_san": engine_best_san,
            "engine_best_move_depth1": d1_best,
            "played_move_rank_top5": rank,
            "is_engine_top1": top1,
            "is_engine_top3": top3,
            "centipawn_loss": cp_loss,
            "best_vs_second_cp_gap": gap,
            "engine_score_before_cp": s_before,
            "engine_score_after_cp": s_after,
            "is_forced_mate": is_mate,
            "is_engine_eval_missing": is_missing,
            "move_quality": quality,
        })
        enriched.append(row)

    return enriched


MOVE_FEATURE_COLUMNS = [
    "game_id",
    "source_file",
    "source_category",
    "event",
    "site",
    "date",
    "round",
    "white_player",
    "black_player",
    "white_elo",
    "black_elo",
    "result",
    "fullmove_number",
    "ply_number",
    "player_color",
    "player_name",
    "player_elo",
    "opponent_name",
    "opponent_elo",
    "san",
    "uci",
    "fen_before",
    "game_phase",
    "legal_moves_count",
    "is_capture",
    "is_check",
    "is_promotion",
    "clock_seconds",
    "engine_best_move_uci",
    "engine_best_move_san",
    "played_move_rank_top5",
    "is_engine_top1",
    "is_engine_top3",
    "centipawn_loss",
    "best_vs_second_cp_gap",
    "engine_score_before_cp",
    "engine_score_after_cp",
    "is_forced_mate",
    "is_engine_eval_missing",
    "move_quality",
]
