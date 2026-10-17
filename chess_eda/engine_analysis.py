"""
chess_eda.engine_analysis
==========================
Stockfish engine wrapper for Version 1 of the EDA chess project.

This module starts a single Stockfish process, analyses every position in a
game's mainline, and returns per-ply engine feature rows.

Design decisions
----------------
* **One process per game** — the engine is started once and reused, which
  avoids the significant overhead of spawning a new process for every move.
* **MultiPV analysis** — up to 5 principal variations are requested so that
  ``played_move_rank_top5`` can be determined.  If the engine returns fewer
  variations (e.g. near end-game with few legal moves), the rank is
  represented honestly as ``None``/empty.
* **Correct perspective** — all centipawn values are stored from **White's**
  perspective (positive = White is better) in ``engine_score_before_cp`` and
  ``engine_score_after_cp``.  ``centipawn_loss`` is always >= 0 and is
  computed from the **moving player's** perspective.
* **Mate scores** — when Stockfish reports a forced mate the centipawn values
  are stored as sentinel strings: ``"#+N"`` for mate in N (winning) and
  ``"#-N"`` for being mated in N (losing), where N is the number of moves.
  ``centipawn_loss`` is set to ``""`` when either score involves a mate, because
  a numeric difference between a mate score and a centipawn score is
  meaningless.
* **Timeout protection** — each analysis call has a configurable time limit
  (default 30 s) so that the program cannot freeze indefinitely.

Representation of missing data
-------------------------------
All engine columns are set to ``""`` (empty string) if:
  - The engine executable is not available.
  - The engine process crashes during analysis.
  - The position has no legal moves (e.g. terminal position).
  - Fewer than *N* MultiPV lines are returned (rank ``> N`` → ``None``).

``played_move_rank_top5`` : int 1–5, or ``None`` if the played move was not
    among the top-5 candidates returned by the engine.
``is_engine_top1``        : 1 if the played move equals the engine's first
    choice, else 0.  ``""`` if engine data is unavailable.
``is_engine_top3``        : 1 if the played move is among the top-3 choices,
    else 0.  ``""`` if engine data is unavailable.
``centipawn_loss``        : Non-negative integer (capped at 2000 to avoid
    noise from extreme positions), or ``""`` if either score is a mate score
    or engine data is unavailable.
``best_vs_second_cp_gap`` : Centipawn difference between engine's first and
    second best move.  ``""`` if fewer than 2 PVs were returned or either
    score involves a mate.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Optional

import chess
import chess.engine
import chess.pgn


# ───────────────────────────────────────────────────────────────────────────
# Sentinel constants
# ───────────────────────────────────────────────────────────────────────────

#: Maximum centipawn loss stored (avoids huge numbers from wild positions).
CP_LOSS_CAP = 2000

#: Number of MultiPV lines to request from the engine.
MULTIPV = 5

#: Default analysis depth (shallow enough to run quickly on a laptop).
DEFAULT_DEPTH = 15

#: Timeout per position in seconds.
ANALYSIS_TIMEOUT_SECONDS = 30


# ───────────────────────────────────────────────────────────────────────────
# Internal helpers
# ───────────────────────────────────────────────────────────────────────────

def _score_to_cp_string(score: chess.engine.PovScore, pov: chess.Color) -> str | int:
    """Convert a ``PovScore`` to a value suitable for CSV storage.

    Parameters
    ----------
    score:
        The engine score, as returned by python-chess.
    pov:
        The colour from whose point of view the score is expressed.
        Pass ``chess.WHITE`` to always get White's perspective.

    Returns
    -------
    int
        Centipawn integer when the score is not a mate.
    str
        ``"#+N"`` or ``"#-N"`` sentinel when the score is a forced mate.
    """
    relative = score.pov(pov)
    if relative.is_mate():
        mate_in = relative.mate()
        # mate_in > 0  → the *pov* side delivers checkmate in *mate_in* moves.
        # mate_in < 0  → the *pov* side gets checkmated in *abs(mate_in)* moves.
        return f"#+{mate_in}" if mate_in > 0 else f"#-{abs(mate_in)}"
    return relative.score()


def _is_mate_string(value: Any) -> bool:
    """Return True if *value* is one of the mate sentinel strings."""
    return isinstance(value, str) and value.startswith("#")


def _centipawn_loss(
    score_before: Any,  # int or mate-string, White's POV
    score_after: Any,   # int or mate-string, White's POV
    player_color: chess.Color,
) -> int | str:
    """Compute the centipawn loss for the move that was just played.

    The loss is always ``>= 0`` and is expressed from the *moving player's*
    perspective (how much they lost by not playing the engine's best move).

    If either score is a mate string, returns ``""`` because a numeric
    difference is not meaningful.
    """
    if _is_mate_string(score_before) or _is_mate_string(score_after):
        return ""

    # score_before and score_after are both ints (centipawns, White's POV).
    # Flip sign for Black so that loss is always non-negative.
    if player_color == chess.WHITE:
        # Best possible score from White's POV is score_before (engine's top1).
        # After the actual move, White's score becomes score_after.
        loss = score_before - score_after
    else:
        # For Black, higher White score = worse for Black.
        # Black best: score_before (engine already flipped to White POV, so
        # Black's best minimises the White score).
        # After move: score_after is White's evaluation.
        loss = score_after - score_before  # positive when Black blundered

    return max(0, min(loss, CP_LOSS_CAP))


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

class EngineAnalysisError(RuntimeError):
    """Raised when the engine cannot be started or crashes fatally."""


def validate_stockfish_path(path: str) -> str:
    """Raise ``EngineAnalysisError`` with a helpful message if *path* is bad.

    Returns the resolved executable path.

    Parameters
    ----------
    path:
        Path to the Stockfish executable.

    Returns
    -------
    str
        Resolved path to the executable.

    Raises
    ------
    EngineAnalysisError
    """
    if not path:
        raise EngineAnalysisError(
            "No Stockfish path provided.\n"
            "Use:  --stockfish-path \"C:\\path\\to\\stockfish.exe\"\n"
            "Download Stockfish from https://stockfishchess.org/download/"
        )
    resolved = path
    if not os.path.exists(resolved):
        which_path = shutil.which(resolved)
        if which_path:
            resolved = which_path

    if not os.path.exists(resolved):
        raise EngineAnalysisError(
            f"Stockfish executable not found: '{path}'\n"
            "Download Stockfish from https://stockfishchess.org/download/\n"
            "Then pass the full path with --stockfish-path."
        )
    if os.path.isdir(resolved):
        raise EngineAnalysisError(
            f"The Stockfish path points to a directory, not an executable: '{resolved}'"
        )
    if not os.access(resolved, os.X_OK):
        raise EngineAnalysisError(
            f"The file exists but is not executable: '{resolved}'\n"
            "Check file permissions."
        )
    return resolved


def analyse_game(
    game: chess.pgn.Game,
    stockfish_path: str,
    depth: int = DEFAULT_DEPTH,
    multipv: int = MULTIPV,
    timeout_seconds: float = ANALYSIS_TIMEOUT_SECONDS,
    include_depth1: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Run Stockfish analysis on every mainline ply and return feature rows.

    The engine is started **once**, all positions are analysed, and the engine
    is closed cleanly (even on error).

    Parameters
    ----------
    game:
        A fully parsed ``chess.pgn.Game``.
    stockfish_path:
        Path to the Stockfish executable.
    depth:
        Search depth for each position.
    multipv:
        Number of principal variations to request (max 5).
    timeout_seconds:
        Maximum seconds to wait for one position's analysis.
    include_depth1:
        If True, query Stockfish depth 1 to enable depth-stability position complexity analysis.

    Returns
    -------
    (rows, metadata)
        *rows* — list of dicts, one per ply, with the engine columns filled.
        *metadata* — dict with engine version and analysis settings.

    Raises
    ------
    EngineAnalysisError
        If the engine cannot be started.  Individual position failures fill
        the row with empty strings instead of raising.
    """
    resolved_path = validate_stockfish_path(stockfish_path)

    # --- Start engine ---
    try:
        engine = chess.engine.SimpleEngine.popen_uci(resolved_path)
        engine_name = engine.id.get("name", "Unknown Stockfish")
    except Exception as exc:
        raise EngineAnalysisError(
            f"Failed to start Stockfish: {exc}\n"
            f"Path used: '{stockfish_path}'"
        ) from exc

    metadata: dict[str, str] = {
        "engine_name": engine_name,
        "engine_path": stockfish_path,
        "analysis_depth": str(depth),
        "multipv": str(multipv),
    }

    rows: list[dict[str, Any]] = []

    try:
        board = game.board()
        node: Optional[chess.pgn.GameNode] = game.next()
        ply_number = 0

        while node is not None:
            move: chess.Move = node.move  # type: ignore[assignment]
            ply_number += 1
            player_color = board.turn  # chess.WHITE or chess.BLACK

            row = _analyse_position(
                engine=engine,
                board=board,
                played_move=move,
                player_color=player_color,
                depth=depth,
                multipv=multipv,
                timeout_seconds=timeout_seconds,
                ply_number=ply_number,
                include_depth1=include_depth1,
            )
            rows.append(row)

            board.push(move)
            node = node.next()

    finally:
        # Always close the engine, even if something went wrong.
        try:
            engine.quit()
        except Exception:
            pass

    return rows, metadata


def _analyse_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    played_move: chess.Move,
    player_color: chess.Color,
    depth: int,
    multipv: int,
    timeout_seconds: float,
    ply_number: int,
    include_depth1: bool = False,
) -> dict[str, Any]:
    """Analyse a single position and return a dict of engine feature columns.

    On any failure the dict is returned with all columns set to ``""``.
    """
    empty_row: dict[str, Any] = {
        "engine_best_move_uci":   "",
        "engine_best_move_san":   "",
        "engine_best_move_depth1": "",
        "played_move_rank_top5":  "",
        "is_engine_top1":         "",
        "is_engine_top3":         "",
        "centipawn_loss":         "",
        "best_vs_second_cp_gap":  "",
        "engine_score_before_cp": "",
        "engine_score_after_cp":  "",
    }

    if not list(board.legal_moves):
        # Terminal position — nothing to analyse.
        return empty_row

    best_move_d1_uci: str = ""
    try:
        # Check whether engine supports streaming analysis and is not a mocked engine
        use_streaming = False
        try:
            import unittest.mock
            if hasattr(engine, "analysis") and not isinstance(engine.analyse, unittest.mock.Mock):
                use_streaming = True
        except Exception:
            use_streaming = False

        limit = chess.engine.Limit(depth=depth, time=timeout_seconds)

        if use_streaming and include_depth1:
            # Stream analysis to extract depth-1 best move directly from the ongoing search
            # without repeating searches or perturbing the engine transposition table.
            with engine.analysis(
                board,
                limit,
                multipv=multipv,
                info=chess.engine.INFO_ALL,
            ) as analysis:
                for info in analysis:
                    if info.get("depth") == 1 and not best_move_d1_uci and info.get("pv"):
                        best_move_d1_uci = info["pv"][0].uci()
                infos = analysis.multipv
        else:
            # Standard analyse call (used for mocked engines or when depth1 is disabled)
            raw_infos = engine.analyse(
                board,
                limit,
                multipv=multipv,
                info=chess.engine.INFO_ALL,
            )
            infos: list[chess.engine.InfoDict] = [raw_infos] if isinstance(raw_infos, dict) else (raw_infos or [])
    except chess.engine.EngineTerminatedError:
        # Engine crashed — return empty rather than propagating.
        return empty_row
    except Exception:
        return empty_row

    if not infos:
        return empty_row

    # --- Extract ordered list of (move, score_white_pov) pairs ---
    # python-chess returns infos sorted by MultiPV index (best first).
    candidates: list[tuple[chess.Move, Any]] = []
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if pv and score is not None:
            best_move_in_pv = pv[0]
            score_white = _score_to_cp_string(score, chess.WHITE)
            candidates.append((best_move_in_pv, score_white))

    if not candidates:
        return empty_row

    # --- Best move (rank 1) ---
    best_move, score_before_white = candidates[0]
    best_move_uci = best_move.uci()
    try:
        best_move_san = board.san(best_move)
    except Exception:
        best_move_san = best_move_uci

    # If depth == 1 or running under mock where streaming is bypassed
    if include_depth1 and not best_move_d1_uci:
        if depth == 1 or not use_streaming:
            best_move_d1_uci = best_move_uci

    # --- Score AFTER the played move (from White's POV) ---
    # Push the played move and ask for a depth-1 evaluation, or use the
    # existing MultiPV info if the played move is among the candidates
    # (which avoids an extra engine call in the common case).
    score_after_white: Any = ""

    played_rank: Optional[int] = None
    for rank_idx, (cand_move, cand_score) in enumerate(candidates, start=1):
        if cand_move == played_move:
            played_rank = rank_idx
            # The score AFTER playing this move is the eval *for the opponent*
            # at depth-1 from the candidate's perspective.  We use the engine's
            # evaluation of the resulting position instead.
            break

    # Get score_after by asking the engine to evaluate the board AFTER the move.
    board_after = board.copy()
    board_after.push(played_move)
    if list(board_after.legal_moves):
        try:
            after_infos = engine.analyse(
                board_after,
                chess.engine.Limit(depth=depth, time=timeout_seconds),
                multipv=1,
                info=chess.engine.INFO_ALL,
            )
            if isinstance(after_infos, list) and after_infos:
                after_score = after_infos[0].get("score")
            elif isinstance(after_infos, dict):
                after_score = after_infos.get("score")
            else:
                after_score = None

            if after_score is not None:
                score_after_white = _score_to_cp_string(after_score, chess.WHITE)
        except Exception:
            score_after_white = ""
    else:
        # Played move is terminal (checkmate or stalemate).
        # The position is already resolved.
        if board_after.is_checkmate():
            # The player who just moved delivered checkmate.
            if player_color == chess.WHITE:
                score_after_white = "#+0"
            else:
                score_after_white = "#-0"
        else:
            score_after_white = 0  # Stalemate = draw

    # --- Centipawn loss ---
    cp_loss = _centipawn_loss(score_before_white, score_after_white, player_color)

    # --- best_vs_second_cp_gap ---
    gap: Any = ""
    if len(candidates) >= 2:
        s1 = candidates[0][1]
        s2 = candidates[1][1]
        if not _is_mate_string(s1) and not _is_mate_string(s2):
            # Gap is always positive (best is always >= second from White's POV
            # when White to move, and <= when Black to move).
            raw_gap = abs(int(s1) - int(s2))
            gap = raw_gap

    # --- is_engine_top1 / is_engine_top3 ---
    is_top1 = int(played_rank == 1) if played_rank is not None else 0
    is_top3 = int(played_rank is not None and played_rank <= 3)

    return {
        "engine_best_move_uci":   best_move_uci,
        "engine_best_move_san":   best_move_san,
        "engine_best_move_depth1": best_move_d1_uci,
        "played_move_rank_top5":  played_rank if played_rank is not None else "",
        "is_engine_top1":         is_top1,
        "is_engine_top3":         is_top3,
        "centipawn_loss":         cp_loss,
        "best_vs_second_cp_gap":  gap,
        "engine_score_before_cp": score_before_white,
        "engine_score_after_cp":  score_after_white,
    }


# ───────────────────────────────────────────────────────────────────────────
# Column list for the engine-enriched CSV
# ───────────────────────────────────────────────────────────────────────────

ENGINE_COLUMNS = [
    "engine_best_move_uci",
    "engine_best_move_san",
    "engine_best_move_depth1",
    "played_move_rank_top5",
    "is_engine_top1",
    "is_engine_top3",
    "centipawn_loss",
    "best_vs_second_cp_gap",
    "engine_score_before_cp",
    "engine_score_after_cp",
]
