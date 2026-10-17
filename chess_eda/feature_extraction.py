"""
chess_eda.feature_extraction
=============================
Extracts one row of features per mainline move from a parsed
``chess.pgn.Game`` object.

Game-phase heuristic
--------------------
The game phase is estimated from the material count **before** each move.
This is a simple heuristic — it is **not** a definitive chess-theory
classification.

  Phase       Condition (piece count excludes pawns and kings)
  ----------- -------------------------------------------------------
  Opening     More than 12 pieces on the board (both sides combined,
              not counting pawns or kings); OR the move number is
              <= 10 and both queens are still present.
  Endgame     Fewer than 6 pieces (excluding pawns and kings).
  Middlegame  Everything in between.

The thresholds were chosen to produce intuitively sensible labels for
typical club-level games and do not correspond to any single authoritative
definition.  Future versions may replace this heuristic with a
table-based or neural-net approach.

Column definitions
------------------
game_id        : Sequential integer starting at 1 (always 1 in Version 0).
event          : PGN Event tag, or "" if missing.
site           : PGN Site tag, or "" if missing.
date           : PGN Date tag, or "" if missing.
round          : PGN Round tag, or "" if missing.
white_player   : PGN White tag, or "" if missing.
black_player   : PGN Black tag, or "" if missing.
white_elo      : PGN WhiteElo tag, or "" if missing.
black_elo      : PGN BlackElo tag, or "" if missing.
result         : PGN Result tag, or "" if missing.
fullmove_number: Chess fullmove number (1 for both White's and Black's first
                 move of the opening).
ply_number     : Sequential half-move index starting at 1 (White's first
                 move = 1, Black's first = 2, etc.).
player_color   : "White" or "Black".
player_name    : Name of the player whose turn it is, or "" if unavailable.
player_elo     : ELO of the player whose turn it is, or "" if unavailable.
san            : Standard Algebraic Notation for the move (e.g. "Nf3").
uci            : UCI notation for the move (e.g. "g1f3").
fen_before     : FEN string representing the position *before* the move.
game_phase     : "Opening", "Middlegame", or "Endgame" (heuristic).
"""

from __future__ import annotations

from typing import Any

import chess
import chess.pgn


# ---------------------------------------------------------------------------
# Piece-count constants used by the phase heuristic
# ---------------------------------------------------------------------------
_MINOR_MAJOR_PIECES = {
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
}

_OPENING_PIECE_THRESHOLD = 12   # > 12 minor/major pieces → Opening
_ENDGAME_PIECE_THRESHOLD = 6    # < 6  minor/major pieces → Endgame


def _count_pieces(board: chess.Board) -> int:
    """Return the total number of non-pawn, non-king pieces on *board*."""
    count = 0
    for piece_type in _MINOR_MAJOR_PIECES:
        count += len(board.pieces(piece_type, chess.WHITE))
        count += len(board.pieces(piece_type, chess.BLACK))
    return count


def _game_phase(board: chess.Board) -> str:
    """Estimate the game phase for the position encoded in *board*.

    Returns one of ``"Opening"``, ``"Middlegame"``, or ``"Endgame"``.

    This is a heuristic — see module docstring for details.
    """
    piece_count = _count_pieces(board)
    if piece_count > _OPENING_PIECE_THRESHOLD:
        return "Opening"
    if piece_count < _ENDGAME_PIECE_THRESHOLD:
        return "Endgame"
    return "Middlegame"


def _header(game: chess.pgn.Game, tag: str) -> str:
    """Return the value of a PGN header *tag*, or ``""`` if absent or unknown.

    python-chess automatically inserts placeholder values for missing headers:
    - ``"?"``       — for most missing tags (White, Black, Site, Round, …)
    - ``"????.??.??"`` — for missing Date

    This function maps all placeholder values to an empty string so that CSV
    consumers can treat missing optional metadata uniformly.
    """
    value = game.headers.get(tag, "")
    # Strip known python-chess placeholder values.
    if not value:
        return ""
    # A value consisting only of "?" characters and date-separator "." or "-"
    # is a placeholder (e.g. "?", "??", "????.??.??").
    stripped = value.replace(".", "").replace("-", "")
    if stripped == "?" * len(stripped) and stripped:
        return ""
    if value in ("-",):
        return ""
    return value


def extract_move_rows(
    game: chess.pgn.Game,
    game_id: int = 1,
) -> list[dict[str, Any]]:
    """Extract one feature row per mainline move.

    Parameters
    ----------
    game:
        A fully parsed ``chess.pgn.Game`` (from ``pgn_parser.load_first_game``).
    game_id:
        Numeric identifier for this game (always 1 in Version 0).

    Returns
    -------
    list[dict[str, Any]]
        Each dict corresponds to one mainline half-move (ply) and contains
        the columns described in this module's docstring.
    """
    # --- Gather metadata from headers (missing fields become "") ---
    event       = _header(game, "Event")
    site        = _header(game, "Site")
    date        = _header(game, "Date")
    round_      = _header(game, "Round")
    white_name  = _header(game, "White")
    black_name  = _header(game, "Black")
    white_elo   = _header(game, "WhiteElo")
    black_elo   = _header(game, "BlackElo")
    result      = _header(game, "Result")

    rows: list[dict[str, Any]] = []
    board = game.board()   # starts at the initial position for the game
    node: chess.pgn.GameNode | None = game.next()
    ply_number = 0

    while node is not None:
        move: chess.Move = node.move  # type: ignore[assignment]

        # --- Snapshot state BEFORE the move is applied ---
        fen_before     = board.fen()
        turn           = board.turn            # chess.WHITE or chess.BLACK
        fullmove_num   = board.fullmove_number  # standard chess move counter

        # --- Compute SAN and UCI from the current board position ---
        san = board.san(move)
        uci = move.uci()

        # --- Phase is assessed on the position before the move ---
        phase = _game_phase(board)

        # --- Colour-specific fields ---
        if turn == chess.WHITE:
            player_color = "White"
            player_name  = white_name
            player_elo   = white_elo
        else:
            player_color = "Black"
            player_name  = black_name
            player_elo   = black_elo

        ply_number += 1

        rows.append({
            "game_id":          game_id,
            "event":            event,
            "site":             site,
            "date":             date,
            "round":            round_,
            "white_player":     white_name,
            "black_player":     black_name,
            "white_elo":        white_elo,
            "black_elo":        black_elo,
            "result":           result,
            "fullmove_number":  fullmove_num,
            "ply_number":       ply_number,
            "player_color":     player_color,
            "player_name":      player_name,
            "player_elo":       player_elo,
            "san":              san,
            "uci":              uci,
            "fen_before":       fen_before,
            "game_phase":       phase,
        })

        # Apply the move to advance the board state.
        board.push(move)
        node = node.next()

    return rows


# Column order guaranteed for CSV export.
CSV_COLUMNS = [
    "game_id",
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
    "san",
    "uci",
    "fen_before",
    "game_phase",
]
