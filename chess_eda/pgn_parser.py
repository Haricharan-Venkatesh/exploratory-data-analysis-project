"""
chess_eda.pgn_parser
====================
Responsible for loading and validating a PGN file, then returning the
first game found in it together with its mainline move sequence.

Version 0 limitation
--------------------
Only the **first game** in a multi-game PGN is processed.  This is an
intentional scope restriction for Version 0; subsequent games are ignored.

Error policy
------------
All errors are raised as ``PGNParseError`` (a subclass of ``ValueError``).
The caller (main.py) is responsible for printing the message and exiting
with a non-zero status code.

Invalid / illegal moves are treated as fatal: the game is rejected in full
rather than silently exporting a partial game.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import chess
import chess.pgn


class PGNParseError(ValueError):
    """Raised for any PGN-level or file-level error during parsing."""


def _check_file(pgn_path: str) -> None:
    """Validate that *pgn_path* is a readable, non-empty regular file.

    Raises
    ------
    PGNParseError
        With a human-readable message describing the specific problem.
    """
    if not os.path.exists(pgn_path):
        raise PGNParseError(
            f"File not found: '{pgn_path}'\n"
            "Please check the path and try again."
        )
    if os.path.isdir(pgn_path):
        raise PGNParseError(
            f"The path points to a directory, not a file: '{pgn_path}'"
        )
    if not os.path.isfile(pgn_path):
        raise PGNParseError(
            f"The path does not point to a regular file: '{pgn_path}'"
        )
    if os.path.getsize(pgn_path) == 0:
        raise PGNParseError(
            f"The file is empty: '{pgn_path}'"
        )
    # Attempt to open for reading
    try:
        with open(pgn_path, "r", encoding="utf-8") as fh:
            fh.read(1)
    except PermissionError as exc:
        raise PGNParseError(
            f"Permission denied — cannot read file: '{pgn_path}'\n"
            f"OS error: {exc}"
        ) from exc
    except OSError as exc:
        raise PGNParseError(
            f"Cannot read file: '{pgn_path}'\nOS error: {exc}"
        ) from exc


class _ErrorCapturingVisitor(chess.pgn.GameBuilder):
    """A GameBuilder that collects parser errors instead of raising them.

    ``chess.pgn.GameBuilder`` already implements the required abstract
    ``result()`` method, so we only need to override ``handle_error``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []

    def handle_error(self, error: Exception) -> None:  # type: ignore[override]
        self.errors.append(str(error))


def load_first_game(pgn_path: str) -> chess.pgn.Game:
    """Parse *pgn_path* and return the first ``chess.pgn.Game`` found.

    Parameters
    ----------
    pgn_path:
        Absolute or relative path to the PGN file.

    Returns
    -------
    chess.pgn.Game
        The parsed game object.  The mainline move tree is fully populated.

    Raises
    ------
    PGNParseError
        If the file is missing, unreadable, empty, contains no game, or
        contains invalid / illegal move notation.
    """
    _check_file(pgn_path)

    try:
        with open(pgn_path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError as exc:
        raise PGNParseError(
            f"Failed to read file: '{pgn_path}'\nOS error: {exc}"
        ) from exc

    # --- First pass: normal parse ---
    buf = io.StringIO(content)
    try:
        game: Optional[chess.pgn.Game] = chess.pgn.read_game(buf)
    except Exception as exc:  # pragma: no cover — defensive
        raise PGNParseError(
            f"The PGN parser raised an unexpected error: {exc}"
        ) from exc

    if game is None:
        raise PGNParseError(
            f"No parseable game found in: '{pgn_path}'\n"
            "The file must contain at least one PGN game header block."
        )

    # --- Second pass: capture parser errors via custom visitor ---
    buf.seek(0)
    visitor = _ErrorCapturingVisitor()
    try:
        chess.pgn.read_game(buf, Visitor=lambda: visitor)
    except Exception:
        pass  # Errors already captured; fall through to check list.

    if visitor.errors:
        raise PGNParseError(
            "The PGN parser reported move errors — game rejected to avoid "
            "exporting a partial or invalid game.\n"
            "Parser errors:\n" + "\n".join(f"  • {e}" for e in visitor.errors)
        )

    # --- Verify the mainline contains at least one move and all moves are legal ---
    validate_game(game)

    return game


def validate_game(game: chess.pgn.Game) -> None:
    """Verify that *game* contains at least one move, has no parse errors, and every move is legal.

    Raises
    ------
    PGNParseError
        If there are syntax errors, no moves, or an illegal move is detected.
    """
    if getattr(game, "errors", None):
        raise PGNParseError(
            "The PGN parser reported move errors — game rejected:\n"
            + "\n".join(f"  • {e}" for e in game.errors)
        )

    if game.next() is None:
        raise PGNParseError(
            "The game was parsed successfully but contains no mainline moves.\n"
            "A game must have at least one move to be exported."
        )

    board = game.board()
    node: Optional[chess.pgn.GameNode] = game.next()
    ply = 0
    while node is not None:
        move: chess.Move = node.move  # type: ignore[assignment]
        if move is None:
            break  # pragma: no cover
        ply += 1
        if move not in board.legal_moves:
            try:
                san_attempt = board.san(move)
            except Exception:
                san_attempt = str(move)
            raise PGNParseError(
                f"Illegal move detected at ply {ply} "
                f"(board position: {board.fen()!r}).\n"
                f"Move: {san_attempt}\n"
                "The game is rejected in full."
            )
        board.push(move)
        node = node.next()


def load_all_games(
    pgn_path: str,
    skip_invalid: bool = True,
) -> tuple[list[chess.pgn.Game], list[dict[str, Any]]]:
    """Parse *pgn_path* and return all valid ``chess.pgn.Game`` objects.

    Parameters
    ----------
    pgn_path:
        Path to the PGN file.
    skip_invalid:
        If True, malformed or invalid games are skipped and recorded in the
        returned errors list instead of aborting the entire file.

    Returns
    -------
    (games, skipped_info)
        *games* — list of validated ``chess.pgn.Game`` objects.
        *skipped_info* — list of dicts describing skipped games and errors.
    """
    _check_file(pgn_path)

    games: list[chess.pgn.Game] = []
    skipped: list[dict[str, Any]] = []

    try:
        with open(pgn_path, "r", encoding="utf-8", errors="replace") as fh:
            game_idx = 0
            while True:
                game_idx += 1
                try:
                    game = chess.pgn.read_game(fh)
                except Exception as exc:
                    if skip_invalid:
                        skipped.append({
                            "game_index": game_idx,
                            "error": f"PGN parser read error: {exc}",
                        })
                        break
                    raise PGNParseError(f"Error reading game #{game_idx}: {exc}") from exc

                if game is None:
                    break

                try:
                    validate_game(game)
                    games.append(game)
                except PGNParseError as exc:
                    if skip_invalid:
                        skipped.append({
                            "game_index": game_idx,
                            "error": str(exc),
                            "white": game.headers.get("White", ""),
                            "black": game.headers.get("Black", ""),
                        })
                    else:
                        raise

    except OSError as exc:
        raise PGNParseError(f"Cannot read '{pgn_path}': {exc}") from exc

    if not games and not skipped:
        raise PGNParseError(f"No parseable game found in: '{pgn_path}'")

    return games, skipped

