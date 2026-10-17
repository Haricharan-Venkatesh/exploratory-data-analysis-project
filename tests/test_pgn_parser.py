"""
tests/test_pgn_parser.py
========================
Comprehensive pytest test suite for Version 0 of the EDA chess project.

Tests cover:
  - Valid PGN parsing (move count, colors, numbers, SAN/UCI, FEN, CSV headers)
  - Missing optional metadata handling
  - Comments and variations (only mainline counted)
  - Empty file, malformed file, nonexistent file, directory path
  - Game with no mainline moves
  - CSV output columns and row count
  - game_phase values are valid strings
"""

from __future__ import annotations

import csv
import os
import tempfile
import stat

import pytest
import chess
import chess.pgn

# ---------------------------------------------------------------------------
# Helpers / path setup
# ---------------------------------------------------------------------------

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def fixture(name: str) -> str:
    """Return the absolute path to a test fixture file."""
    return os.path.join(FIXTURES, name)


# ---------------------------------------------------------------------------
# Import the modules under test
# ---------------------------------------------------------------------------

from chess_eda.pgn_parser import load_first_game, PGNParseError
from chess_eda.feature_extraction import extract_move_rows, CSV_COLUMNS


# ===========================================================================
# 1. Valid PGN parsing
# ===========================================================================

class TestValidPGN:
    """Tests against minimal_game.pgn (10 mainline plies, 5 fullmoves)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.game = load_first_game(fixture("minimal_game.pgn"))
        self.rows = extract_move_rows(self.game, game_id=1)

    def test_game_loads_without_error(self):
        """load_first_game returns a chess.pgn.Game object."""
        assert isinstance(self.game, chess.pgn.Game)

    def test_expected_move_count(self):
        """Exactly 10 mainline plies (5 fullmoves × 2 sides)."""
        assert len(self.rows) == 10

    def test_game_id_is_one(self):
        """game_id column is 1 for all rows in Version 0."""
        assert all(r["game_id"] == 1 for r in self.rows)

    # --- Player colors ---

    def test_first_move_is_white(self):
        """Ply 1 must be White."""
        assert self.rows[0]["player_color"] == "White"

    def test_second_move_is_black(self):
        """Ply 2 must be Black."""
        assert self.rows[1]["player_color"] == "Black"

    def test_colors_alternate(self):
        """Colors must strictly alternate White/Black throughout."""
        for i, row in enumerate(self.rows):
            expected = "White" if i % 2 == 0 else "Black"
            assert row["player_color"] == expected, (
                f"Ply {i+1}: expected {expected}, got {row['player_color']}"
            )

    # --- Fullmove numbers ---

    def test_fullmove_numbers(self):
        """
        fullmove_number equals the chess fullmove counter:
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5.
        """
        expected = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        actual = [r["fullmove_number"] for r in self.rows]
        assert actual == expected

    # --- Ply numbers ---

    def test_ply_numbers_sequential(self):
        """ply_number starts at 1 and increments by 1."""
        for i, row in enumerate(self.rows):
            assert row["ply_number"] == i + 1, (
                f"Row {i}: expected ply_number={i+1}, got {row['ply_number']}"
            )

    # --- SAN / UCI for first three plies ---

    def test_ply1_san(self):
        assert self.rows[0]["san"] == "e4"

    def test_ply1_uci(self):
        assert self.rows[0]["uci"] == "e2e4"

    def test_ply2_san(self):
        assert self.rows[1]["san"] == "e5"

    def test_ply2_uci(self):
        assert self.rows[1]["uci"] == "e7e5"

    def test_ply3_san(self):
        assert self.rows[2]["san"] == "Nf3"

    def test_ply3_uci(self):
        assert self.rows[2]["uci"] == "g1f3"

    # --- FEN before move ---

    def test_ply1_fen_before_is_starting_position(self):
        """Before White's first move the board is in the starting position."""
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert self.rows[0]["fen_before"] == starting_fen

    def test_ply2_fen_before(self):
        """Before Black's first move e4 has been played."""
        expected = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        assert self.rows[1]["fen_before"] == expected

    def test_ply3_fen_before(self):
        """Before White's Nf3, both e4 and e5 have been played."""
        expected = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        assert self.rows[2]["fen_before"] == expected

    def test_fen_before_is_not_after(self):
        """
        fen_before for ply N must not equal fen_before for ply N+1 when
        the move changes the position (always true for legal non-null moves).
        """
        for i in range(len(self.rows) - 1):
            assert self.rows[i]["fen_before"] != self.rows[i + 1]["fen_before"], (
                f"fen_before unchanged between ply {i+1} and ply {i+2}"
            )

    # --- Player metadata ---

    def test_white_player_name(self):
        white_rows = [r for r in self.rows if r["player_color"] == "White"]
        assert all(r["player_name"] == "Player One" for r in white_rows)

    def test_black_player_name(self):
        black_rows = [r for r in self.rows if r["player_color"] == "Black"]
        assert all(r["player_name"] == "Player Two" for r in black_rows)

    def test_white_elo_populated(self):
        white_rows = [r for r in self.rows if r["player_color"] == "White"]
        assert all(r["player_elo"] == "1500" for r in white_rows)

    def test_black_elo_populated(self):
        black_rows = [r for r in self.rows if r["player_color"] == "Black"]
        assert all(r["player_elo"] == "1500" for r in black_rows)

    # --- Game phase is a valid value ---

    def test_game_phase_valid_values(self):
        valid = {"Opening", "Middlegame", "Endgame"}
        for row in self.rows:
            assert row["game_phase"] in valid, (
                f"Invalid game_phase: {row['game_phase']!r}"
            )

    def test_opening_moves_labelled_opening(self):
        """With 14+ minor/major pieces on the board, phase must be Opening."""
        # In a 10-ply opening game all pieces are present → Opening
        assert all(r["game_phase"] == "Opening" for r in self.rows)

    # --- Shared headers ---

    def test_shared_headers_consistent(self):
        """Columns game_id, event, site, date, etc. are identical for all rows."""
        for col in ("game_id", "event", "site", "date", "round",
                    "white_player", "black_player", "white_elo",
                    "black_elo", "result"):
            values = {r[col] for r in self.rows}
            assert len(values) == 1, (
                f"Column '{col}' has inconsistent values across rows: {values}"
            )


# ===========================================================================
# 2. Missing optional metadata
# ===========================================================================

class TestMissingMetadata:
    """Tests against missing_metadata.pgn — only Event and Result headers."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.game = load_first_game(fixture("missing_metadata.pgn"))
        self.rows = extract_move_rows(self.game, game_id=1)

    def test_parses_without_error(self):
        assert isinstance(self.game, chess.pgn.Game)

    def test_move_count(self):
        """6 moves: d4 d5 c4 c6 Nf3 Nf6."""
        assert len(self.rows) == 6

    def test_missing_white_becomes_empty_string(self):
        white_rows = [r for r in self.rows if r["player_color"] == "White"]
        for row in white_rows:
            assert row["player_name"] == "", (
                f"Expected empty string for missing White, got {row['player_name']!r}"
            )

    def test_missing_black_becomes_empty_string(self):
        black_rows = [r for r in self.rows if r["player_color"] == "Black"]
        for row in black_rows:
            assert row["player_name"] == ""

    def test_missing_elo_becomes_empty_string(self):
        for row in self.rows:
            assert row["player_elo"] == ""

    def test_missing_site_becomes_empty_string(self):
        assert all(r["site"] == "" for r in self.rows)

    def test_missing_date_becomes_empty_string(self):
        assert all(r["date"] == "" for r in self.rows)


# ===========================================================================
# 3. Comments and variations
# ===========================================================================

class TestCommentsAndVariations:
    """Verifies that only mainline moves are extracted."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.game = load_first_game(fixture("comments_and_variations.pgn"))
        self.rows = extract_move_rows(self.game, game_id=1)

    def test_only_mainline_moves_counted(self):
        """
        The fixture has 8 mainline moves.  The King's Gambit variation and
        comments must not be included.
        """
        assert len(self.rows) == 8

    def test_first_mainline_move_is_e4(self):
        """The mainline starts with 1.e4 regardless of comments."""
        assert self.rows[0]["san"] == "e4"

    def test_second_mainline_move_is_e5(self):
        assert self.rows[1]["san"] == "e5"

    def test_third_mainline_move_is_nf3(self):
        """Move 2 mainline is Nf3, not the variation f4."""
        assert self.rows[2]["san"] == "Nf3"


# ===========================================================================
# 4. Error handling — empty / malformed / missing files
# ===========================================================================

class TestErrorHandling:

    def test_nonexistent_file_raises(self):
        """A path to a file that does not exist raises PGNParseError."""
        with pytest.raises(PGNParseError, match="not found"):
            load_first_game("tests/fixtures/does_not_exist.pgn")

    def test_directory_path_raises(self):
        """Passing a directory path raises PGNParseError."""
        with pytest.raises(PGNParseError, match="directory"):
            load_first_game("tests/fixtures")

    def test_empty_file_raises(self):
        """An empty file raises PGNParseError."""
        with pytest.raises(PGNParseError, match="empty"):
            load_first_game(fixture("empty.pgn"))

    def test_malformed_pgn_raises(self):
        """A file with no PGN headers raises PGNParseError."""
        with pytest.raises(PGNParseError):
            load_first_game(fixture("malformed.pgn"))

    def test_no_moves_raises(self):
        """A game with valid headers but no moves raises PGNParseError."""
        with pytest.raises(PGNParseError, match="no mainline moves"):
            load_first_game(fixture("no_moves.pgn"))


# ===========================================================================
# 5. CSV output
# ===========================================================================

class TestCSVOutput:
    """End-to-end test: write CSV, read it back, verify structure."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.game = load_first_game(fixture("minimal_game.pgn"))
        self.rows = extract_move_rows(self.game, game_id=1)
        self.csv_path = str(tmp_path / "test_output.csv")
        # Write CSV the same way main.py does
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def test_csv_file_exists(self):
        assert os.path.isfile(self.csv_path)

    def test_csv_has_correct_headers(self):
        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames) == CSV_COLUMNS

    def test_csv_row_count(self):
        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 10

    def test_csv_ply_numbers(self):
        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            plies = [int(r["ply_number"]) for r in reader]
        assert plies == list(range(1, 11))

    def test_csv_san_first_move(self):
        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            first = next(reader)
        assert first["san"] == "e4"

    def test_csv_uci_first_move(self):
        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            first = next(reader)
        assert first["uci"] == "e2e4"

    def test_csv_fen_before_first_move(self):
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            first = next(reader)
        assert first["fen_before"] == starting_fen

    def test_csv_contains_all_required_columns(self):
        """Every column in CSV_COLUMNS is present in the CSV file."""
        with open(self.csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert set(reader.fieldnames) == set(CSV_COLUMNS)


# ===========================================================================
# 6. Sample PGN smoke test
# ===========================================================================

class TestSamplePGN:
    """Quick smoke test against data/sample.pgn (The Immortal Game, 45 plies)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        pgn_path = os.path.join(PROJECT_ROOT, "data", "sample.pgn")
        self.game = load_first_game(pgn_path)
        self.rows = extract_move_rows(self.game, game_id=1)

    def test_immortal_game_move_count(self):
        """The Immortal Game has 45 mainline plies."""
        assert len(self.rows) == 45

    def test_white_is_anderssen(self):
        white_rows = [r for r in self.rows if r["player_color"] == "White"]
        assert all(r["player_name"] == "Adolf Anderssen" for r in white_rows)

    def test_black_is_kieseritzky(self):
        black_rows = [r for r in self.rows if r["player_color"] == "Black"]
        assert all(r["player_name"] == "Lionel Kieseritzky" for r in black_rows)

    def test_game_phases_present(self):
        """At least Opening and Middlegame phases appear in a 45-ply game."""
        phases = {r["game_phase"] for r in self.rows}
        assert "Opening" in phases
        assert "Middlegame" in phases

    def test_last_move_is_checkmate(self):
        """The last move of the Immortal Game is Be7# (checkmate)."""
        assert self.rows[-1]["san"] == "Be7#"

    def test_result_is_white_win(self):
        assert self.rows[0]["result"] == "1-0"
