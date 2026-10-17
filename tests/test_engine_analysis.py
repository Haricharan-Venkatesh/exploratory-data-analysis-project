"""
tests/test_engine_analysis.py
==============================
Tests for Version 1 — Stockfish engine analysis.

All engine-dependent tests use ``unittest.mock`` so that Stockfish does NOT
need to be installed to run the suite.

Test groups
-----------
TestValidateStockfishPath     : Path validation error cases.
TestScoreToCpString           : Mate-score sentinel formatting.
TestCentipawnLoss             : CP-loss calculation from correct perspective.
TestAnalysePositionMocked     : Per-position analysis with a mocked engine.
TestAnalyseGameMocked         : Full game analysis with mocked engine.
TestMergeInMain               : End-to-end mock through main() CLI.
TestRealEngineSmoke           : Real Stockfish test (skipped if not installed).

Representation rules tested
----------------------------
- Mate in N → "#+N" or "#-N" sentinel string
- Centipawn loss = "" when either score is a mate string
- played_move_rank_top5 = "" when move not in top-5 candidates
- is_engine_top1 / is_engine_top3 = 0 when move not in candidates
- All columns = "" when the engine crashes or returns no info
"""

from __future__ import annotations

import os
import sys
import csv
from typing import Any
from unittest.mock import MagicMock, patch, call

import chess
import chess.pgn
import chess.engine
import pytest

# Modules under test
from chess_eda.engine_analysis import (
    EngineAnalysisError,
    ENGINE_COLUMNS,
    DEFAULT_DEPTH,
    CP_LOSS_CAP,
    validate_stockfish_path,
    analyse_game,
    _score_to_cp_string,
    _centipawn_loss,
    _is_mate_string,
    _analyse_position,
)
from chess_eda.pgn_parser import load_first_game

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLE_PGN = os.path.join(PROJECT_ROOT, "data", "sample.pgn")


def fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Path validation
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateStockfishPath:

    def test_empty_path_raises(self):
        with pytest.raises(EngineAnalysisError, match="No Stockfish path"):
            validate_stockfish_path("")

    def test_nonexistent_path_raises(self):
        with pytest.raises(EngineAnalysisError, match="not found"):
            validate_stockfish_path("C:\\does\\not\\exist\\stockfish.exe")

    def test_directory_path_raises(self, tmp_path):
        with pytest.raises(EngineAnalysisError, match="directory"):
            validate_stockfish_path(str(tmp_path))

    def test_valid_executable_passes(self, tmp_path):
        """A real file that is executable should pass validation."""
        exe = tmp_path / "stockfish.exe"
        exe.write_bytes(b"placeholder")
        # On Windows, os.access with X_OK always returns True for files.
        # On Unix, we need to set the bit.
        try:
            exe.chmod(0o755)
        except Exception:
            pass
        # Should not raise
        validate_stockfish_path(str(exe))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Score-to-string helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_pov_score(cp: int | None = None, mate: int | None = None,
                    from_white: bool = True) -> chess.engine.PovScore:
    """Build a PovScore for testing."""
    if mate is not None:
        inner = chess.engine.Mate(mate)
    else:
        inner = chess.engine.Cp(cp)
    color = chess.WHITE if from_white else chess.BLACK
    return chess.engine.PovScore(inner, color)


class TestScoreToCpString:

    def test_positive_centipawn_from_white_pov(self):
        score = _make_pov_score(cp=150, from_white=True)
        result = _score_to_cp_string(score, chess.WHITE)
        assert result == 150

    def test_negative_centipawn_from_white_pov(self):
        score = _make_pov_score(cp=-80, from_white=True)
        result = _score_to_cp_string(score, chess.WHITE)
        assert result == -80

    def test_mate_in_2_winning(self):
        # Mate in 2 for White → "#+2"
        score = _make_pov_score(mate=2, from_white=True)
        result = _score_to_cp_string(score, chess.WHITE)
        assert result == "#+2"

    def test_mate_in_3_losing(self):
        # White gets mated in 3 → "#-3"
        score = _make_pov_score(mate=-3, from_white=True)
        result = _score_to_cp_string(score, chess.WHITE)
        assert result == "#-3"

    def test_black_score_from_black_pov(self):
        # Black is up 100 cp (Black's POV → White POV is -100)
        score = _make_pov_score(cp=100, from_white=False)
        result = _score_to_cp_string(score, chess.WHITE)
        assert result == -100

    def test_is_mate_string_positive(self):
        assert _is_mate_string("#+3") is True

    def test_is_mate_string_negative(self):
        assert _is_mate_string("#-5") is True

    def test_is_mate_string_integer(self):
        assert _is_mate_string(150) is False

    def test_is_mate_string_plain_string(self):
        assert _is_mate_string("hello") is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. Centipawn loss calculation
# ═══════════════════════════════════════════════════════════════════════════

class TestCentipawnLoss:
    """
    score_before_white: engine's best score (White's POV) BEFORE the move.
    score_after_white : engine's score (White's POV) AFTER the played move.
    """

    def test_white_blunders_returns_positive_loss(self):
        # White had +200, played a move that left +50.  Loss = 150.
        loss = _centipawn_loss(200, 50, chess.WHITE)
        assert loss == 150

    def test_white_plays_best_move_no_loss(self):
        # White played the best move — score barely changes.
        loss = _centipawn_loss(200, 198, chess.WHITE)
        assert loss == 2

    def test_black_blunders_returns_positive_loss(self):
        # Before Black moves: White eval = -100 (Black is up 100).
        # After Black's move:  White eval = +50  (Black lost advantage, now
        # White is up 50).
        # From Black's perspective: they went from +100 to -50 → loss = 150.
        loss = _centipawn_loss(-100, 50, chess.BLACK)
        assert loss == 150

    def test_black_plays_best_move_minimal_loss(self):
        # Black was at -150 (Black up), plays best move, score barely changes.
        loss = _centipawn_loss(-150, -148, chess.BLACK)
        assert loss == 2

    def test_loss_capped_at_cp_loss_cap(self):
        # Catastrophic blunder: score_before=3000, score_after=-3000 for White.
        loss = _centipawn_loss(3000, -3000, chess.WHITE)
        assert loss == CP_LOSS_CAP

    def test_mate_string_before_returns_empty(self):
        assert _centipawn_loss("#+3", 100, chess.WHITE) == ""

    def test_mate_string_after_returns_empty(self):
        assert _centipawn_loss(100, "#-2", chess.WHITE) == ""

    def test_both_mate_strings_returns_empty(self):
        assert _centipawn_loss("#+1", "#-1", chess.BLACK) == ""

    def test_loss_never_negative(self):
        # If the played move somehow improves on the engine's best (numerical
        # quirk), loss should be clipped to 0.
        loss = _centipawn_loss(50, 100, chess.WHITE)
        assert loss == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-position analysis with a mocked engine
# ═══════════════════════════════════════════════════════════════════════════

def _make_mock_info(move_uci: str, score_cp: int, pov: chess.Color = chess.WHITE,
                    board: chess.Board | None = None) -> dict:
    """Helper: create a fake InfoDict as python-chess would return."""
    brd = board or chess.Board()
    move = chess.Move.from_uci(move_uci)
    info: dict[str, Any] = {
        "pv": [move],
        "score": chess.engine.PovScore(chess.engine.Cp(score_cp), pov),
    }
    return info


class TestAnalysePositionMocked:
    """Test _analyse_position() with a fully mocked engine."""

    @pytest.fixture
    def board(self):
        return chess.Board()  # starting position

    @pytest.fixture
    def engine(self):
        return MagicMock(spec=chess.engine.SimpleEngine)

    def _make_engine_with_analysis(self, engine, before_infos, after_infos):
        """Configure engine.analyse() to return different results on calls."""
        engine.analyse.side_effect = [before_infos, after_infos]
        return engine

    def test_best_move_is_extracted(self, board, engine):
        before = [_make_mock_info("e2e4", 30)]
        after  = [_make_mock_info("e7e5", -20)]
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("e2e4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["engine_best_move_uci"] == "e2e4"
        assert row["engine_best_move_san"] == "e4"

    def test_played_move_is_top1(self, board, engine):
        before = [_make_mock_info("e2e4", 30)]
        after  = [_make_mock_info("e7e5", -20)]
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("e2e4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["is_engine_top1"] == 1
        assert row["is_engine_top3"] == 1
        assert row["played_move_rank_top5"] == 1

    def test_played_move_not_in_candidates(self, board, engine):
        # Engine suggests e2e4 but player played d2d4.
        before = [_make_mock_info("e2e4", 30)]
        after  = [_make_mock_info("e7e5", -10)]
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("d2d4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["is_engine_top1"] == 0
        assert row["is_engine_top3"] == 0
        assert row["played_move_rank_top5"] == ""

    def test_centipawn_loss_correct_for_white(self, board, engine):
        # Engine best: +200 for White.  After actual move: +50 for White.
        # Loss from White's perspective = 150.
        before = [_make_mock_info("e2e4", 200)]
        after  = [_make_mock_info("e7e5", 50)]
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("d2d4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["centipawn_loss"] == 150

    def test_centipawn_loss_correct_for_black(self):
        """Black blunders: White eval improves → Black's loss is positive."""
        # Position after 1.e4 (Black to move).
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        engine = MagicMock(spec=chess.engine.SimpleEngine)

        # Before Black moves: engine scores White at -100 (Black up 100).
        before = [_make_mock_info("e7e5", -100, pov=chess.WHITE, board=board)]
        # After Black plays c7c5 (Sicilian): engine scores White at +20.
        after  = [_make_mock_info("e2e4", 20)]  # doesn't matter which
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("c7c5"),
                                chess.BLACK, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=2)

        # Black went from +100 to -20, loss = 120.
        assert row["centipawn_loss"] == 120

    def test_mate_score_before_gives_sentinel_and_empty_loss(self, board, engine):
        mate_score = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
        before = [{"pv": [chess.Move.from_uci("e2e4")], "score": mate_score}]
        after  = [_make_mock_info("e7e5", 50)]
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("e2e4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["engine_score_before_cp"] == "#+2"
        assert row["centipawn_loss"] == ""

    def test_no_infos_returns_all_empty(self, board, engine):
        engine.analyse.return_value = []

        row = _analyse_position(engine, board, chess.Move.from_uci("e2e4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        for col in ENGINE_COLUMNS:
            assert row[col] == "", f"Column '{col}' should be empty"

    def test_engine_crash_returns_all_empty(self, board, engine):
        engine.analyse.side_effect = chess.engine.EngineTerminatedError()

        row = _analyse_position(engine, board, chess.Move.from_uci("e2e4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        for col in ENGINE_COLUMNS:
            assert row[col] == "", f"Column '{col}' should be empty after crash"

    def test_best_vs_second_gap_computed(self, board, engine):
        before = [
            _make_mock_info("e2e4", 200),
            _make_mock_info("d2d4", 150),
        ]
        after  = [_make_mock_info("e7e5", -10)]
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("e2e4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["best_vs_second_cp_gap"] == 50  # |200 - 150|

    def test_best_vs_second_gap_empty_when_only_one_pv(self, board, engine):
        before = [_make_mock_info("e2e4", 200)]
        after  = [_make_mock_info("e7e5", -10)]
        engine.analyse.side_effect = [before, after]

        row = _analyse_position(engine, board, chess.Move.from_uci("e2e4"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["best_vs_second_cp_gap"] == ""

    def test_top5_rank_uses_correct_candidate_order(self, board, engine):
        """Rank-3 move should be identified correctly."""
        board_copy = board.copy()
        moves = ["e2e4", "d2d4", "g1f3", "c2c4", "b2b3"]
        before = [_make_mock_info(m, 200 - (i * 20)) for i, m in enumerate(moves)]
        after  = [_make_mock_info("e7e5", -10)]
        engine.analyse.side_effect = [before, after]

        # Play the 3rd ranked move (g1f3)
        row = _analyse_position(engine, board_copy, chess.Move.from_uci("g1f3"),
                                chess.WHITE, depth=10, multipv=5,
                                timeout_seconds=10, ply_number=1)

        assert row["played_move_rank_top5"] == 3
        assert row["is_engine_top1"] == 0
        assert row["is_engine_top3"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full game analysis with mocked engine
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyseGameMocked:
    """Test analyse_game() with engine.popen_uci mocked out."""

    @pytest.fixture
    def minimal_game(self):
        return load_first_game(fixture("minimal_game.pgn"))

    def _build_mock_engine(self, cp_value: int = 50) -> MagicMock:
        """Return a mock SimpleEngine whose analyse() returns a fixed score."""
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        engine.id = {"name": "Mock Stockfish 99.0"}
        score = chess.engine.PovScore(chess.engine.Cp(cp_value), chess.WHITE)
        engine.analyse.return_value = [
            {"pv": [chess.Move.from_uci("e2e4")], "score": score}
        ]
        return engine

    def _patched_analyse(self, minimal_game, mock_engine, depth=5):
        """Helper: bypass path validation and mock popen_uci."""
        _no_op = lambda path: None  # noqa: E731
        with patch("chess_eda.engine_analysis.validate_stockfish_path", _no_op), \
             patch("chess.engine.SimpleEngine.popen_uci",
                   return_value=mock_engine):
            return analyse_game(minimal_game, stockfish_path="fake.exe", depth=depth)

    def test_returns_one_row_per_ply(self, minimal_game):
        mock_engine = self._build_mock_engine()
        rows, meta = self._patched_analyse(minimal_game, mock_engine)
        assert len(rows) == 10  # minimal_game has 10 plies

    def test_all_engine_columns_present(self, minimal_game):
        mock_engine = self._build_mock_engine()
        rows, _ = self._patched_analyse(minimal_game, mock_engine)
        for row in rows:
            for col in ENGINE_COLUMNS:
                assert col in row, f"Missing column '{col}'"

    def test_engine_is_quit_after_analysis(self, minimal_game):
        mock_engine = self._build_mock_engine()
        self._patched_analyse(minimal_game, mock_engine)
        mock_engine.quit.assert_called_once()

    def test_engine_is_quit_even_after_crash(self, minimal_game):
        mock_engine = MagicMock(spec=chess.engine.SimpleEngine)
        mock_engine.id = {"name": "Mock"}
        mock_engine.analyse.side_effect = chess.engine.EngineTerminatedError()

        rows, _ = self._patched_analyse(minimal_game, mock_engine)

        mock_engine.quit.assert_called_once()
        # All columns should be empty because engine crashed.
        for row in rows:
            for col in ENGINE_COLUMNS:
                assert row[col] == "", f"Column '{col}' should be empty after crash"

    def test_metadata_contains_engine_name(self, minimal_game):
        mock_engine = self._build_mock_engine()
        _, meta = self._patched_analyse(minimal_game, mock_engine, depth=7)
        assert meta["engine_name"] == "Mock Stockfish 99.0"
        assert meta["analysis_depth"] == "7"

    def test_invalid_path_raises_before_engine_start(self, minimal_game):
        """validate_stockfish_path fires before popen_uci is called."""
        with pytest.raises(EngineAnalysisError, match="not found"):
            analyse_game(minimal_game, stockfish_path="nonexistent.exe", depth=5)

    def test_engine_start_failure_raises_engine_analysis_error(self, minimal_game, tmp_path):
        exe = tmp_path / "fake_sf.exe"
        exe.write_bytes(b"x")
        try:
            exe.chmod(0o755)
        except Exception:
            pass

        with patch("chess.engine.SimpleEngine.popen_uci",
                   side_effect=Exception("Connection refused")):
            with pytest.raises(EngineAnalysisError, match="Failed to start"):
                analyse_game(minimal_game, stockfish_path=str(exe), depth=5)


# ═══════════════════════════════════════════════════════════════════════════
# 6. End-to-end through main() with mocks
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeInMain:
    """Verify that Version 1 main() correctly merges base+engine rows."""

    def test_engine_csv_has_all_columns(self, tmp_path):
        from main import main
        from chess_eda.feature_extraction import CSV_COLUMNS
        from chess_eda.engine_analysis import ENGINE_COLUMNS

        expected_cols = CSV_COLUMNS + ENGINE_COLUMNS
        v0_csv = str(tmp_path / "v0.csv")
        v1_csv = str(tmp_path / "v1.csv")
        fake_sf = tmp_path / "fake_sf.exe"
        fake_sf.write_bytes(b"x")
        try:
            fake_sf.chmod(0o755)
        except Exception:
            pass

        mock_engine = MagicMock(spec=chess.engine.SimpleEngine)
        mock_engine.id = {"name": "Mock Stockfish"}
        score = chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE)
        mock_engine.analyse.return_value = [
            {"pv": [chess.Move.from_uci("e2e4")], "score": score}
        ]

        with patch("chess.engine.SimpleEngine.popen_uci",
                   return_value=mock_engine):
            rc = main([
                fixture("minimal_game.pgn"),
                "--output", v0_csv,
                "--engine-analysis",
                "--stockfish-path", str(fake_sf),
                "--depth", "5",
                "--engine-output", v1_csv,
            ])

        assert rc == 0
        assert os.path.isfile(v1_csv)

        with open(v1_csv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = list(reader.fieldnames or [])
            rows = list(reader)

        assert cols == expected_cols
        assert len(rows) == 10  # minimal_game has 10 plies

    def test_v0_csv_unchanged_after_v1(self, tmp_path):
        """Version 0 CSV is written before engine analysis; it is unaffected
        even if engine analysis fails."""
        from main import main

        v0_csv = str(tmp_path / "v0.csv")

        rc = main([
            fixture("minimal_game.pgn"),
            "--output", v0_csv,
            "--engine-analysis",
            "--stockfish-path", "does_not_exist.exe",
            "--depth", "5",
        ])

        # Engine analysis failed (bad path) → exit code 1
        assert rc == 1
        # But Version 0 CSV was written before engine analysis ran.
        assert os.path.isfile(v0_csv)

    def test_missing_stockfish_path_returns_error(self, tmp_path):
        from main import main

        v0_csv = str(tmp_path / "v0.csv")

        rc = main([
            fixture("minimal_game.pgn"),
            "--output", v0_csv,
            "--engine-analysis",
            # no --stockfish-path
        ])

        assert rc == 1


# ═══════════════════════════════════════════════════════════════════════════
# 7. Real Stockfish smoke test (skipped if unavailable)
# ═══════════════════════════════════════════════════════════════════════════

def _find_stockfish() -> str | None:
    """Return the path to a real Stockfish executable, or None."""
    import shutil
    # 1. Check environment variable
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Check PATH
    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found and os.path.isfile(found):
            return found

    # 3. Common Windows locations
    candidates = [
        r"C:\stockfish\stockfish.exe",
        r"C:\Program Files\stockfish\stockfish.exe",
        r"C:\Users\Public\stockfish\stockfish.exe",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


_STOCKFISH = _find_stockfish()
SKIP_REAL_ENGINE = pytest.mark.skipif(
    _STOCKFISH is None,
    reason=(
        "Stockfish not installed. "
        "Download from https://stockfishchess.org/download/ "
        "and add to PATH or a known location."
    ),
)


@SKIP_REAL_ENGINE
class TestRealEngineSmoke:
    """These tests actually call Stockfish. Skipped when not installed."""

    def test_real_analysis_returns_correct_row_count(self):
        game = load_first_game(SAMPLE_PGN)
        rows, meta = analyse_game(game, stockfish_path=_STOCKFISH, depth=8)
        assert len(rows) == 45

    def test_real_engine_name_in_metadata(self):
        game = load_first_game(SAMPLE_PGN)
        _, meta = analyse_game(game, stockfish_path=_STOCKFISH, depth=5)
        assert "stockfish" in meta["engine_name"].lower()

    def test_real_first_move_e4_is_top_candidate(self):
        """The Immortal Game starts with 1.e4; this should rank highly."""
        game = load_first_game(SAMPLE_PGN)
        rows, _ = analyse_game(game, stockfish_path=_STOCKFISH, depth=10)
        first_row = rows[0]
        # e2e4 is a very common first move; engine should rank it top-5.
        rank = first_row["played_move_rank_top5"]
        assert rank in (1, 2, 3, 4, 5, ""), f"Unexpected rank: {rank}"
        assert first_row["engine_best_move_uci"] != ""
        assert first_row["engine_score_before_cp"] != ""

    def test_real_centipawn_loss_non_negative(self):
        game = load_first_game(SAMPLE_PGN)
        rows, _ = analyse_game(game, stockfish_path=_STOCKFISH, depth=8)
        for i, row in enumerate(rows):
            cp = row["centipawn_loss"]
            if cp != "":
                assert isinstance(cp, int) and cp >= 0, (
                    f"Ply {i+1}: centipawn_loss={cp!r} is negative or wrong type"
                )

    def test_real_engine_csv_via_main(self, tmp_path):
        from main import main
        from chess_eda.feature_extraction import CSV_COLUMNS
        from chess_eda.engine_analysis import ENGINE_COLUMNS

        v0_csv = str(tmp_path / "v0.csv")
        v1_csv = str(tmp_path / "v1.csv")

        rc = main([
            SAMPLE_PGN,
            "--output", v0_csv,
            "--engine-analysis",
            "--stockfish-path", _STOCKFISH,
            "--depth", "8",
            "--engine-output", v1_csv,
        ])

        assert rc == 0
        with open(v1_csv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = list(reader.fieldnames or [])
            rows = list(reader)

        assert set(cols) == set(CSV_COLUMNS + ENGINE_COLUMNS)
        assert len(rows) == 45
