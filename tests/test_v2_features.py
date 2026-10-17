"""
tests/test_v2_features.py
=========================
Comprehensive tests for Version 2:
- Multi-game PGN loading and error handling.
- Board-derived move features (legal_moves_count, is_capture, is_check, is_promotion).
- Move quality tiers and mate score handling.
- Player-game level feature aggregation and denominator accuracy.
- Batch pipeline execution and dataset manifest generation.
"""

from __future__ import annotations

import csv
import io
import json
import os
from unittest.mock import MagicMock, patch

import chess
import chess.pgn
import pytest

from chess_eda.aggregation import (
    aggregate_player_game_features,
    PLAYER_GAME_COLUMNS,
    _compute_stats,
    _rate,
)
from chess_eda.move_features import (
    extract_move_features,
    enrich_moves_with_engine,
    _classify_move_quality,
    MOVE_FEATURE_COLUMNS,
)
from chess_eda.pgn_parser import (
    load_all_games,
    load_first_game,
    validate_game,
    PGNParseError,
)
from chess_eda.pipeline import run_batch_pipeline

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Multi-game PGN Loading
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiGameParsing:

    def test_load_all_games_multiple(self):
        games, skipped = load_all_games(fixture("multi_game.pgn"))
        assert len(games) == 2
        assert len(skipped) == 0
        assert games[0].headers.get("White") == "PlayerOne"
        assert games[1].headers.get("White") == "PlayerThree"

    def test_load_all_games_move_counts(self):
        games, _ = load_all_games(fixture("multi_game.pgn"))
        moves_g1 = list(games[0].mainline_moves())
        moves_g2 = list(games[1].mainline_moves())
        assert len(moves_g1) == 7  # Scholar's mate
        assert len(moves_g2) == 4  # Fool's mate

    def test_load_first_game_regression_on_multi(self):
        """load_first_game must only load game #1 (Version 0 preservation)."""
        first = load_first_game(fixture("multi_game.pgn"))
        assert first.headers.get("White") == "PlayerOne"

    def test_skip_invalid_games_gracefully(self, tmp_path):
        bad_pgn = tmp_path / "mixed.pgn"
        bad_pgn.write_text(
            '[Event "Good Game 1"]\n\n1. e4 e5 2. Nf3 1-0\n\n'
            '[Event "Bad Game: No moves"]\n\n*\n\n'
            '[Event "Good Game 2"]\n\n1. d4 d5 2. c4 1-0\n\n',
            encoding="utf-8"
        )
        games, skipped = load_all_games(str(bad_pgn), skip_invalid=True)
        assert len(games) == 2
        assert len(skipped) == 1
        assert "no mainline moves" in skipped[0]["error"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Board-derived Move Features
# ═══════════════════════════════════════════════════════════════════════════

class TestBoardDerivedMoveFeatures:

    @pytest.fixture
    def scholar_game(self):
        games, _ = load_all_games(fixture("multi_game.pgn"))
        return games[0]

    def test_feature_columns_complete(self, scholar_game):
        rows = extract_move_features(scholar_game, game_id=1, source_file="test.pgn", source_category="human_games")
        assert len(rows) == 7
        for r in rows:
            assert r["game_id"] == 1
            assert r["source_file"] == "test.pgn"
            assert r["source_category"] == "human_games"
            assert "legal_moves_count" in r
            assert "is_capture" in r
            assert "is_check" in r
            assert "is_promotion" in r

    def test_legal_moves_count_starting_position(self, scholar_game):
        rows = extract_move_features(scholar_game)
        # Starting position has 20 legal moves (16 pawn pushes + 4 knight hops)
        assert rows[0]["legal_moves_count"] == 20
        # After 1. e4, Black also has 20 legal moves
        assert rows[1]["legal_moves_count"] == 20

    def test_capture_and_check_detection(self, scholar_game):
        rows = extract_move_features(scholar_game)
        # 1. e4 (no capture, no check)
        assert rows[0]["is_capture"] == 0
        assert rows[0]["is_check"] == 0

        # Last move: 4. Qxf7# (Queen captures pawn on f7 and delivers check/mate)
        last_move = rows[-1]
        assert last_move["san"] == "Qxf7#"
        assert last_move["is_capture"] == 1
        assert last_move["is_check"] == 1

    def test_promotion_detection(self):
        pgn = io.StringIO(
            '[Event "Promotion Test"]\n\n'
            '1. e4 d5 2. exd5 c6 3. dxc6 Nf6 4. cxb7 e5 5. bxa8=Q 1-0\n'
        )
        game = chess.pgn.read_game(pgn)
        rows = extract_move_features(game)
        promo_row = [r for r in rows if "=Q" in r["san"]][0]
        assert promo_row["is_promotion"] == 1
        assert rows[0]["is_promotion"] == 0

    def test_fen_before_is_strictly_before_move(self, scholar_game):
        rows = extract_move_features(scholar_game)
        # Move 1: e4. fen_before must be standard starting FEN
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert rows[0]["fen_before"] == starting_fen


# ═══════════════════════════════════════════════════════════════════════════
# 3. Move Quality Classification
# ═══════════════════════════════════════════════════════════════════════════

class TestMoveQualityClassification:

    def test_best_move(self):
        assert _classify_move_quality(0, 1, 0, False) == "Best"
        assert _classify_move_quality(5, 1, 0, False) == "Best"

    def test_tiers(self):
        assert _classify_move_quality(15, 0, 0, False) == "Excellent"
        assert _classify_move_quality(35, 0, 0, False) == "Good"
        assert _classify_move_quality(75, 0, 0, False) == "Inaccuracy"
        assert _classify_move_quality(150, 0, 0, False) == "Mistake"
        assert _classify_move_quality(350, 0, 0, False) == "Blunder"

    def test_mate_sequence(self):
        assert _classify_move_quality("", 0, 1, False) == "Forced Mate"

    def test_missing_evaluation(self):
        assert _classify_move_quality("", 0, 0, True) == ""


# ═══════════════════════════════════════════════════════════════════════════
# 4. Player-Game Feature Aggregation
# ═══════════════════════════════════════════════════════════════════════════

class TestPlayerGameAggregation:

    def test_compute_stats_helper(self):
        mean_v, med_v, std_v, min_v, max_v = _compute_stats([10, 20, 30])
        assert mean_v == 20.0
        assert med_v == 20.0
        assert std_v == 10.0
        assert min_v == 10
        assert max_v == 30

    def test_compute_stats_empty(self):
        assert _compute_stats([]) == ("", "", "", "", "")

    def test_compute_stats_single(self):
        mean_v, med_v, std_v, min_v, max_v = _compute_stats([25])
        assert mean_v == 25.0
        assert std_v == 0.0

    def test_rate_calculation(self):
        assert _rate(3, 10) == 0.3
        assert _rate(0, 0) == ""

    def test_aggregate_produces_two_rows_per_game(self):
        # 4 plies: White, Black, White, Black
        moves = [
            {"game_id": 1, "player_color": "White", "centipawn_loss": 0, "is_engine_top1": 1, "is_engine_top3": 1, "is_engine_eval_missing": 0, "game_phase": "Opening", "is_capture": 0, "is_check": 0, "is_promotion": 0, "result": "1-0"},
            {"game_id": 1, "player_color": "Black", "centipawn_loss": 30, "is_engine_top1": 0, "is_engine_top3": 1, "is_engine_eval_missing": 0, "game_phase": "Opening", "is_capture": 0, "is_check": 0, "is_promotion": 0, "result": "1-0"},
            {"game_id": 1, "player_color": "White", "centipawn_loss": 10, "is_engine_top1": 1, "is_engine_top3": 1, "is_engine_eval_missing": 0, "game_phase": "Opening", "is_capture": 0, "is_check": 0, "is_promotion": 0, "result": "1-0"},
            {"game_id": 1, "player_color": "Black", "centipawn_loss": 250, "is_engine_top1": 0, "is_engine_top3": 0, "is_engine_eval_missing": 0, "game_phase": "Opening", "is_capture": 1, "is_check": 1, "is_promotion": 0, "result": "1-0"},
        ]
        agg = aggregate_player_game_features(moves)
        assert len(agg) == 2
        w = [r for r in agg if r["player_color"] == "White"][0]
        b = [r for r in agg if r["player_color"] == "Black"][0]

        # White checks
        assert w["total_moves"] == 2
        assert w["scored_moves"] == 2
        assert w["mean_centipawn_loss"] == 5.0  # (0 + 10) / 2
        assert w["top1_match_rate"] == 1.0
        assert w["player_result"] == "win"

        # Black checks
        assert b["total_moves"] == 2
        assert b["scored_moves"] == 2
        assert b["mean_centipawn_loss"] == 140.0  # (30 + 250) / 2
        assert b["blunder_count"] == 1
        assert b["blunder_rate"] == 0.5
        assert b["player_result"] == "loss"
        assert b["captures_count"] == 1
        assert b["checks_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Batch Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchPipelineExecution:

    def test_run_batch_pipeline_no_engine(self, tmp_path):
        raw_root = tmp_path / "raw"
        processed_root = tmp_path / "processed"
        (raw_root / "human_games").mkdir(parents=True)
        (raw_root / "engine_generated_games").mkdir(parents=True)
        (raw_root / "unlabeled").mkdir(parents=True)

        # Place fixture in human_games
        with open(fixture("multi_game.pgn"), "r", encoding="utf-8") as src:
            content = src.read()
        (raw_root / "human_games" / "sample.pgn").write_text(content, encoding="utf-8")

        manifest = run_batch_pipeline(
            raw_dir=str(raw_root),
            output_dir=str(processed_root),
            enable_engine=False,
            verbose=False,
        )

        assert manifest["summary"]["total_files_scanned"] == 1
        assert manifest["summary"]["total_games_processed"] == 2
        assert manifest["summary"]["total_moves_extracted"] == 11  # 7 + 4
        assert manifest["summary"]["total_player_games"] == 4  # 2 players * 2 games

        move_csv = processed_root / "move_level_features.csv"
        player_csv = processed_root / "player_game_features.csv"
        manifest_json = processed_root / "dataset_manifest.json"

        assert move_csv.is_file()
        assert player_csv.is_file()
        assert manifest_json.is_file()

        with open(move_csv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames or []) == MOVE_FEATURE_COLUMNS
            assert len(list(reader)) == 11

        with open(player_csv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert list(reader.fieldnames or []) == PLAYER_GAME_COLUMNS
            assert len(list(reader)) == 4

        # Provenance audit fields
        assert "provenance_audit_summary" in manifest
        assert "provenance_audit" in manifest["files_processed"][0]
        assert manifest["provenance_audit_summary"]["research_readiness"] != ""


def test_provenance_audit_classification():
    from chess_eda.pipeline import _audit_file_provenance
    
    immortal = _audit_file_provenance("human_games/immortal_game.pgn")
    assert immortal["provenance_status"] == "verified_historical_human"
    assert "Independently verified" in immortal["authenticity"]

    opera = _audit_file_provenance("human_games/opera_game_1858.pgn")
    assert opera["provenance_status"] == "verified_historical_human"
    assert "Independently verified" in opera["authenticity"]

    tcec = _audit_file_provenance("engine_generated_games/tcec_engine_game.pgn")
    assert tcec["provenance_status"] == "illustrative_engine_demonstration_unverified"
    assert "Unverified" in tcec["authenticity"]

    casual = _audit_file_provenance("unlabeled/casual_game.pgn")
    assert casual["provenance_status"] == "synthetic_programmatic_demonstration"
    assert "Programmatically constructed" in casual["authenticity"]
