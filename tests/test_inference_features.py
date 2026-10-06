"""
Test Suite: Feature Calculation & Temporal Leakage Integrity
============================================================
Tests:
- PlayerMoveTracker reproduces exact training feature formulas
- Zero future leakage: move k features are completely invariant to future moves
- Rolling window calculations (W = 10) handle edge window boundaries correctly
- Endgame clock ratio and phase deliberation behave according to definition
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference_features import PlayerMoveTracker, FEATURE_COLUMNS, determine_game_phase
import chess

class TestInferenceFeatures(unittest.TestCase):
    def test_tracker_initial_move(self):
        tracker = PlayerMoveTracker("White", base_time=180, inc_time=0)
        feats = tracker.record_move(move_time=2.5, clock_after_move=177.5, phase="Opening")
        
        self.assertEqual(len(feats), 10)
        self.assertEqual(feats["consecutive_fast_moves"], 0.0)
        self.assertEqual(feats["premove_rate_10"], 0.0)
        self.assertEqual(feats["cumulative_premove_rate"], 0.0)
        self.assertEqual(feats["current_endgame_clock_ratio"], 0.0)
        self.assertEqual(feats["phase_deliberation_ratio"], 1.0)
        self.assertEqual(feats["move_time_cv_10"], 0.0)
        self.assertEqual(feats["relative_move_time"], 2.5 / (180.0 / 40.0))
        self.assertAlmostEqual(feats["time_spent_ratio"], 2.5 / 180.0, places=4)
        self.assertEqual(feats["tank_move_count_10"], 0.0)

    def test_consecutive_fast_streak(self):
        tracker = PlayerMoveTracker("White", base_time=180, inc_time=0)
        # Move 1: fast (0.2s)
        f1 = tracker.record_move(0.2, 179.8, "Opening")
        self.assertEqual(f1["consecutive_fast_moves"], 1.0)
        
        # Move 2: fast (0.1s)
        f2 = tracker.record_move(0.1, 179.7, "Opening")
        self.assertEqual(f2["consecutive_fast_moves"], 2.0)
        
        # Move 3: deliberate (1.5s) -> streak resets to 0
        f3 = tracker.record_move(1.5, 178.2, "Opening")
        self.assertEqual(f3["consecutive_fast_moves"], 0.0)

    def test_premove_rate(self):
        tracker = PlayerMoveTracker("Black", base_time=60, inc_time=0)
        # 3 premoves (0.0s), 1 normal move (1.0s)
        tracker.record_move(0.0, 60.0, "Opening")
        tracker.record_move(0.0, 60.0, "Opening")
        tracker.record_move(0.0, 60.0, "Opening")
        f4 = tracker.record_move(1.0, 59.0, "Opening")
        
        self.assertEqual(f4["cumulative_premove_rate"], 3.0 / 4.0)
        self.assertEqual(f4["premove_rate_10"], 3.0 / 4.0)

    def test_zero_future_leakage(self):
        """
        Critical test: verifies that recording additional future moves
        does not retrospectively alter the features computed at move k.
        """
        # Run 1: record up to move 5
        tracker_short = PlayerMoveTracker("White", base_time=180, inc_time=2)
        history_short = []
        moves = [
            (1.5, 180.5, "Opening"),
            (0.8, 181.7, "Opening"),
            (0.0, 183.7, "Opening"),
            (4.2, 181.5, "Middlegame"),
            (12.0, 171.5, "Middlegame")
        ]
        for mt, clk, ph in moves:
            history_short.append(tracker_short.record_move(mt, clk, ph))
            
        # Run 2: record up to move 5, then continue to move 10 with extreme moves
        tracker_long = PlayerMoveTracker("White", base_time=180, inc_time=2)
        history_long = []
        future_moves = moves + [
            (35.0, 138.5, "Middlegame"),  # tank move
            (0.0, 140.5, "Endgame"),     # premove
            (0.2, 142.3, "Endgame"),
            (5.0, 139.3, "Endgame"),
            (1.1, 140.2, "Endgame")
        ]
        for mt, clk, ph in future_moves:
            history_long.append(tracker_long.record_move(mt, clk, ph))
            
        # Verify moves 1..5 in both runs have IDENTICAL feature values
        for i in range(5):
            for col in FEATURE_COLUMNS:
                val_short = history_short[i][col]
                val_long = history_long[i][col]
                self.assertAlmostEqual(
                    val_short, val_long, places=6,
                    msg=f"Leakage detected at move {i+1} on feature '{col}': short={val_short}, long={val_long}"
                )

    def test_determine_game_phase(self):
        board = chess.Board()
        # Initial board at move 1 is Opening
        self.assertEqual(determine_game_phase(board, 1), "Opening")
        
        # Board with only kings is Endgame
        empty_board = chess.Board(None)
        empty_board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        empty_board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
        self.assertEqual(determine_game_phase(empty_board, 40), "Endgame")


if __name__ == "__main__":
    unittest.main()
