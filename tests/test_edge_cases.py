"""
Test Suite: Edge Cases and Resilience
=====================================
Tests:
- PGN without clock information ([%clk] absent)
- Very short games (e.g. 4-ply Scholar's / Fool's mate)
- All-zero move-time windows (repeated premoves)
- Missing ratings in headers (defaults gracefully to 1500)
- Unsupported or unusual time controls
- Empty or malformed PGN strings
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference_features import parse_pgn_game, PlayerMoveTracker
from src.predict import analyze_pgn, predict_player

class TestEdgeCases(unittest.TestCase):
    def test_pgn_without_clock_comments(self):
        pgn_no_clk = """
[Event "Casual Game"]
[Site "Chess.com"]
[Date "2026.10.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[TimeControl "300+0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 5. d3 d6 1-0
"""
        parsed = parse_pgn_game(pgn_no_clk)
        self.assertFalse(parsed["has_clock_data"])
        self.assertIsNotNone(parsed["warning"])
        self.assertEqual(len(parsed["moves"]), 10)
        
        # Ensure analysis runs without crashing and clearly rejects behavioral timing claims
        res = analyze_pgn(pgn_no_clk)
        self.assertEqual(len(res["moves"]), 10)
        self.assertEqual(res["has_clock_data"], False)
        self.assertIn("UNAVAILABLE", res["summary"]["white"]["overall_prediction"])
        self.assertIn("Move clock data unavailable", res["warning"])

    def test_very_short_game(self):
        # 4-ply Scholar's mate
        short_pgn = """
[Event "Speed Game"]
[White "Tactician"]
[Black "Novice"]
[Result "1-0"]
[TimeControl "180+0"]

1. e4 {[%clk 0:02:59]} 1... e5 {[%clk 0:02:58]} 2. Qh5 {[%clk 0:02:57]} 2... Nc6 {[%clk 0:02:56]} 1-0
"""
        res = analyze_pgn(short_pgn)
        self.assertEqual(len(res["moves"]), 4)
        self.assertIn("Short game", res["warning"])
        self.assertIn("prediction", res["summary"]["white"])

    def test_all_zero_movetime_window(self):
        # Tracker with 10 consecutive premoves (0.0s)
        tracker = PlayerMoveTracker("White", base_time=60, inc_time=0)
        for _ in range(10):
            feats = tracker.record_move(0.0, 60.0, "Opening")
            
        self.assertEqual(feats["consecutive_fast_moves"], 10.0)
        self.assertEqual(feats["premove_rate_10"], 1.0)
        self.assertEqual(feats["cumulative_premove_rate"], 1.0)
        self.assertEqual(feats["move_time_cv_10"], 0.0)  # No division by zero

        # Predict should recognize extreme AI-like pacing
        pred = predict_player(feats)
        self.assertEqual(pred["predicted_class"], "AI")
        self.assertGreater(pred["ai_probability"], 0.80)

    def test_missing_ratings(self):
        pgn_no_elo = """
[Event "Rated Blitz game"]
[White "Unknown1"]
[Black "Unknown2"]
[Result "0-1"]
[TimeControl "180+2"]

1. d4 {[%clk 0:03:00]} 1... d5 {[%clk 0:03:00]} 0-1
"""
        parsed = parse_pgn_game(pgn_no_elo)
        self.assertEqual(parsed["metadata"]["white_elo"], "1500")
        self.assertEqual(parsed["metadata"]["black_elo"], "1500")

    def test_unsupported_time_control_string(self):
        pgn_odd_tc = """
[Event "Correspondence"]
[White "Alice"]
[Black "Bob"]
[Result "*"]
[TimeControl "-"]

1. e4 1... e5 *
"""
        parsed = parse_pgn_game(pgn_odd_tc)
        # Should fall back to standard base_time (180, 0)
        self.assertEqual(parsed["metadata"]["base_time"], 180)
        self.assertEqual(parsed["metadata"]["increment"], 0)

    def test_empty_pgn_error_handling(self):
        with self.assertRaises(ValueError):
            parse_pgn_game("")

    def test_zero_move_aborted_game(self):
        aborted_pgn = """
[Event "Aborted Game"]
[White "Player1"]
[Black "Player2"]
[Result "*"]
[TimeControl "180+0"]

*
"""
        parsed = parse_pgn_game(aborted_pgn)
        self.assertEqual(len(parsed["moves"]), 0)
        self.assertIn("no moves", parsed["warning"])


if __name__ == "__main__":
    unittest.main()
