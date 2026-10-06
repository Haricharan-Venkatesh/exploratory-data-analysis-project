"""
Test Suite: Model Loading, Schema, and Prediction Sanity
========================================================
Tests:
- best_model.pkl loads successfully
- exactly 10 features matching feature_columns.json
- feature ordering is preserved
- predict_player produces valid probabilities in [0, 1]
- decision thresholding behaves as expected
"""

import os
import sys
import unittest
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_player, get_model_metadata, get_feature_columns
from src.inference_features import FEATURE_COLUMNS

class TestModelIntegrity(unittest.TestCase):
    def setUp(self):
        self.model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
        self.columns_path = os.path.join(PROJECT_ROOT, "models", "feature_columns.json")
        self.metadata_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

    def test_model_files_exist(self):
        self.assertTrue(os.path.exists(self.model_path), "best_model.pkl missing")
        self.assertTrue(os.path.exists(self.columns_path), "feature_columns.json missing")
        self.assertTrue(os.path.exists(self.metadata_path), "model_metadata.json missing")

    def test_feature_columns_schema(self):
        cols = get_feature_columns()
        self.assertEqual(len(cols), 10, f"Expected 10 features, got {len(cols)}")
        self.assertEqual(cols, FEATURE_COLUMNS, "Feature order mismatch between inference and training")

    def test_model_metadata_contents(self):
        meta = get_model_metadata()
        self.assertIn("model_name", meta)
        self.assertIn("operating_threshold", meta)
        self.assertEqual(meta["operating_threshold"], 0.50)
        self.assertIn("test_metrics_unseen_game", meta)
        self.assertGreater(meta["test_metrics_unseen_game"]["roc_auc"], 0.85)

    def test_predict_player_output_structure(self):
        sample_input = {
            "consecutive_fast_moves": 2.0,
            "premove_rate_10": 0.20,
            "cumulative_premove_rate": 0.15,
            "current_endgame_clock_ratio": 0.0,
            "phase_deliberation_ratio": 1.4,
            "move_time_cv_10": 0.85,
            "time_pressure_jitter": 0.0,
            "relative_move_time": 1.2,
            "time_spent_ratio": 0.04,
            "tank_move_count_10": 0.0
        }
        res = predict_player(sample_input, threshold=0.50)
        self.assertIn("prediction", res)
        self.assertIn("predicted_class", res)
        self.assertIn("ai_probability", res)
        self.assertIn("human_probability", res)
        self.assertIn("behavioral_signals", res)
        self.assertIn("disclaimer", res)

        # Probabilities sum to 1.0
        self.assertAlmostEqual(res["ai_probability"] + res["human_probability"], 1.0, places=3)
        self.assertGreaterEqual(res["ai_probability"], 0.0)
        self.assertLessEqual(res["ai_probability"], 1.0)

    def test_threshold_sensitivity(self):
        sample_input = {col: 0.1 for col in FEATURE_COLUMNS}
        res_low = predict_player(sample_input, threshold=0.01)
        res_high = predict_player(sample_input, threshold=0.99)
        self.assertEqual(res_low["predicted_class"], "AI")
        self.assertEqual(res_high["predicted_class"], "HUMAN")


if __name__ == "__main__":
    unittest.main()
