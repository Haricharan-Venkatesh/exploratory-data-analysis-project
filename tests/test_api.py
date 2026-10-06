"""
Test Suite: FastAPI Endpoints & Static Asset Delivery
=====================================================
Tests:
- GET /health returns 200 with model version and 10 features
- POST /predict scores single 10-feature vector
- POST /analyze-pgn runs full game analysis and returns moves & profiles
- GET /demo-games returns bundled demo games
- GET /feature-importance returns empirical rankings
- GET / returns 200 with index.html dashboard
"""

import os
import sys
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.backend.main import app

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["feature_count"], 10)
        self.assertEqual(data["operating_threshold"], 0.50)
        self.assertIn("LightGBM", data["model_name"])

    def test_predict_single_vector(self):
        payload = {
            "consecutive_fast_moves": 0.0,
            "premove_rate_10": 0.1,
            "cumulative_premove_rate": 0.1,
            "current_endgame_clock_ratio": 0.0,
            "phase_deliberation_ratio": 1.5,
            "move_time_cv_10": 0.9,
            "time_pressure_jitter": 0.0,
            "relative_move_time": 1.0,
            "time_spent_ratio": 0.02,
            "tank_move_count_10": 0.0
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("ai_probability", data)
        self.assertIn("human_probability", data)
        self.assertIn("prediction", data)
        self.assertIn("behavioral_signals", data)

    def test_analyze_pgn_endpoint(self):
        sample_pgn = """
[Event "Rated Blitz game"]
[White "PlayerOne"]
[Black "PlayerTwo"]
[Result "1-0"]
[TimeControl "180+0"]

1. e4 {[%clk 0:03:00]} 1... c5 {[%clk 0:03:00]} 2. Nf3 {[%clk 0:02:58]} 2... d6 {[%clk 0:02:57]} 1-0
"""
        res = self.client.post("/analyze-pgn", json={"pgn": sample_pgn})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("summary", data)
        self.assertIn("white", data["summary"])
        self.assertIn("black", data["summary"])
        self.assertEqual(len(data["moves"]), 4)

    def test_demo_games_endpoint(self):
        res = self.client.get("/demo-games")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("demo_games", data)
        self.assertGreaterEqual(len(data["demo_games"]), 1)

    def test_feature_importance_endpoint(self):
        res = self.client.get("/feature-importance")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("features", data)
        self.assertEqual(len(data["features"]), 10)

    def test_static_frontend_delivery(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("CHESS BEHAVIORAL AI DETECTOR", res.text)


if __name__ == "__main__":
    unittest.main()
