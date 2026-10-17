"""
tests.test_v5_complexity
========================
Tests for Version 5 Position-Complexity-Aware Feature Engineering:
- Column schema definitions and backward-compatibility defaults
- Feature extraction from the ChessFraud benchmark DataFrame
- Engine row extraction with complexity features
- Physical match zero-leakage grouping
- Experimental model serialization, loading, and inference
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import pytest

from chess_eda.ml.dataset import DEFAULT_DATA_PATH, load_chess_fraud_dataset
from chess_eda.ml.features import (
    ML_FEATURE_COLUMNS,
    V5_COMPLEXITY_FEATURE_COLUMNS,
    V5_ALL_FEATURE_COLUMNS,
    extract_player_features_from_df,
    extract_player_features_from_engine_rows,
)
from chess_eda.ml.v5_experiments import run_v5_complexity_experiment


def test_v5_feature_constants():
    """Verify schema column constants and expected counts."""
    assert len(ML_FEATURE_COLUMNS) == 13
    assert len(V5_COMPLEXITY_FEATURE_COLUMNS) == 5
    assert len(V5_ALL_FEATURE_COLUMNS) == 18

    expected_complexity = [
        "unstable_top1_match_rate",
        "stable_top1_match_rate",
        "depth_drop_rate",
        "critical_top1_match_rate",
        "critical_mean_cp_loss",
    ]
    for col in expected_complexity:
        assert col in V5_COMPLEXITY_FEATURE_COLUMNS
        assert col in V5_ALL_FEATURE_COLUMNS


def test_extract_player_features_df_backward_compatibility():
    """Verify that default include_complexity=False preserves the exact 13 baseline columns."""
    if not os.path.isfile(DEFAULT_DATA_PATH):
        pytest.skip(f"Benchmark file not found at {DEFAULT_DATA_PATH}")

    df = load_chess_fraud_dataset(DEFAULT_DATA_PATH)
    base_df = extract_player_features_from_df(df, include_complexity=False)

    for col in ML_FEATURE_COLUMNS:
        assert col in base_df.columns
    for col in V5_COMPLEXITY_FEATURE_COLUMNS:
        assert col not in base_df.columns


def test_extract_player_features_df_with_complexity():
    """Verify that include_complexity=True extracts all 18 features with valid ranges."""
    if not os.path.isfile(DEFAULT_DATA_PATH):
        pytest.skip(f"Benchmark file not found at {DEFAULT_DATA_PATH}")

    df = load_chess_fraud_dataset(DEFAULT_DATA_PATH)
    v5_df = extract_player_features_from_df(df, include_complexity=True)

    assert len(v5_df) == 1010
    for col in V5_ALL_FEATURE_COLUMNS:
        assert col in v5_df.columns

    # Verify rates are in valid [0, 1] range (or NaN if no moves in condition)
    for rate_col in ["unstable_top1_match_rate", "stable_top1_match_rate", "critical_top1_match_rate"]:
        valid_vals = v5_df[rate_col].dropna()
        assert (valid_vals >= 0.0).all()
        assert (valid_vals <= 1.0).all()

    # Verify depth drop relationship: stable - unstable
    has_both = v5_df["stable_top1_match_rate"].notna() & v5_df["unstable_top1_match_rate"].notna()
    diff = v5_df.loc[has_both, "stable_top1_match_rate"] - v5_df.loc[has_both, "unstable_top1_match_rate"]
    np.testing.assert_allclose(v5_df.loc[has_both, "depth_drop_rate"].values, diff.values, atol=1e-5)


def test_extract_player_features_from_engine_rows_complexity():
    """Verify engine row extraction computes complexity features when requested."""
    # Synthesize sample merged engine rows
    sample_rows = [
        {
            "player_color": "White",
            "ply_number": 22,
            "is_engine_top1": 1,
            "centipawn_loss": 10.0,
            "engine_score_before_cp": 50,
            "engine_best_move_depth1": "e2e4",
            "engine_best_move_uci": "e2e4",
            "uci": "e2e4",
        },
        {
            "player_color": "White",
            "ply_number": 24,
            "is_engine_top1": 0,
            "centipawn_loss": 80.0,
            "engine_score_before_cp": 120,
            "engine_best_move_depth1": "g1f3",
            "engine_best_move_uci": "d2d4",
            "uci": "c2c4",
        },
    ]

    base_feats = extract_player_features_from_engine_rows(sample_rows, "White", include_complexity=False)
    assert "critical_top1_match_rate" not in base_feats
    assert len(base_feats) == len(ML_FEATURE_COLUMNS)

    v5_feats = extract_player_features_from_engine_rows(sample_rows, "White", include_complexity=True)
    assert len(v5_feats) == len(V5_ALL_FEATURE_COLUMNS)
    assert v5_feats["critical_top1_match_rate"] == 0.5  # 1 out of 2 matched in |score| <= 200
    assert v5_feats["critical_mean_cp_loss"] == 45.0   # (10 + 80) / 2
    assert v5_feats["stable_top1_match_rate"] == 1.0   # 1 out of 1 matched when d1 == best
    assert v5_feats["unstable_top1_match_rate"] == 0.0 # 0 out of 1 matched when d1 != best
    assert v5_feats["depth_drop_rate"] == 1.0          # 1.0 - 0.0 = 1.0


def test_experimental_model_artifacts_exist_and_predict():
    """Verify serialized V5 experimental models exist, load cleanly, and produce valid inference."""
    exp_dir = os.path.join("models", "experimental")
    rf_file = os.path.join(exp_dir, "chess_fraud_rf_v5_exp.joblib")
    lr_file = os.path.join(exp_dir, "chess_fraud_lr_v5_exp.joblib")
    meta_file = os.path.join(exp_dir, "v5_experiment_metadata.json")

    assert os.path.isfile(rf_file), f"Missing {rf_file}"
    assert os.path.isfile(lr_file), f"Missing {lr_file}"
    assert os.path.isfile(meta_file), f"Missing {meta_file}"

    # Load pipelines
    rf_pipe = joblib.load(rf_file)
    lr_pipe = joblib.load(lr_file)

    # Test inference on a 18-feature dummy row
    dummy_input = np.ones((1, len(V5_ALL_FEATURE_COLUMNS))) * 0.5
    rf_prob = rf_pipe.predict_proba(dummy_input)[0]
    lr_prob = lr_pipe.predict_proba(dummy_input)[0]

    assert len(rf_prob) == 2
    assert pytest.approx(sum(rf_prob), abs=1e-4) == 1.0
    assert len(lr_prob) == 2
    assert pytest.approx(sum(lr_prob), abs=1e-4) == 1.0

    # Verify metadata contents
    with open(meta_file, encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["total_player_games"] == 1010
    assert meta["total_physical_matches"] == 505
    assert "comparisons" in meta
    assert "ablations_rf" in meta


def test_v5_experiment_runner_dry_run(tmp_path):
    """Verify run_v5_complexity_experiment runs end-to-end to a temporary directory."""
    if not os.path.isfile(DEFAULT_DATA_PATH):
        pytest.skip(f"Benchmark file not found at {DEFAULT_DATA_PATH}")

    out_dir = str(tmp_path / "exp")
    report = run_v5_complexity_experiment(
        data_path=DEFAULT_DATA_PATH,
        output_dir=out_dir,
        random_state=42,
        n_splits=5,
        save_models=True,
        verbose=False,
    )

    assert "comparisons" in report
    assert "logistic_regression" in report["comparisons"]
    assert "random_forest" in report["comparisons"]
    assert report["comparisons"]["random_forest"]["augmented"]["roc_auc"] > 0.75


def test_analyse_game_returns_depth1_move_format():
    """Verify analyse_game() returns depth-1 best move in UCI format matching training schema."""
    stockfish_path = os.environ.get("STOCKFISH_PATH", r"C:\stockfish\stockfish.exe")
    if not os.path.isfile(stockfish_path):
        pytest.skip(f"Stockfish not found at {stockfish_path}")

    from chess_eda.engine_analysis import analyse_game
    from chess_eda.pgn_parser import load_first_game

    audited_path = os.path.join("data", "raw", "unlabeled", "thegreatimu_game.pgn")
    game = load_first_game(audited_path)

    rows, meta = analyse_game(game, stockfish_path=stockfish_path, depth=5, include_depth1=True)
    assert len(rows) > 0
    for row in rows:
        assert "engine_best_move_depth1" in row
        d1_move = row["engine_best_move_depth1"]
        assert isinstance(d1_move, str)
        # Should be non-empty UCI string of length 4 or 5 (e.g. 'e2e4', 'e7e8q')
        assert len(d1_move) in (4, 5)


def test_stability_features_use_actual_depth1_data_not_fallback():
    """Verify stability features genuinely calculate rates from move data rather than imputing medians."""
    # Move 1: d1 == best == player -> stable match
    # Move 2: d1 != best, player == best -> unstable match
    # Move 3: d1 != best, player != best -> unstable non-match
    sample_rows = [
        {
            "player_color": "Black",
            "ply_number": 22,
            "is_engine_top1": 1,
            "centipawn_loss": 0.0,
            "engine_score_before_cp": -20,
            "engine_best_move_depth1": "c7c5",
            "engine_best_move_uci": "c7c5",
            "uci": "c7c5",
        },
        {
            "player_color": "Black",
            "ply_number": 24,
            "is_engine_top1": 1,
            "centipawn_loss": 0.0,
            "engine_score_before_cp": -15,
            "engine_best_move_depth1": "g8f6",
            "engine_best_move_uci": "b8c6",
            "uci": "b8c6",
        },
        {
            "player_color": "Black",
            "ply_number": 26,
            "is_engine_top1": 0,
            "centipawn_loss": 50.0,
            "engine_score_before_cp": -30,
            "engine_best_move_depth1": "d7d6",
            "engine_best_move_uci": "g8f6",
            "uci": "e7e6",
        },
    ]

    features = extract_player_features_from_engine_rows(sample_rows, "Black", include_complexity=True)

    # 1 move in stable condition (d1 == best), which was matched: 1.0
    assert features["stable_top1_match_rate"] == 1.0
    # 2 moves in unstable condition (d1 != best), 1 of which matched: 0.5
    assert features["unstable_top1_match_rate"] == 0.5
    # depth_drop_rate = stable - unstable = 1.0 - 0.5 = 0.5
    assert features["depth_drop_rate"] == 0.5
    # None of the stability features should be NaN when data is present
    for k in ["stable_top1_match_rate", "unstable_top1_match_rate", "depth_drop_rate"]:
        assert not np.isnan(features[k])


def test_live_pgn_v5_inference_and_baseline_invariance():
    """Verify live PGN runs baseline invariance and genuinely populates V5 complexity features."""
    stockfish_path = os.environ.get("STOCKFISH_PATH", r"C:\stockfish\stockfish.exe")
    if not os.path.isfile(stockfish_path):
        pytest.skip(f"Stockfish not found at {stockfish_path}")

    from chess_eda.ml.predict import predict_game

    audited_path = os.path.join("data", "raw", "unlabeled", "thegreatimu_game.pgn")
    # 1. Ordinary baseline analysis does NOT run Version 5 or attach experimental predictions
    report_base = predict_game(audited_path, stockfish_path=stockfish_path, depth=10, verbose=False)
    assert pytest.approx(report_base["black_player"]["engine_assisted_likelihood"], abs=0.005) == 0.513
    assert report_base.get("v5_experimental") is None

    # 2. When include_v5=True at depth 10, correctly enforces depth-15 requirement
    report_v5 = predict_game(
        audited_path,
        stockfish_path=stockfish_path,
        depth=10,
        verbose=False,
        include_v5=True,
    )
    v5 = report_v5.get("v5_experimental")
    assert v5 is not None
    assert v5["is_depth_compatible"] is False
    assert v5["status"] == "incompatible_depth"
    assert v5["white_score"] is None
    assert v5["black_score"] is None

    # 3. When require_v5_depth15=False and include_v5=True, verify stability features are genuinely computed (not imputed medians)
    report_unverified = predict_game(
        audited_path,
        stockfish_path=stockfish_path,
        depth=10,
        verbose=False,
        include_v5=True,
        require_v5_depth15=False,
    )
    v5_unverified = report_unverified.get("v5_experimental")
    assert v5_unverified is not None
    assert v5_unverified["black_score"] is not None
    black_feats = v5_unverified["features_black"]
    assert not np.isnan(black_feats["unstable_top1_match_rate"])
    assert not np.isnan(black_feats["stable_top1_match_rate"])
    assert not np.isnan(black_feats["depth_drop_rate"])
    assert 0.0 <= black_feats["unstable_top1_match_rate"] <= 1.0
    assert 0.0 <= black_feats["stable_top1_match_rate"] <= 1.0


def test_v5_training_depth_consistency_schema():
    """Verify that Version 5 training explicitly uses depth-1 and depth-15 moves."""
    from chess_eda.ml.features import extract_player_features_from_df

    sample_df = pd.DataFrame({
        "game_id": [1, 1],
        "player_id": ["p1", "p1"],
        "player_color": ["white", "white"],
        "is_cheating_player_game": [0, 0],
        "is_used": [True, True],
        "ply_number": [22, 24],
        "half_move": [22, 24],
        "centipawn_loss": [10.0, 20.0],
        "normalized_centipawn_loss": [0.01, 0.02],
        "eval_before": [50.0, 60.0],
        "eval_after": [40.0, 40.0],
        "move_player": ["e2e4", "g1f3"],
        "move_stockfish_1": ["e2e4", "d2d4"],
        "move_stockfish_15": ["e2e4", "g1f3"],
    })
    feats = extract_player_features_from_df(sample_df, include_complexity=True)
    assert "unstable_top1_match_rate" in feats.columns
    assert "stable_top1_match_rate" in feats.columns
    assert "depth_drop_rate" in feats.columns
    # Ply 1: SF1 == SF15 == player -> stable match (1.0)
    # Ply 2: SF1 != SF15, player == SF15 -> unstable match (1.0)
    assert feats.loc[0, "stable_top1_match_rate"] == 1.0
    assert feats.loc[0, "unstable_top1_match_rate"] == 1.0
    assert feats.loc[0, "depth_drop_rate"] == 0.0


def test_dashboard_v5_comparison_ui_absent():
    """Verify app.py removes Version 5 experimental comparison from user interface while keeping Version 4 intact."""
    app_path = "app.py"
    assert os.path.isfile(app_path)
    with open(app_path, "r", encoding="utf-8") as fh:
        code = fh.read()

    # Version 5 experimental UI must be absent from user dashboard
    assert "Compare with Version 5 Experimental Model" not in code
    assert "Version 5 Experimental Model Comparison" not in code
    assert "Version 5 Experimental: Position-Complexity Feature Investigation" not in code
    assert "reanalyze_d15_btn" not in code

    # Version 4 dashboard elements must remain present and intact
    assert "Version 4 Interactive Dashboard" in code
    assert "Player-Level Engine-Assistance Estimates" in code
    assert "ChessFraud Benchmark & Validation Metrics" in code

