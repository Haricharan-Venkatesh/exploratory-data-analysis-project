"""
tests/test_v3_ml.py
===================
Unit and integration tests for Version 3 Supervised Machine Learning baseline.

Verifies:
- Dataset loading and metadata schema
- Feature extraction and compatibility with Version 2
- Group-aware splitting without player leakage
- Model pipeline training and metric calculations
- Model serialization and loading
- Local PGN prediction workflow and player-level probability calculations
- Error handling on invalid/empty PGNs and missing Stockfish
"""

import json
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from chess_eda.ml.dataset import (
    DEFAULT_DATA_PATH,
    inspect_dataset_metadata,
    load_chess_fraud_dataset,
)
from chess_eda.ml.features import (
    ML_FEATURE_COLUMNS,
    compute_normalized_cp_loss,
    extract_player_features_from_df,
    extract_player_features_from_engine_rows,
)
from chess_eda.ml.train import (
    evaluate_model_pipeline,
    train_baseline_models,
)
from chess_eda.ml.predict import predict_game


def test_normalized_cp_loss_math():
    """Verify that normalized centipawn loss matches the Lichess Win% formula."""
    # When eval_before == eval_after, loss must be 0.0
    assert compute_normalized_cp_loss(100.0, 100.0) == 0.0
    # When eval drops from 500 to 0, win probability decreases, loss > 0
    loss = compute_normalized_cp_loss(500.0, 0.0)
    assert 0.0 < loss <= 1.0
    # When eval improves from 0 to 500, loss is 0 (max(0, p(before) - p(after)))
    assert compute_normalized_cp_loss(0.0, 500.0) == 0.0


def test_chess_fraud_dataset_loading_and_metadata():
    """Verify that the benchmark parquet file loads correctly and metadata summarizes schema."""
    if not os.path.isfile(DEFAULT_DATA_PATH):
        pytest.skip(f"Benchmark file not found at {DEFAULT_DATA_PATH}")

    df = load_chess_fraud_dataset(DEFAULT_DATA_PATH)
    assert df.shape == (38510, 33)
    assert "game_id" in df.columns
    assert "player_id" in df.columns
    assert "is_cheating_player_game" in df.columns
    assert "is_cheating_move" in df.columns

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = os.path.join(tmp_dir, "meta.json")
        meta = inspect_dataset_metadata(df, save_path=meta_file)
        assert os.path.isfile(meta_file)
        assert meta["summary"]["total_move_rows"] == 38510
        assert meta["summary"]["total_player_games"] == 1010
        assert meta["summary"]["unique_players"] == 49
        assert meta["license"] == "CC BY 4.0"


def test_player_game_feature_extraction():
    """Verify that player-game features extracted from DataFrame align with ML_FEATURE_COLUMNS."""
    if not os.path.isfile(DEFAULT_DATA_PATH):
        pytest.skip(f"Benchmark file not found at {DEFAULT_DATA_PATH}")

    df = load_chess_fraud_dataset(DEFAULT_DATA_PATH)
    # Take a small subset of 4 games for speed
    sample_games = df["game_id"].unique()[:4]
    sub_df = df[df["game_id"].isin(sample_games)]

    feat_df = extract_player_features_from_df(sub_df)
    assert len(feat_df) == len(sample_games)
    for col in ML_FEATURE_COLUMNS:
        assert col in feat_df.columns
    assert "is_cheating" in feat_df.columns
    assert "player_id" in feat_df.columns
    assert "physical_game_id" in feat_df.columns


def test_feature_extraction_from_engine_rows():
    """Verify feature extraction from mock engine analysis rows."""
    mock_rows = [
        {
            "ply_number": 1,
            "player_color": "White",
            "score_before": 10,
            "score_after": 10,
            "centipawn_loss": 0,
            "is_engine_top1": 1,
        },
        {
            "ply_number": 2,
            "player_color": "Black",
            "score_before": 10,
            "score_after": 25,
            "centipawn_loss": 15,
            "is_engine_top1": 0,
        },
        {
            "ply_number": 21,
            "player_color": "White",
            "score_before": 30,
            "score_after": 10,
            "centipawn_loss": 20,
            "is_engine_top1": 1,
        },
        {
            "ply_number": 22,
            "player_color": "Black",
            "score_before": 10,
            "score_after": 250,
            "centipawn_loss": 240,
            "is_engine_top1": 0,
        },
    ]

    white_feats = extract_player_features_from_engine_rows(mock_rows, "White")
    black_feats = extract_player_features_from_engine_rows(mock_rows, "Black")

    assert white_feats["move_count"] == 2
    assert black_feats["move_count"] == 2
    assert white_feats["mean_centipawn_loss"] == 10.0  # (0 + 20) / 2
    assert black_feats["blunder_rate"] == 0.5          # 1 blunder (>200) out of 2 plies
    assert white_feats["top1_match_rate"] == 1.0       # top1 on ply 21


def test_physical_match_group_integrity():
    """Verify that physical match normalization correctly groups White and Black sides of the same game."""
    from chess_eda.ml.features import extract_physical_match_id

    assert extract_physical_match_id("01zsFrq4_w") == "01zsFrq4"
    assert extract_physical_match_id("01zsFrq4_b") == "01zsFrq4"
    assert extract_physical_match_id("game123") == "game123"

    if os.path.isfile(DEFAULT_DATA_PATH):
        df = load_chess_fraud_dataset(DEFAULT_DATA_PATH)
        pg_df = extract_player_features_from_df(df)
        assert "physical_game_id" in pg_df.columns
        assert pg_df["physical_game_id"].nunique() == 505
        # Each physical match should have exactly 2 player-game records
        counts = pg_df["physical_game_id"].value_counts()
        assert (counts == 2).all()


def test_training_and_serialization(tmp_path):
    """Verify that baseline models train, report metrics, and save serialized pipelines."""
    if not os.path.isfile(DEFAULT_DATA_PATH):
        pytest.skip(f"Benchmark file not found at {DEFAULT_DATA_PATH}")

    model_dir = str(tmp_path / "models")
    report = train_baseline_models(
        data_path=DEFAULT_DATA_PATH,
        output_dir=model_dir,
        random_state=42,
        n_splits=5,
        test_size=0.20,
        verbose=False,
    )

    # Check that report structure is valid
    assert "models" in report
    assert "logistic_regression" in report["models"]
    assert "random_forest" in report["models"]
    assert report["split"]["physical_game_leakage_detected"] is False

    lr_acc = report["models"]["logistic_regression"]["test_metrics"]["accuracy"]
    rf_acc = report["models"]["random_forest"]["test_metrics"]["accuracy"]
    majority_acc = report["class_balance"]["majority_class_baseline_accuracy"]

    # Both models should outperform the majority class baseline
    assert lr_acc > majority_acc
    assert rf_acc > majority_acc

    # Check serialized files
    rf_file = os.path.join(model_dir, "chess_fraud_rf.joblib")
    lr_file = os.path.join(model_dir, "chess_fraud_lr.joblib")
    meta_file = os.path.join(model_dir, "model_metadata.json")

    assert os.path.isfile(rf_file)
    assert os.path.isfile(lr_file)
    assert os.path.isfile(meta_file)

    with open(meta_file, encoding="utf-8") as fh:
        loaded_meta = json.load(fh)
        assert loaded_meta["dataset"] == report["dataset"]


def test_predict_game_error_handling():
    """Verify that predict_game handles missing files and invalid paths gracefully."""
    with pytest.raises(FileNotFoundError, match="PGN file not found"):
        predict_game(
            pgn_path="non_existent_file.pgn",
            stockfish_path="C:\\stockfish\\stockfish.exe",
            verbose=False,
        )

    with pytest.raises(FileNotFoundError, match="Fitted model not found"):
        predict_game(
            pgn_path="data/sample.pgn",
            stockfish_path="C:\\stockfish\\stockfish.exe",
            model_path="non_existent_model.joblib",
            verbose=False,
        )


def test_predict_game_empty_pgn(tmp_path):
    """Verify that games with 0 mainline moves raise an informative ValueError."""
    empty_pgn = tmp_path / "empty.pgn"
    empty_pgn.write_text('[Event "Empty"]\n[Result "*"]\n\n*', encoding="utf-8")

    from chess_eda.engine_analysis import EngineAnalysisError
    with pytest.raises((ValueError, EngineAnalysisError)):
        predict_game(
            pgn_path=str(empty_pgn),
            stockfish_path="C:\\stockfish\\stockfish.exe",
            model_path="models/chess_fraud_rf.joblib",
            verbose=False,
        )


def test_end_to_end_prediction_with_stockfish(capsys):
    """Verify end-to-end inference on data/sample.pgn using the trained model and Stockfish."""
    if not os.path.isfile("C:\\stockfish\\stockfish.exe"):
        pytest.skip("Stockfish not available at C:\\stockfish\\stockfish.exe")
    if not os.path.isfile("models/chess_fraud_rf.joblib"):
        pytest.skip("Model not found at models/chess_fraud_rf.joblib")

    report = predict_game(
        pgn_path="data/sample.pgn",
        stockfish_path="C:\\stockfish\\stockfish.exe",
        model_path="models/chess_fraud_rf.joblib",
        depth=6,  # shallow depth for fast unit test
        verbose=True,
    )

    assert "white_player" in report
    assert "black_player" in report
    # Confirm matchup and joint probabilities are completely removed
    assert "matchup" not in report

    # White and Black probabilities must sum to 1.0
    w_assist = report["white_player"]["engine_assisted_likelihood"]
    w_human = report["white_player"]["human_like_likelihood"]
    assert pytest.approx(w_assist + w_human, abs=1e-3) == 1.0

    b_assist = report["black_player"]["engine_assisted_likelihood"]
    b_human = report["black_player"]["human_like_likelihood"]
    assert pytest.approx(b_assist + b_human, abs=1e-3) == 1.0

    # Also verify uncalibrated score aliases
    assert report["white_player"]["engine_assistance_score"] == w_assist
    assert report["white_player"]["complementary_human_like_score"] == w_human
    assert pytest.approx(report["white_player"]["complementary_human_like_score"], abs=1e-3) == (1.0 - w_assist)

    assert report["black_player"]["engine_assistance_score"] == b_assist
    assert report["black_player"]["complementary_human_like_score"] == b_human
    assert pytest.approx(report["black_player"]["complementary_human_like_score"], abs=1e-3) == (1.0 - b_assist)

    # Verify stdout text format with uncalibrated score terminology
    captured = capsys.readouterr().out
    assert "PLAYER-LEVEL ENGINE-ASSISTANCE ESTIMATES" in captured
    assert "Adolf Anderssen (White)" in captured
    assert "Lionel Kieseritzky (Black)" in captured
    assert "Engine-assistance score (uncalibrated):" in captured
    assert "Complementary human-like score (uncalibrated):" in captured
    assert "Classification (50% software threshold):" in captured
    assert "METHODOLOGICAL NOTE & SCIENTIFIC DISCLAIMER:" in captured
    assert "100% minus the engine-assistance score" in captured
    assert "operational software boundary" in captured

    # Confirm misleading descriptions DO NOT appear
    assert "Engine-assisted likelihood:" not in captured
    assert "Human-like likelihood:" not in captured

    # Confirm the four joint-probability lines and matchup assessment DO NOT appear
    assert "MATCHUP ASSESSMENT" not in captured
    assert "Joint Probabilities" not in captured
    assert "human vs human" not in captured
    assert "engine-assisted vs human" not in captured
    assert "human vs engine-assisted" not in captured
    assert "engine-assisted vs engine-assisted" not in captured


def test_prediction_with_logistic_regression(capsys):
    """Verify inference also functions seamlessly with Logistic Regression."""
    if not os.path.isfile("C:\\stockfish\\stockfish.exe"):
        pytest.skip("Stockfish not available at C:\\stockfish\\stockfish.exe")
    if not os.path.isfile("models/chess_fraud_lr.joblib"):
        pytest.skip("Model not found at models/chess_fraud_lr.joblib")

    report = predict_game(
        pgn_path="data/sample.pgn",
        stockfish_path="C:\\stockfish\\stockfish.exe",
        model_path="models/chess_fraud_lr.joblib",
        depth=6,
        verbose=True,
    )

    assert "white_player" in report
    assert "black_player" in report
    assert "matchup" not in report

    captured = capsys.readouterr().out
    assert "PLAYER-LEVEL ENGINE-ASSISTANCE ESTIMATES" in captured
    assert "Engine-assistance score (uncalibrated):" in captured
    assert "Complementary human-like score (uncalibrated):" in captured
    assert "Engine-assisted likelihood:" not in captured
    assert "Human-like likelihood:" not in captured
    assert "MATCHUP ASSESSMENT" not in captured
    assert "Joint Probabilities" not in captured
