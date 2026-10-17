"""
tests/test_v4_app.py
====================
Unit and integration tests for Version 4 Streamlit dashboard helper logic.

Verifies:
- Model metadata loading and schema integrity
- Metric definitions and educational descriptions
- Feature descriptions for all 13 ML features
- Preset game paths and integrity
- PGN input validation (valid, empty, and no-move games)
- Temporary PGN file saving and cleanup
- Chart data extraction and quality breakdown calculation
- app.py import and compilation syntax
"""

import os
import py_compile
import pytest
import pandas as pd

from chess_eda.dashboard import (
    DEFAULT_METADATA_PATH,
    extract_eda_chart_data,
    get_feature_descriptions,
    get_metric_definitions,
    get_preset_games,
    load_model_metadata,
    save_uploaded_pgn_to_temp,
    validate_pgn_input,
)
from chess_eda.ml.features import ML_FEATURE_COLUMNS


def test_load_model_metadata():
    """Verify loading model_metadata.json contains valid evaluation metrics."""
    if not os.path.isfile(DEFAULT_METADATA_PATH):
        pytest.skip(f"Metadata not found at {DEFAULT_METADATA_PATH}")

    data = load_model_metadata(DEFAULT_METADATA_PATH)
    assert "models" in data
    assert "logistic_regression" in data["models"]
    assert "random_forest" in data["models"]

    # Verify split metadata
    assert "split" in data
    assert data["split"]["physical_game_leakage_detected"] is False
    assert "StratifiedGroupKFold" in data["split"]["strategy"]
    assert data["total_physical_matches"] == 505
    assert data["total_player_games"] == 1010

    lr_metrics = data["models"]["logistic_regression"]["test_metrics"]
    rf_metrics = data["models"]["random_forest"]["test_metrics"]

    assert "accuracy" in lr_metrics
    assert "precision" in lr_metrics
    assert "recall" in lr_metrics
    assert "f1_score" in lr_metrics
    assert "roc_auc" in lr_metrics

    assert 0.0 < lr_metrics["accuracy"] < 1.0
    assert 0.0 < rf_metrics["accuracy"] < 1.0


def test_get_metric_definitions():
    """Verify definitions are available for key ML metrics."""
    defs = get_metric_definitions()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]:
        assert metric in defs
        assert "short" in defs[metric]
        assert "detailed" in defs[metric]
        assert "benchmark_context" in defs[metric]


def test_get_feature_descriptions():
    """Verify that all 13 ML features have descriptive definitions."""
    desc = get_feature_descriptions()
    for col in ML_FEATURE_COLUMNS:
        assert col in desc, f"Missing description for feature: {col}"
        assert len(desc[col]) > 10


def test_get_preset_games():
    """Verify preset games dictionary returns existing file paths."""
    presets = get_preset_games()
    assert len(presets) >= 3
    for name, info in presets.items():
        assert "path" in info
        assert "description" in info
        assert "type" in info
        assert os.path.isfile(info["path"]), f"Preset file missing: {info['path']}"


def test_validate_pgn_input_valid():
    """Verify valid PGN text passes validation and returns a parsed game."""
    valid_pgn = (
        '[Event "Test Game"]\n'
        '[White "Player 1"]\n'
        '[Black "Player 2"]\n'
        '[Result "1-0"]\n\n'
        '1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 1-0\n'
    )
    is_valid, msg, game = validate_pgn_input(valid_pgn)
    assert is_valid is True
    assert game is not None
    assert game.headers["White"] == "Player 1"


def test_validate_pgn_input_empty():
    """Verify empty or whitespace-only PGN returns False."""
    is_valid, msg, game = validate_pgn_input("   \n\t  ")
    assert is_valid is False
    assert game is None
    assert "empty" in msg.lower()


def test_validate_pgn_input_no_moves():
    """Verify PGN headers without mainline moves are rejected gracefully."""
    no_moves_pgn = (
        '[Event "Empty Game"]\n'
        '[White "Alice"]\n'
        '[Black "Bob"]\n'
        '[Result "*"]\n\n'
        '*\n'
    )
    is_valid, msg, game = validate_pgn_input(no_moves_pgn)
    assert is_valid is False
    assert game is None
    assert "no mainline moves" in msg.lower()


def test_save_uploaded_pgn_to_temp():
    """Verify saving PGN content to a temp file and cleaning it up."""
    content = '[Event "Temp"]\n\n1. d4 d5 *'
    temp_path = save_uploaded_pgn_to_temp(content)
    try:
        assert os.path.isfile(temp_path)
        with open(temp_path, "r", encoding="utf-8") as f:
            read_back = f.read()
        assert "1. d4 d5" in read_back
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)


def test_extract_eda_chart_data():
    """Verify extracting chart data from a mock prediction report with real schema."""
    mock_report = {
        "game_info": {
            "white": "Anderssen",
            "black": "Kieseritzky",
            "result": "1-0",
            "total_plies": 4,
            "depth": 10,
            "engine_used": "Stockfish",
        },
        "white_features": {
            "mean_centipawn_loss": 35.0,
            "std_centipawn_loss": 20.0,
            "median_centipawn_loss": 30.0,
            "mean_norm_cp_loss": 0.08,
            "std_norm_cp_loss": 0.05,
            "blunder_rate": 0.05,
            "mistake_rate": 0.10,
            "inaccuracy_rate": 0.15,
            "excellent_rate": 0.60,
            "top1_match_rate": 0.65,
            "used_mean_cp_loss": 30.0,
            "used_blunder_rate": 0.05,
            "move_count": 20,
        },
        "black_features": {
            "mean_centipawn_loss": 45.0,
            "std_centipawn_loss": 25.0,
            "median_centipawn_loss": 40.0,
            "mean_norm_cp_loss": 0.12,
            "std_norm_cp_loss": 0.07,
            "blunder_rate": 0.10,
            "mistake_rate": 0.15,
            "inaccuracy_rate": 0.20,
            "excellent_rate": 0.50,
            "top1_match_rate": 0.50,
            "used_mean_cp_loss": 40.0,
            "used_blunder_rate": 0.10,
            "move_count": 20,
        },
        "move_analysis": [
            {"ply_number": 1, "player_color": "White", "centipawn_loss": 10.0, "san": "e4"},
            {"ply_number": 2, "player_color": "Black", "centipawn_loss": 15.0, "san": "e5"},
            {"ply_number": 3, "player_color": "White", "centipawn_loss": 5.0, "san": "f4"},
            {"ply_number": 4, "player_color": "Black", "centipawn_loss": 210.0, "san": "exf4"},
        ],
    }

    res = extract_eda_chart_data(mock_report)
    assert "df_moves" in res
    assert "df_comparison" in res
    assert "df_quality" in res
    df_moves = res["df_moves"]
    assert len(df_moves) == 4
    assert len(res["df_comparison"]) == len(ML_FEATURE_COLUMNS)
    assert "Blunder (>200 cp)" in res["df_quality"]["Category"].values

    # Verify both canonical column and alias are accessible
    assert "player_color" in df_moves.columns
    assert "player" in df_moves.columns
    assert "ply_number" in df_moves.columns
    assert "ply" in df_moves.columns

    # Verify White and Black filtering works without KeyError
    w_moves = df_moves[df_moves["player_color"] == "White"]
    b_moves = df_moves[df_moves["player_color"] == "Black"]
    assert len(w_moves) == 2
    assert len(b_moves) == 2


def test_chart_data_handles_empty_moves_gracefully():
    """Verify that extract_eda_chart_data handles empty move_analysis without errors."""
    empty_report = {
        "game_info": {"white": "W", "black": "B", "result": "*", "total_plies": 0},
        "white_features": {},
        "black_features": {},
        "move_analysis": [],
    }
    res = extract_eda_chart_data(empty_report)
    assert res["df_moves"].empty
    assert not res["df_comparison"].empty


def test_real_prediction_move_analysis_schema():
    """Regression test: verify predict_game move_analysis schema directly supports EDA charts."""
    if not os.path.isfile("C:\\stockfish\\stockfish.exe"):
        pytest.skip("Stockfish not available at C:\\stockfish\\stockfish.exe")
    if not os.path.isfile("models/chess_fraud_rf.joblib"):
        pytest.skip("Model not found at models/chess_fraud_rf.joblib")

    from chess_eda.ml.predict import predict_game

    report = predict_game("data/sample.pgn", depth=6, verbose=False)
    assert "move_analysis" in report
    assert len(report["move_analysis"]) > 0

    res = extract_eda_chart_data(report)
    df_moves = res["df_moves"]
    assert not df_moves.empty

    # Confirm player_color, ply_number, and centipawn_loss exist
    assert "player_color" in df_moves.columns
    assert "ply_number" in df_moves.columns
    assert "centipawn_loss" in df_moves.columns

    # Test separation of White and Black moves exactly as in app.py
    w_moves = df_moves[df_moves["player_color"] == "White"]
    b_moves = df_moves[df_moves["player_color"] == "Black"]
    assert len(w_moves) > 0
    assert len(b_moves) > 0

    # Ensure centipawn_loss is numeric and can be plotted
    w_cp = w_moves["centipawn_loss"].dropna()
    b_cp = b_moves["centipawn_loss"].dropna()
    assert len(w_cp) > 0
    assert len(b_cp) > 0
    assert float(w_cp.iloc[0]) >= 0.0



def test_default_stockfish_path_constant():
    """Verify that DEFAULT_STOCKFISH_PATH is exported and defaults to C:\\stockfish\\stockfish.exe."""
    from chess_eda.ml.predict import DEFAULT_STOCKFISH_PATH as ML_SF_PATH
    from chess_eda.dashboard import DEFAULT_STOCKFISH_PATH as DASH_SF_PATH

    assert ML_SF_PATH.endswith("stockfish.exe")
    assert DASH_SF_PATH.endswith("stockfish.exe")
    assert ML_SF_PATH == DASH_SF_PATH


def test_app_py_compiles_and_imports():
    """Verify that app.py compiles and imports without any ImportError."""
    import importlib
    app_path = "app.py"
    assert os.path.isfile(app_path)
    compiled = py_compile.compile(app_path, doraise=True)
    assert compiled is not None

    # Dynamically import app module to verify all imports resolve
    app_mod = importlib.import_module("app")
    assert app_mod is not None


def test_audited_thegreatimu_game_reproducibility():
    """Verify that prediction on the audited TheGreatIMU game is fully reproducible, valid, and consistent."""
    if not os.path.isfile("C:\\stockfish\\stockfish.exe"):
        pytest.skip("Stockfish not available at C:\\stockfish\\stockfish.exe")

    from chess_eda.ml.predict import predict_game

    audited_path = os.path.join("data", "raw", "unlabeled", "thegreatimu_game.pgn")
    assert os.path.isfile(audited_path)

    # 1. Random forest check: valid prediction ranges, probability summation, schema
    rf_report1 = predict_game(audited_path, model_path="models/chess_fraud_rf.joblib", depth=10, verbose=False)
    rf_report2 = predict_game(audited_path, model_path="models/chess_fraud_rf.joblib", depth=10, verbose=False)

    b_rf1 = rf_report1["black_player"]["engine_assisted_likelihood"]
    b_rf2 = rf_report2["black_player"]["engine_assisted_likelihood"]
    # Verify deterministic reproducibility
    assert b_rf1 == b_rf2
    # Verify valid probability range
    assert 0.0 <= b_rf1 <= 1.0
    assert pytest.approx(b_rf1 + rf_report1["black_player"]["human_like_likelihood"], abs=1e-3) == 1.0
    # Verify uncalibrated score aliases
    assert rf_report1["black_player"]["engine_assistance_score"] == b_rf1
    assert rf_report1["black_player"]["complementary_human_like_score"] == rf_report1["black_player"]["human_like_likelihood"]
    # Verify regression score remains unchanged: ~51.3%
    assert pytest.approx(b_rf1, abs=0.005) == 0.513

    # 2. Logistic regression check: valid prediction ranges, probability summation, schema
    lr_report1 = predict_game(audited_path, model_path="models/chess_fraud_lr.joblib", depth=10, verbose=False)
    b_lr1 = lr_report1["black_player"]["engine_assisted_likelihood"]
    assert 0.0 <= b_lr1 <= 1.0
    assert pytest.approx(b_lr1 + lr_report1["black_player"]["human_like_likelihood"], abs=1e-3) == 1.0
    # Verify uncalibrated score aliases
    assert lr_report1["black_player"]["engine_assistance_score"] == b_lr1
    assert lr_report1["black_player"]["complementary_human_like_score"] == lr_report1["black_player"]["human_like_likelihood"]
    # Verify regression score remains unchanged: ~56.9%
    assert pytest.approx(b_lr1, abs=0.005) == 0.569

    # 3. Verify feature schema integrity across all 13 ML features
    for col in ML_FEATURE_COLUMNS:
        assert col in rf_report1["black_features"]
        assert col in rf_report1["white_features"]
        assert not pd.isna(rf_report1["black_features"][col])

    # 4. Verify game structure: 22 full moves (44 plies)
    assert rf_report1["game_info"]["white"] == "Engaliawad"
    assert rf_report1["game_info"]["black"] == "TheGreatIMU"
    assert rf_report1["black_features"]["move_count"] == 22


def test_player_card_presentation_data_case_a_engine_assisted():
    """Verify Case A: ENGINE-ASSISTED classification emphasizes Engine-assistance score."""
    from chess_eda.dashboard import get_player_card_presentation_data

    mock_assisted_player = {
        "name": "CheaterX",
        "classification": "engine-assisted",
        "engine_assisted_likelihood": 0.724,
        "human_like_likelihood": 0.276,
        "engine_assistance_score": 0.724,
        "human_like_score": 0.276,
    }

    pres = get_player_card_presentation_data(mock_assisted_player)

    assert pres["is_assisted"] is True
    assert pres["badge_text"] == "ENGINE-ASSISTED"
    assert pres["badge_class"] == "badge-assisted"
    assert pres["primary_label"] == "Engine-assistance score"
    assert pres["primary_pct"] == pytest.approx(72.4, abs=0.01)
    assert pres["primary_prob"] == pytest.approx(0.724, abs=0.001)
    assert pres["primary_color_class"] == "score-assisted"

    # Secondary score remains available
    assert pres["secondary_label"] == "Human-like score"
    assert pres["secondary_pct"] == pytest.approx(27.6, abs=0.01)
    assert pres["secondary_prob"] == pytest.approx(0.276, abs=0.001)


def test_player_card_presentation_data_case_b_human_like():
    """Verify Case B: HUMAN-LIKE classification emphasizes Human-like score."""
    from chess_eda.dashboard import get_player_card_presentation_data

    mock_human_player = {
        "name": "FairMaster",
        "classification": "human-like",
        "engine_assisted_likelihood": 0.142,
        "human_like_likelihood": 0.858,
        "engine_assistance_score": 0.142,
        "human_like_score": 0.858,
    }

    pres = get_player_card_presentation_data(mock_human_player)

    assert pres["is_assisted"] is False
    assert pres["badge_text"] == "HUMAN-LIKE"
    assert pres["badge_class"] == "badge-human"
    assert pres["primary_label"] == "Human-like score"
    assert pres["primary_pct"] == pytest.approx(85.8, abs=0.01)
    assert pres["primary_prob"] == pytest.approx(0.858, abs=0.001)
    assert pres["primary_color_class"] == "score-human"

    # Secondary score remains available
    assert pres["secondary_label"] == "Engine-assistance score"
    assert pres["secondary_pct"] == pytest.approx(14.2, abs=0.01)
    assert pres["secondary_prob"] == pytest.approx(0.142, abs=0.001)


def test_thegreatimu_game_ui_card_emphasis():
    """Verify end-to-end card emphasis on the audited TheGreatIMU game under both models."""
    if not os.path.isfile("C:\\stockfish\\stockfish.exe"):
        pytest.skip("Stockfish not available at C:\\stockfish\\stockfish.exe")

    from chess_eda.ml.predict import predict_game
    from chess_eda.dashboard import get_player_card_presentation_data

    audited_path = os.path.join("data", "raw", "unlabeled", "thegreatimu_game.pgn")

    # 1. Logistic Regression (Default Model)
    lr_report = predict_game(audited_path, model_path="models/chess_fraud_lr.joblib", depth=10, verbose=False)
    w_lr_pres = get_player_card_presentation_data(lr_report["white_player"])
    b_lr_pres = get_player_card_presentation_data(lr_report["black_player"])

    # White is human-like (~89.3% human-like)
    assert w_lr_pres["badge_text"] == "HUMAN-LIKE"
    assert w_lr_pres["primary_label"] == "Human-like score"
    assert pytest.approx(w_lr_pres["primary_pct"], abs=0.5) == 89.3
    assert w_lr_pres["secondary_label"] == "Engine-assistance score"
    assert pytest.approx(w_lr_pres["secondary_pct"], abs=0.5) == 10.7

    # Black is engine-assisted (~56.9% engine-assisted)
    assert b_lr_pres["badge_text"] == "ENGINE-ASSISTED"
    assert b_lr_pres["primary_label"] == "Engine-assistance score"
    assert pytest.approx(b_lr_pres["primary_pct"], abs=0.5) == 56.9
    assert b_lr_pres["secondary_label"] == "Human-like score"
    assert pytest.approx(b_lr_pres["secondary_pct"], abs=0.5) == 43.1

    # 2. Random Forest (Alternative Model)
    rf_report = predict_game(audited_path, model_path="models/chess_fraud_rf.joblib", depth=10, verbose=False)
    w_rf_pres = get_player_card_presentation_data(rf_report["white_player"])
    b_rf_pres = get_player_card_presentation_data(rf_report["black_player"])

    # White is human-like (~78.6% human-like)
    assert w_rf_pres["badge_text"] == "HUMAN-LIKE"
    assert w_rf_pres["primary_label"] == "Human-like score"
    assert pytest.approx(w_rf_pres["primary_pct"], abs=0.5) == 78.6

    # Black is engine-assisted (~51.3% engine-assisted)
    assert b_rf_pres["badge_text"] == "ENGINE-ASSISTED"
    assert b_rf_pres["primary_label"] == "Engine-assistance score"
    assert pytest.approx(b_rf_pres["primary_pct"], abs=0.5) == 51.3



