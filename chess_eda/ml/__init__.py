"""
chess_eda.ml
============
Version 3: Supervised Machine Learning Baseline for Engine-Assistance Detection.
Provides dataset management, feature engineering, group-aware training,
and inference pipelines based on the public ChessFraud benchmark.
"""

from chess_eda.ml.dataset import load_chess_fraud_dataset, inspect_dataset_metadata
from chess_eda.ml.features import (
    ML_FEATURE_COLUMNS,
    V5_COMPLEXITY_FEATURE_COLUMNS,
    V5_ALL_FEATURE_COLUMNS,
    extract_player_features_from_df,
    extract_player_features_from_engine_rows,
)
from chess_eda.ml.train import train_baseline_models
from chess_eda.ml.predict import predict_game
from chess_eda.ml.v5_experiments import run_v5_complexity_experiment

__all__ = [
    "load_chess_fraud_dataset",
    "inspect_dataset_metadata",
    "ML_FEATURE_COLUMNS",
    "V5_COMPLEXITY_FEATURE_COLUMNS",
    "V5_ALL_FEATURE_COLUMNS",
    "extract_player_features_from_df",
    "extract_player_features_from_engine_rows",
    "train_baseline_models",
    "predict_game",
    "run_v5_complexity_experiment",
]
