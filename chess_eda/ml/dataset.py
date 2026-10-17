"""
chess_eda.ml.dataset
====================
Dataset loading, verification, and provenance auditing for the ChessFraud benchmark.

Reference:
----------
* Source: https://huggingface.co/datasets/artemlepin/chess-fraud
* Paper: Linich et al. (2026), "ChessFraud: Exploring the Capabilities of Human-Aligned
  Models for Cheating Detection in Online Chess", ACM SIGKDD 2026.
* License: Creative Commons Attribution 4.0 International (CC BY 4.0).
* Configuration: `chess_fraud` (Controlled-Tournament Benchmark).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

import pandas as pd


CHESS_FRAUD_HF_URL = (
    "https://huggingface.co/datasets/artemlepin/chess-fraud/resolve/main/data/chess_fraud/full.parquet"
)
DEFAULT_DATA_PATH = os.path.join("data", "benchmarks", "chess_fraud", "full.parquet")
DEFAULT_METADATA_PATH = os.path.join("data", "benchmarks", "chess_fraud", "dataset_metadata.json")


def ensure_chess_fraud_dataset(data_path: str = DEFAULT_DATA_PATH, url: str = CHESS_FRAUD_HF_URL) -> str:
    """Ensure the ChessFraud controlled-tournament dataset is available locally.

    If not found at *data_path*, downloads it from Hugging Face.

    Parameters
    ----------
    data_path:
        Target local file path.
    url:
        Remote download URL.

    Returns
    -------
    str
        Absolute path to the downloaded parquet file.
    """
    abs_path = os.path.abspath(data_path)
    if os.path.isfile(abs_path) and os.path.getsize(abs_path) > 0:
        return abs_path

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    print(f"Downloading ChessFraud benchmark dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, abs_path)
    except Exception as exc:
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to download ChessFraud dataset: {exc}") from exc

    return abs_path


def load_chess_fraud_dataset(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the ChessFraud controlled benchmark parquet file into a pandas DataFrame.

    Parameters
    ----------
    data_path:
        Path to the local parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame with 38,510 rows and 33 columns.
    """
    if not os.path.exists(data_path):
        data_path = ensure_chess_fraud_dataset(data_path)

    df = pd.read_parquet(data_path)
    return df


def inspect_dataset_metadata(
    df: pd.DataFrame,
    save_path: str = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Inspect schema, missingness, class balance, and provenance of the ChessFraud dataset.

    Parameters
    ----------
    df:
        The loaded ChessFraud DataFrame.
    save_path:
        Path to save the JSON metadata summary.

    Returns
    -------
    dict[str, Any]
        Dictionary with verified metadata metrics.
    """
    # Move-level class distribution
    move_cheat_counts = df["is_cheating_move"].value_counts().to_dict()
    total_moves = len(df)
    cheat_moves = int(move_cheat_counts.get(True, 0))

    # Player-game level class distribution
    game_first = df.groupby("game_id").first()
    game_cheat_counts = game_first["is_cheating_player_game"].value_counts().to_dict()
    total_player_games = len(game_first)
    cheat_player_games = int(game_cheat_counts.get(True, 0))
    fair_player_games = int(game_cheat_counts.get(False, 0))

    # Physical match count
    physical_matches = df["game_id"].apply(
        lambda x: str(x)[:-2] if str(x).endswith(("_w", "_b")) else str(x)
    )
    unique_physical_matches = int(physical_matches.nunique())

    # Missing values
    missing_counts = {col: int(count) for col, count in df.isnull().sum().items() if count > 0}

    metadata: dict[str, Any] = {
        "dataset_name": "ChessFraud",
        "configuration": "chess_fraud (Controlled-Tournament Benchmark)",
        "source_url": "https://huggingface.co/datasets/artemlepin/chess-fraud",
        "citation": "Linich et al. (2026), ChessFraud: Exploring the Capabilities of Human-Aligned Models for Cheating Detection in Online Chess, ACM SIGKDD 2026",
        "license": "CC BY 4.0",
        "scope": "Controlled blitz tournament (5+0 time control) with logged Stockfish 15 assistance protocol",
        "separation_from_synthetic": "Controlled-tournament data is strictly separated from synthetic move substitutions (chess_fraud_synth)",
        "limitations": [
            "Covers a controlled cohort of 49 participants playing 5-minute blitz games.",
            "Labels reflect logged Stockfish 15 hints; other assistance modalities are outside scope.",
            "Not generalizable to population-wide online chess or classical time controls without caution.",
            "Statistical estimates from this benchmark must not be used as definitive proof of real-world cheating."
        ],
        "summary": {
            "total_move_rows": total_moves,
            "total_player_games": total_player_games,
            "unique_source_games": unique_physical_matches,
            "unique_physical_matches": unique_physical_matches,
            "unique_players": int(df["player_id"].nunique()),
            "used_decision_points": int(df["is_used"].sum()),
            "move_level_class_balance": {
                "fair_moves": total_moves - cheat_moves,
                "cheating_moves": cheat_moves,
                "cheating_move_rate": round(cheat_moves / total_moves, 4),
            },
            "player_game_class_balance": {
                "fair_games": fair_player_games,
                "cheating_games": cheat_player_games,
                "cheating_game_rate": round(cheat_player_games / total_player_games, 4),
                "majority_class_baseline_accuracy": round(fair_player_games / total_player_games, 4),
            },
            "columns_count": len(df.columns),
            "columns": list(df.columns),
            "missing_values_per_column": missing_counts,
        }
    }

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    return metadata
