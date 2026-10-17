"""
chess_eda.pipeline
==================
Batch processing pipeline for multi-game PGN ingestion, behavioral feature
extraction, Stockfish engine analysis, and player-game level aggregation.

Key Design Decisions:
---------------------
* Category-driven ingestion:
    data/raw/human_games/           -> source_category = "human_games"
    data/raw/engine_generated_games/ -> source_category = "engine_generated_games"
    data/raw/unlabeled/             -> source_category = "unlabeled"
* Non-fatal error handling:
    Corrupt games, illegal moves, or single-game engine crashes are recorded
    in the dataset manifest and skipped without corrupting valid games.
* Provenance tracking:
    Every row in both output datasets retains its source filename, game ID,
    and category.
* Atomic CSV writing:
    Files are written to temporary filenames first to prevent partial outputs.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sys
from typing import Any, Optional

import chess
import chess.engine

from chess_eda.aggregation import (
    aggregate_player_game_features,
    PLAYER_GAME_COLUMNS,
)
from chess_eda.engine_analysis import (
    analyse_game,
    validate_stockfish_path,
    EngineAnalysisError,
)
from chess_eda.move_features import (
    extract_move_features,
    enrich_moves_with_engine,
    MOVE_FEATURE_COLUMNS,
)
from chess_eda.pgn_parser import load_all_games, PGNParseError


def _write_csv_atomic(rows: list[dict[str, Any]], output_path: str, columns: list[str]) -> None:
    """Write rows to a CSV file atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def _audit_file_provenance(rel_path: str, category: str = "") -> dict[str, str]:
    """Return provenance audit classification and factual verification notes for a file."""
    norm_path = rel_path.lower().replace("\\", "/")
    if "immortal_game.pgn" in norm_path:
        return {
            "provenance_status": "verified_historical_human",
            "authenticity": "Independently verified",
            "historical_reference": "Adolf Anderssen vs. Lionel Kieseritzky (London, 21 June 1851). Extensively documented in chess literature and tournament records (e.g. Chessgames ID 1018910).",
        }
    if "opera_game_1858.pgn" in norm_path:
        return {
            "provenance_status": "verified_historical_human",
            "authenticity": "Independently verified",
            "historical_reference": "Paul Morphy vs. Duke Karl of Brunswick and Count Isouard (Paris, 2 November 1858). Widely documented in chess history (e.g. Chessgames ID 1233404).",
        }
    if "tcec_engine_game.pgn" in norm_path:
        return {
            "provenance_status": "illustrative_engine_demonstration_unverified",
            "authenticity": "Unverified demonstration data",
            "historical_reference": "Headers mention TCEC Season 20 Superfinal, but moves do not match the historical TCEC Season 20 Superfinal Game 1. Retained as an illustrative demonstration game for pipeline verification only.",
        }
    if "casual_game.pgn" in norm_path:
        return {
            "provenance_status": "synthetic_programmatic_demonstration",
            "authenticity": "Programmatically constructed test game",
            "historical_reference": "Created programmatically during development for pipeline testing and unrated move handling. Not a verified real-world game.",
        }
    return {
        "provenance_status": "unverified_user_input",
        "authenticity": "Unverified",
        "historical_reference": "User-supplied file. Provenance not independently audited.",
    }


def run_batch_pipeline(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    stockfish_path: str = "",
    depth: int = 10,
    enable_engine: bool = True,
    max_games_per_file: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute the multi-game data pipeline across raw PGN directories.

    Parameters
    ----------
    raw_dir:
        Root raw data directory containing category subdirectories.
    output_dir:
        Destination directory for processed CSV files and manifest.
    stockfish_path:
        Path to Stockfish executable (optional if enable_engine=False).
    depth:
        Engine search depth per move position.
    enable_engine:
        Whether to run Stockfish evaluations.
    max_games_per_file:
        Optional limit on the number of games processed per file.
    verbose:
        If True, prints progress messages to stdout.

    Returns
    -------
    dict[str, Any]
        The dataset manifest summarizing files, games, moves, and errors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Validate stockfish path if engine analysis requested
    sf_name = "None (Engine analysis disabled)"
    if enable_engine and stockfish_path:
        try:
            resolved_sf = validate_stockfish_path(stockfish_path)
            # Check engine identification
            try:
                test_engine = chess.engine.SimpleEngine.popen_uci(resolved_sf)
                sf_name = test_engine.id.get("name", "Stockfish")
                test_engine.quit()
            except Exception as exc:
                sf_name = f"Stockfish (probe error: {exc})"
        except EngineAnalysisError as exc:
            if verbose:
                print(f"[WARNING] Stockfish validation failed: {exc}", file=sys.stderr)
                print("Proceeding without engine evaluations for batch.", file=sys.stderr)
            enable_engine = False
            resolved_sf = ""
    else:
        enable_engine = False
        resolved_sf = ""

    # Known categories
    categories = ["human_games", "engine_generated_games", "unlabeled"]

    all_move_rows: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []

    global_game_id = 0
    total_files_scanned = 0

    if verbose:
        print("=" * 65)
        print("VERSION 2: MULTI-GAME DATA PREPARATION & FEATURE PIPELINE")
        print("=" * 65)
        print(f"Raw directory    : {raw_dir}")
        print(f"Output directory : {output_dir}")
        print(f"Engine enabled   : {enable_engine}")
        if enable_engine:
            print(f"Engine name      : {sf_name}")
            print(f"Stockfish path   : {resolved_sf}")
            print(f"Analysis depth   : {depth}")
        print("-" * 65)

    for category in categories:
        cat_dir = os.path.join(raw_dir, category)
        if not os.path.isdir(cat_dir):
            continue

        pgn_files = sorted([
            f for f in os.listdir(cat_dir)
            if f.lower().endswith(".pgn") and os.path.isfile(os.path.join(cat_dir, f))
        ])

        for pgn_file in pgn_files:
            total_files_scanned += 1
            full_path = os.path.join(cat_dir, pgn_file)
            rel_path = os.path.relpath(full_path, start=raw_dir).replace("\\", "/")

            if verbose:
                print(f"\nProcessing [{category}] {rel_path}...")

            try:
                games, skipped = load_all_games(full_path, skip_invalid=True)
            except PGNParseError as exc:
                err_msg = f"Failed to parse PGN file: {exc}"
                if verbose:
                    print(f"  [ERROR] {err_msg}")
                skipped_records.append({
                    "file": rel_path,
                    "category": category,
                    "error": err_msg,
                })
                continue

            for skip_info in skipped:
                skipped_records.append({
                    "file": rel_path,
                    "category": category,
                    "game_index": skip_info.get("game_index"),
                    "error": skip_info.get("error"),
                })

            if max_games_per_file:
                games = games[:max_games_per_file]

            valid_games_in_file = 0
            moves_in_file = 0

            for game_idx, game in enumerate(games, start=1):
                global_game_id += 1
                valid_games_in_file += 1

                white = game.headers.get("White", "Unknown")
                black = game.headers.get("Black", "Unknown")
                result = game.headers.get("Result", "*")

                # Extract move-level base & tactical features
                base_moves = extract_move_features(
                    game=game,
                    game_id=global_game_id,
                    source_file=rel_path,
                    source_category=category,
                )

                # Stockfish analysis per game (if enabled)
                engine_rows = None
                if enable_engine and resolved_sf:
                    if verbose:
                        print(f"  Analyzing Game #{global_game_id} ({white} vs {black}, {len(base_moves)} moves)...", end="", flush=True)
                    try:
                        engine_rows, _ = analyse_game(
                            game=game,
                            stockfish_path=resolved_sf,
                            depth=depth,
                        )
                        if verbose:
                            print(" done.")
                    except Exception as exc:
                        if verbose:
                            print(f" [ENGINE ERROR: {exc}]")
                        skipped_records.append({
                            "file": rel_path,
                            "category": category,
                            "game_id": global_game_id,
                            "error": f"Engine analysis error: {exc}",
                        })
                        engine_rows = None

                enriched_moves = enrich_moves_with_engine(base_moves, engine_rows)
                all_move_rows.extend(enriched_moves)
                moves_in_file += len(enriched_moves)

            file_provenance = _audit_file_provenance(rel_path, category)
            file_records.append({
                "file": rel_path,
                "category": category,
                "games_processed": valid_games_in_file,
                "moves_extracted": moves_in_file,
                "provenance_audit": file_provenance,
            })

    # Player-game level aggregation
    if verbose:
        print("\nAggregating player-game features...")

    player_game_rows = aggregate_player_game_features(all_move_rows)

    # Export paths
    move_csv_path = os.path.join(output_dir, "move_level_features.csv")
    player_csv_path = os.path.join(output_dir, "player_game_features.csv")
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")

    _write_csv_atomic(all_move_rows, move_csv_path, MOVE_FEATURE_COLUMNS)
    _write_csv_atomic(player_game_rows, player_csv_path, PLAYER_GAME_COLUMNS)

    # Build manifest
    category_summary: dict[str, dict[str, int]] = {}
    for cat in categories:
        cat_files = [f for f in file_records if f["category"] == cat]
        category_summary[cat] = {
            "files": len(cat_files),
            "games": sum(f["games_processed"] for f in cat_files),
            "moves": sum(f["moves_extracted"] for f in cat_files),
        }

    manifest: dict[str, Any] = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pipeline_version": "2.0.0",
        "engine_info": {
            "enabled": enable_engine,
            "engine_name": sf_name,
            "stockfish_path": resolved_sf,
            "depth": depth if enable_engine else None,
        },
        "summary": {
            "total_files_scanned": total_files_scanned,
            "total_games_processed": global_game_id,
            "total_moves_extracted": len(all_move_rows),
            "total_player_games": len(player_game_rows),
            "categories": category_summary,
        },
        "provenance_audit_summary": {
            "verified_historical_games": sum(1 for f in file_records if f.get("provenance_audit", {}).get("provenance_status") == "verified_historical_human"),
            "illustrative_unverified_engine_games": sum(1 for f in file_records if f.get("provenance_audit", {}).get("provenance_status") == "illustrative_engine_demonstration_unverified"),
            "synthetic_demonstration_games": sum(1 for f in file_records if f.get("provenance_audit", {}).get("provenance_status") == "synthetic_programmatic_demonstration"),
            "research_readiness": "Demonstration pipeline only. Not suitable for training machine learning models or claiming human/AI discrimination without ingesting a verified, balanced research corpus.",
        },
        "files_processed": file_records,
        "skipped_and_errors": skipped_records,
        "outputs": {
            "move_level_csv": os.path.abspath(move_csv_path).replace("\\", "/"),
            "player_game_csv": os.path.abspath(player_csv_path).replace("\\", "/"),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if verbose:
        print("\n" + "=" * 65)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 65)
        print(f"Total files processed : {total_files_scanned}")
        print(f"Total games processed : {global_game_id}")
        print(f"Total moves extracted : {len(all_move_rows)}")
        print(f"Player-game profiles  : {len(player_game_rows)}")
        print(f"Move CSV exported     : {move_csv_path}")
        print(f"Player CSV exported   : {player_csv_path}")
        print(f"Manifest exported     : {manifest_path}")
        print("=" * 65 + "\n")

    return manifest
