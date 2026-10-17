"""
main.py — EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection

Supports two modes:

VERSION 0 (default) — PGN Parsing and Move Data Extraction
    python main.py data/sample.pgn
    python main.py data/sample.pgn --output output/sample_moves.csv

VERSION 1 — Engine Analysis (requires Stockfish)
    python main.py data/sample.pgn --engine-analysis \\
        --stockfish-path "C:\\path\\to\\stockfish.exe" \\
        --depth 15

Version 0 limitation
--------------------
Only the first game in the PGN file is processed in both versions.
This is intentional; multi-game support is planned for a later version.

Version 1 notes
---------------
Engine analysis adds nine additional columns to the output CSV.
The engine-enriched CSV is written to a separate file (default:
output/<stem>_engine_analysis.csv) and does NOT overwrite the Version 0
CSV unless you explicitly set --output to the same path.

Stockfish must be obtained separately from https://stockfishchess.org/download/
The program cannot and does not download Stockfish automatically.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (used by both V0 and V1)
# ─────────────────────────────────────────────────────────────────────────────

def _check_output_path(output_path: str) -> None:
    """Validate that the output directory exists and is writable.

    Raises
    ------
    SystemExit
        Prints an error message and exits with code 1.
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(output_dir):
        print(
            f"[ERROR] Output directory does not exist: '{output_dir}'\n"
            "Please create it first, or choose a different output path.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.access(output_dir, os.W_OK):
        print(
            f"[ERROR] Cannot write to output directory: '{output_dir}'\n"
            "Check folder permissions.",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_csv(rows: list[dict], output_path: str, columns: list[str]) -> None:
    """Write *rows* to a CSV file at *output_path* atomically.

    The file is written to a temp name first, then renamed, so a partial
    file is never left behind on error.

    Raises
    ------
    SystemExit
        Prints an error message and exits with code 1 if writing fails.
    """
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, output_path)
    except OSError as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"[ERROR] Failed to write CSV: {exc}", file=sys.stderr)
        sys.exit(1)


def _derive_engine_output_path(v0_output: str) -> str:
    """Derive the default engine-analysis CSV path from the V0 output path.

    Example: ``output/moves.csv`` → ``output/moves_engine_analysis.csv``
    """
    base, ext = os.path.splitext(v0_output)
    return f"{base}_engine_analysis{ext or '.csv'}"


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "EDA-Driven Behavioral Feature Engineering for Human-AI Chess Move Detection\n"
            "Supports Version 0 (move extraction) and Version 1 (engine analysis).\n\n"
            "VERSION 0 (default):\n"
            "  Parses a PGN file and exports one CSV row per mainline move.\n\n"
            "VERSION 1 (--engine-analysis):\n"
            "  Runs Stockfish on every position and adds 9 evaluation columns.\n"
            "  Requires Stockfish: https://stockfishchess.org/download/\n\n"
            "VERSION 2 (--batch):\n"
            "  Runs multi-game ingestion across raw directories and exports\n"
            "  move-level and player-game level feature datasets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py data/sample.pgn\n"
            "  python main.py data/sample.pgn --output output/moves.csv\n"
            "  python main.py data/sample.pgn --engine-analysis \\\n"
            "      --stockfish-path \"C:\\stockfish\\stockfish.exe\" --depth 15\n"
            "  python main.py --batch --depth 10 \\\n"
            "      --stockfish-path \"C:\\stockfish\\stockfish.exe\"\n\n"
            "Note: Default mode processes only the first game in PGN_FILE."
        ),
    )

    # ── Positional ──
    parser.add_argument(
        "pgn_file",
        metavar="PGN_FILE",
        nargs="?",
        default=None,
        help="Path to the input PGN file (required unless --batch is used).",
    )

    # ── Version 0 output ──
    parser.add_argument(
        "--output",
        metavar="CSV_FILE",
        default="output/moves.csv",
        help=(
            "Path for the Version 0 CSV output file. "
            "Default: output/moves.csv"
        ),
    )

    # ── Version 1 flags ──
    v1 = parser.add_argument_group("Version 1 - Engine analysis (optional)")
    v1.add_argument(
        "--engine-analysis",
        action="store_true",
        default=False,
        help=(
            "Enable Stockfish engine analysis. Adds 9 evaluation columns to a "
            "separate CSV. Requires --stockfish-path."
        ),
    )
    v1.add_argument(
        "--stockfish-path",
        metavar="PATH",
        default="",
        help=(
            "Full path to the Stockfish executable. "
            'Example: --stockfish-path "C:\\stockfish\\stockfish.exe"'
        ),
    )
    v1.add_argument(
        "--depth",
        metavar="N",
        type=int,
        default=15,
        help=(
            "Stockfish search depth per position (default: 15). "
            "Higher values are more accurate but slower."
        ),
    )
    v1.add_argument(
        "--engine-output",
        metavar="CSV_FILE",
        default="",
        help=(
            "Path for the engine-analysis CSV. "
            "Default: derived from --output by inserting '_engine_analysis'."
        ),
    )

    # ── Version 2 flags ──
    v2 = parser.add_argument_group("Version 2 - Multi-game batch pipeline")
    v2.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Run batch pipeline across raw PGN directories.",
    )
    v2.add_argument(
        "--raw-dir",
        metavar="DIR",
        default="data/raw",
        help="Root directory for raw PGN subdirectories (default: data/raw).",
    )
    v2.add_argument(
        "--processed-dir",
        metavar="DIR",
        default="data/processed",
        help="Output directory for processed CSV datasets (default: data/processed).",
    )

    # ── Version 3 flags ──
    v3 = parser.add_argument_group("Version 3 - Machine learning baseline & prediction")
    v3.add_argument(
        "--train-model",
        action="store_true",
        default=False,
        help="Train baseline Logistic Regression and Random Forest models on ChessFraud.",
    )
    v3.add_argument(
        "--predict",
        metavar="PGN_FILE",
        default="",
        help="Predict engine assistance likelihood for a PGN game.",
    )
    v3.add_argument(
        "--model-type",
        choices=["rf", "lr"],
        default="rf",
        help="Model architecture to use for prediction: rf (Random Forest) or lr (Logistic Regression) (default: rf).",
    )
    v3.add_argument(
        "--model-path",
        metavar="PATH",
        default="",
        help="Path to fitted model pipeline (defaults to models/chess_fraud_<type>.joblib).",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the exit code."""

    # Import chess here so a missing package gives a clean error.
    try:
        import chess          # noqa: F401
        import chess.pgn      # noqa: F401
    except ImportError:
        print(
            "[ERROR] The 'chess' package is not installed.\n"
            "Install it with:  python -m pip install chess",
            file=sys.stderr,
        )
        return 1

    from chess_eda.pgn_parser import load_first_game, PGNParseError
    from chess_eda.feature_extraction import extract_move_rows, CSV_COLUMNS

    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Version 3 Model Training ──
    if args.train_model:
        from chess_eda.ml.train import train_baseline_models
        try:
            train_baseline_models(output_dir="models", verbose=True)
            return 0
        except Exception as exc:
            print(f"[ERROR] Model training failed: {exc}", file=sys.stderr)
            return 1

    # ── Version 3 Prediction Workflow ──
    if args.predict:
        from chess_eda.ml.predict import predict_game
        sf_path = args.stockfish_path or os.environ.get("STOCKFISH_PATH", "")
        if not sf_path:
            print(
                "[ERROR] Prediction requires Stockfish to evaluate moves.\n"
                'Please specify --stockfish-path "C:\\stockfish\\stockfish.exe" or set STOCKFISH_PATH.',
                file=sys.stderr,
            )
            return 1

        chosen_model = args.model_path
        if not chosen_model:
            chosen_model = os.path.join("models", f"chess_fraud_{args.model_type}.joblib")

        try:
            predict_game(
                pgn_path=args.predict,
                stockfish_path=sf_path,
                model_path=chosen_model,
                depth=args.depth if args.depth != 15 else 10,
                verbose=True,
            )
            return 0
        except Exception as exc:
            print(f"[ERROR] Prediction failed: {exc}", file=sys.stderr)
            return 1

    # ── Version 2 Batch Pipeline ──
    if args.batch:
        from chess_eda.pipeline import run_batch_pipeline
        sf_path = args.stockfish_path or os.environ.get("STOCKFISH_PATH", "")
        # In batch mode, depth defaults to args.depth (or 10 if depth is default 15 and not explicitly overridden)
        batch_depth = args.depth if args.depth != 15 else 10
        try:
            run_batch_pipeline(
                raw_dir=args.raw_dir,
                output_dir=args.processed_dir,
                stockfish_path=sf_path,
                depth=batch_depth,
                enable_engine=bool(sf_path),
                verbose=True,
            )
            return 0
        except Exception as exc:
            print(f"[ERROR] Batch pipeline failed: {exc}", file=sys.stderr)
            return 1

    pgn_path    = args.pgn_file
    if not pgn_path:
        parser.error("the following arguments are required: PGN_FILE (or pass --batch, --train-model, --predict)")

    output_path = args.output
    do_engine   = args.engine_analysis

    # ── Validate Version 0 output path ──
    _check_output_path(output_path)

    # ── Parse PGN ──
    try:
        game = load_first_game(pgn_path)
    except PGNParseError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # ── Version 0: extract base features ──
    rows = extract_move_rows(game, game_id=1)

    if not rows:  # pragma: no cover
        print(
            "[ERROR] Feature extraction produced no rows. "
            "The game appears to contain no mainline moves.",
            file=sys.stderr,
        )
        return 1

    # ── Write Version 0 CSV ──
    _write_csv(rows, output_path, CSV_COLUMNS)

    headers = game.headers
    white  = headers.get("White", "?")
    black  = headers.get("Black", "?")
    result = headers.get("Result", "?")

    print(
        "\nPGN ANALYSIS - VERSION 0\n"
        f"\nInput file    : {pgn_path}"
        f"\nWhite         : {white}"
        f"\nBlack         : {black}"
        f"\nResult        : {result}"
        f"\nMainline moves: {len(rows)}"
        f"\nCSV exported  : {output_path}\n"
    )

    # ── Version 1: engine analysis (optional) ──
    if do_engine:
        return _run_engine_analysis(args, game, rows, pgn_path, output_path)

    return 0


def _run_engine_analysis(
    args: argparse.Namespace,
    game,
    base_rows: list[dict],
    pgn_path: str,
    output_path: str,
) -> int:
    """Run Stockfish analysis and write the engine-enriched CSV."""
    from chess_eda.engine_analysis import (
        analyse_game,
        EngineAnalysisError,
        ENGINE_COLUMNS,
    )
    from chess_eda.feature_extraction import CSV_COLUMNS

    ENGINE_CSV_COLUMNS = CSV_COLUMNS + ENGINE_COLUMNS

    # Determine engine output path.
    engine_output = args.engine_output or _derive_engine_output_path(output_path)
    _check_output_path(engine_output)

    stockfish_path = args.stockfish_path or os.environ.get("STOCKFISH_PATH", "")
    depth = args.depth

    if not stockfish_path:
        print(
            "[ERROR] --engine-analysis requires --stockfish-path.\n"
            'Example: --stockfish-path "C:\\stockfish\\stockfish.exe"\n'
            "Download Stockfish from https://stockfishchess.org/download/",
            file=sys.stderr,
        )
        return 1

    print("ENGINE ANALYSIS - VERSION 1\n")
    print(f"  Stockfish   : {stockfish_path}")
    print(f"  Depth       : {depth}")
    print(f"  Positions   : {len(base_rows)}")
    print(f"  Output      : {engine_output}")
    print()
    print("  Analysing positions...", flush=True)

    try:
        engine_rows, metadata = analyse_game(
            game,
            stockfish_path=stockfish_path,
            depth=depth,
        )
    except EngineAnalysisError as exc:
        print(f"[ERROR] Engine analysis failed:\n  {exc}", file=sys.stderr)
        return 1

    # Merge base rows with engine rows (one-to-one by ply index).
    merged: list[dict] = []
    for base, eng in zip(base_rows, engine_rows):
        combined = {**base, **eng}
        merged.append(combined)

    _write_csv(merged, engine_output, ENGINE_CSV_COLUMNS)

    print(
        f"  Engine      : {metadata['engine_name']}\n"
        f"  CSV exported: {engine_output}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
