r"""
batch_pipeline.py — Dedicated runner for Version 2 Batch Processing.

Usage:
    python batch_pipeline.py
    python batch_pipeline.py --stockfish-path "C:\stockfish\stockfish.exe" --depth 10
    python batch_pipeline.py --raw-dir data/raw --output-dir data/processed
"""

from __future__ import annotations

import argparse
import os
import sys

from chess_eda.pipeline import run_batch_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Version 2: Batch PGN Ingestion & Feature Engineering Pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Root directory containing category subdirectories (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for processed CSV files (default: data/processed).",
    )
    parser.add_argument(
        "--stockfish-path",
        default=os.environ.get("STOCKFISH_PATH", "C:\\stockfish\\stockfish.exe"),
        help="Path to Stockfish executable (default: env STOCKFISH_PATH or C:\\stockfish\\stockfish.exe).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Search depth per position (default: 10).",
    )
    parser.add_argument(
        "--no-engine",
        action="store_true",
        default=False,
        help="Disable Stockfish engine evaluations.",
    )
    args = parser.parse_args()

    enable_engine = not args.no_engine
    try:
        run_batch_pipeline(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            stockfish_path=args.stockfish_path if enable_engine else "",
            depth=args.depth,
            enable_engine=enable_engine,
            verbose=True,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] Batch pipeline execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
