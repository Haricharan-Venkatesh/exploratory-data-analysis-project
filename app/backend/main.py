"""
FastAPI Backend Application for Human-AI Chess Move Detection
============================================================
Provides REST API endpoints for:
- /health: System health and model metadata
- /predict: Score individual 10-feature behavioral vectors
- /analyze-pgn: Full PGN ingestion and move-by-move behavioral analysis
- /demo-games: Bundled offline sample games
- /feature-importance: Model feature attribution benchmarks
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import json

from src.predict import (
    predict_player,
    analyze_pgn,
    get_model_metadata,
    get_feature_columns,
    DISCLAIMER_TEXT
)
from src.inference_features import FEATURE_COLUMNS

app = FastAPI(
    title="Human-AI Chess Move Detection API",
    description="EDA-driven behavioral classification API for human vs. AI chess play.",
    version="1.0.0"
)

# Enable CORS for local testing and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeatureVectorInput(BaseModel):
    consecutive_fast_moves: float = Field(0.0, description="Consecutive moves <= 0.5s")
    premove_rate_10: float = Field(0.0, description="Rolling pre-move rate over last 10 moves (0.0 to 1.0)")
    cumulative_premove_rate: float = Field(0.0, description="Cumulative pre-move rate across the game (0.0 to 1.0)")
    current_endgame_clock_ratio: float = Field(0.0, description="Clock remaining / base time if in Endgame, else 0.0")
    phase_deliberation_ratio: float = Field(1.0, description="Middlegame mean move time / Opening mean move time")
    move_time_cv_10: float = Field(0.5, description="Coefficient of variation of move time over last 10 moves")
    time_pressure_jitter: float = Field(0.0, description="Std dev of move time when clock < 15s")
    relative_move_time: float = Field(1.0, description="Move time / (base_time / 40.0)")
    time_spent_ratio: float = Field(0.05, description="Move time / available clock bank (0.0 to 1.0)")
    tank_move_count_10: float = Field(0.0, description="Count of moves > 30s in last 10 moves")
    threshold: Optional[float] = Field(None, description="Optional custom operating threshold (defaults to 0.50)")

class PGNInput(BaseModel):
    pgn: str = Field(..., description="Raw PGN string with move notations and [%clk] comments")
    threshold: Optional[float] = Field(None, description="Optional custom operating threshold (defaults to 0.50)")


@app.get("/health")
def health():
    """Returns server health, model metadata, and operating parameters."""
    metadata = get_model_metadata()
    feature_cols = get_feature_columns()
    return {
        "status": "healthy",
        "service": "Human-AI Chess Move Detection",
        "model_name": metadata.get("model_name"),
        "model_type": metadata.get("model_type"),
        "version": metadata.get("version"),
        "operating_threshold": metadata.get("operating_threshold", 0.50),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "test_roc_auc": metadata.get("test_metrics_unseen_game", {}).get("roc_auc")
    }


@app.post("/predict")
def predict_move(input_data: FeatureVectorInput):
    """
    Score a single 10-feature behavioral vector.
    """
    try:
        data = input_data.model_dump()
        thresh = data.pop("threshold", None)
        result = predict_player(data, threshold=thresh)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze-pgn")
def analyze_pgn_endpoint(input_data: PGNInput):
    """
    Parse a complete PGN and score each move sequentially without lookahead.
    """
    if not input_data.pgn or not input_data.pgn.strip():
        raise HTTPException(status_code=400, detail="Empty PGN content provided.")
    try:
        result = analyze_pgn(input_data.pgn, threshold=input_data.threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PGN analysis error: {str(e)}")


@app.post("/upload-pgn")
async def upload_pgn_file(file: UploadFile = File(...), threshold: Optional[float] = Form(None)):
    """
    Accepts PGN file upload and returns complete behavioral analysis.
    """
    try:
        content = await file.read()
        pgn_text = content.decode("utf-8", errors="replace")
        return analyze_pgn(pgn_text, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {str(e)}")


@app.get("/demo-games")
def get_demo_games():
    """
    Returns bundled demo games for offline demonstration.
    """
    demo_dir = os.path.join(PROJECT_ROOT, "data", "demo_games")
    games = []
    
    presets = [
        {
            "id": "human_match",
            "title": "Human Master Match",
            "subtitle": "Natural deliberation surges, variable pacing, deep think pauses",
            "file": "human_match.pgn",
            "expected_white": "HUMAN",
            "expected_black": "HUMAN"
        },
        {
            "id": "ai_bot_match",
            "title": "AI Engine Bot Match",
            "subtitle": "Consistent machine pacing, flat deliberation curve, zero hesitation",
            "file": "ai_bot_match.pgn",
            "expected_white": "AI",
            "expected_black": "AI"
        },
        {
            "id": "human_vs_bot_match",
            "title": "Human vs. Bot Challenge",
            "subtitle": "Direct side-by-side comparison under identical match conditions",
            "file": "human_vs_bot_match.pgn",
            "expected_white": "HUMAN",
            "expected_black": "AI"
        }
    ]
    
    for p in presets:
        fpath = os.path.join(demo_dir, p["file"])
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                pgn_str = f.read()
            games.append({
                "id": p["id"],
                "title": p["title"],
                "subtitle": p["subtitle"],
                "expected_white": p["expected_white"],
                "expected_black": p["expected_black"],
                "pgn": pgn_str
            })
            
    return {"demo_games": games}


@app.get("/feature-importance")
def get_feature_importance():
    """
    Returns top model feature importance rankings from empirical training results.
    """
    csv_path = os.path.join(PROJECT_ROOT, "results", "feature_importance.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {
            "features": df.to_dict(orient="records"),
            "disclaimer": "Feature importance reflects empirical predictive contribution in LightGBM, not psychological or cognitive proof of intent."
        }
    return {"features": [], "disclaimer": ""}


# Serve frontend static assets if available
frontend_dir = os.path.join(PROJECT_ROOT, "app", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
