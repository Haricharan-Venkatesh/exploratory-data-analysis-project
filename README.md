# EDA-Driven Behavioral Feature Engineering for Human-AI Chess Move Detection

[![Tests](https://img.shields.io/badge/tests-161%20passing-brightgreen)](#running-the-test-suite)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Project Purpose & Technologies

This project investigates whether behavioral patterns and decision-quality metrics in chess games can distinguish human play from computer-assisted play. The pipeline combines **Exploratory Data Analysis (EDA)** with **Supervised Machine Learning**, extracting board-level tactical metrics and Stockfish engine evaluations to build classification models and interactive visual dashboards.

### Core Technologies
- **Chess Logic & PGN Parsing:** [python-chess](https://python-chess.readthedocs.io/) (SAN/UCI notation, FEN states, board validation, UCI engine communication).
- **Chess Engine Evaluation:** [Stockfish](https://stockfishchess.org/) via standard UCI protocol (multi-PV analysis, score evaluation, centipawn loss).
- **Data Engineering & Analysis:** [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [pyarrow](https://arrow.apache.org/pyarrow/).
- **Machine Learning & Evaluation:** [scikit-learn](https://scikit-learn.org/), [joblib](https://joblib.readthedocs.io/) (Logistic Regression, Random Forest, Stratified Group Cross-Validation, ROC-AUC, PR-AUC, Brier score).
- **Interactive Web Dashboard:** [Streamlit](https://streamlit.io/) (Version 4 user-facing application).
- **Data Visualization:** [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/).
- **Testing & Quality Assurance:** [pytest](https://docs.pytest.org/) (161 automated unit and integration tests).

---

## Architectural Version Roadmap

- **Version 0:** Single-game PGN parser and move-level CSV feature exporter.
- **Version 1:** Stockfish UCI engine integration: centipawn loss calculation, move rankings, candidate move comparison.
- **Version 2:** Multi-game batch ingestion orchestrator, exploratory data analysis, and player-game behavioral aggregation.
- **Version 3:** Supervised machine learning baseline trained on the peer-reviewed **ChessFraud** benchmark dataset with strict physical-match-grouped cross-validation.
- **Version 4 (Current Main Dashboard):** Polished interactive Streamlit web application (`app.py`) providing PGN upload/paste, master presets, move trajectory EDA charts, and player-level engine-assistance similarity scores (Logistic Regression default, Random Forest alternative).
- **Version 5 (Experimental Research Branch):** Offline position-complexity-aware feature investigation (depth-drop rate, stability features, tactical criticality) retained in project files for continued research (`chess_eda/ml/v5_experiments.py`, `models/experimental/`, `reports/v5_position_complexity_findings.md`, `tests/test_v5_complexity.py`).

---

## Repository Structure

```
EDA_Project/
├── app.py                         ← Version 4 Streamlit interactive dashboard (current main UI)
├── main.py                        ← Unified CLI entry point for V0, V1, V2, and V3
├── batch_pipeline.py              ← Batch feature extraction runner (V2)
├── generate_notebook.py           ← Programmatic generator for the EDA Jupyter notebook
├── run_notebook.py                ← Headless Jupyter EDA notebook executor
├── requirements.txt               ← Python package dependencies
├── .env.example                   ← Template for configurable environment variables
├── .gitignore                     ← Git exclusion rules
├── LICENSE                        ← MIT License & ChessFraud CC BY 4.0 attribution
├── README.md                      ← Project documentation
├── chess_eda/                     ← Core Python package
│   ├── __init__.py                ← Package initialization
│   ├── pgn_parser.py              ← PGN parsing and validation
│   ├── feature_extraction.py      ← Base move feature extraction (V0)
│   ├── engine_analysis.py         ← Stockfish engine UCI wrapper (V1)
│   ├── move_features.py           ← Tactical and engine move-level features (V2)
│   ├── aggregation.py             ← Player-game behavioral aggregation (V2)
│   ├── pipeline.py                ← Batch ingestion orchestrator (V2)
│   ├── dashboard.py               ← Streamlit presentation and chart transformation helpers
│   └── ml/                        ← Supervised machine learning package (V3 & V5)
│       ├── __init__.py
│       ├── dataset.py             ← ChessFraud benchmark loader and provenance auditor
│       ├── features.py            ← ML feature alignment and extraction
│       ├── train.py               ← 5-fold physical-match-grouped training pipeline
│       ├── predict.py             ← End-to-end PGN inference pipeline
│       └── v5_experiments.py      ← Version 5 position-complexity experiment runner
├── data/
│   ├── sample.pgn                 ← Immortal Game (Anderssen vs Kieseritzky, 1851)
│   ├── benchmarks/
│   │   └── chess_fraud/           ← ChessFraud controlled-tournament benchmark (full.parquet)
│   ├── raw/                       ← Raw sample PGNs (human, computer, and online rapid)
│   └── processed/                 ← Engineered feature datasets and dataset manifest
├── models/                        ← Serialized production model pipelines (V3/V4)
│   ├── chess_fraud_lr.joblib      ← Trained Logistic Regression pipeline (Default)
│   ├── chess_fraud_rf.joblib      ← Trained Random Forest pipeline (Alternative)
│   ├── model_metadata.json        ← Grouped cross-validation evaluation metrics
│   └── experimental/              ← Version 5 experimental models and metadata
├── notebooks/
│   └── 02_eda_and_feature_engineering.ipynb ← Executable EDA Jupyter notebook
├── output/                        ← Directory for CLI-generated outputs (.gitkeep tracked)
├── reports/                       ← Detailed technical findings and audit reports
└── tests/                         ← Comprehensive automated test suite (161 tests)
```

---

## Installation & Setup Guide

### 1. Prerequisites
- **Python 3.10 to 3.13** installed on your system.
- **Git** (optional, for cloning).
- **Stockfish Chess Engine** (see Stockfish Setup below).

### 2. Virtual Environment Setup on Windows (PowerShell)

Open PowerShell and navigate to the project directory:

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt
```

> **Tip for Linux / macOS users:**
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> ```

---

## Stockfish Installation & Configuration

Stockfish is an external standalone executable (not a Python package). It is required for engine analysis, centipawn loss calculation, and live PGN inference.

### 1. Download Stockfish
1. Visit the official download page: https://stockfishchess.org/download/
2. Download the appropriate binary for your OS and CPU architecture.
3. Extract the archive and place the executable in a preferred directory, for example:
   - **Windows:** `C:\stockfish\stockfish.exe`
   - **Linux:** `/usr/games/stockfish` (or `sudo apt install stockfish`)
   - **macOS:** `/opt/homebrew/bin/stockfish` (or `brew install stockfish`)

### 2. Configure the Path
The project checks for Stockfish in the following order:
1. UI sidebar input field in the Streamlit dashboard.
2. The `STOCKFISH_PATH` environment variable.
3. Default path: `C:\stockfish\stockfish.exe`.

To set the environment variable persistently:
```powershell
# Windows PowerShell
$env:STOCKFISH_PATH = "C:\stockfish\stockfish.exe"

# Copy the example environment file if using a .env loader
Copy-Item .env.example .env
```

Verify your Stockfish installation:
```powershell
C:\stockfish\stockfish.exe bench
```

---

## Launching the Streamlit Dashboard (Version 4)

Version 4 is the current primary user-facing dashboard.

### Command

From PowerShell in the project directory with the virtual environment activated:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Or simply:
```powershell
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

### Dashboard Capabilities
1. **Game Ingestion:** Choose from verified preset master games, upload any custom `.pgn` file, or paste raw PGN text directly.
2. **Model Selection:**
   - **Logistic Regression (Default):** Selected as primary model due to higher discrimination (ROC-AUC 0.7559 vs 0.7437), higher recall (56.27% vs 54.79%), and superior probability calibration under 5-fold physical-match-grouped cross-validation.
   - **Random Forest (Alternative):** Available as a tree-ensemble alternative.
3. **Stockfish Controls:** Configurable search depth (default: 10 plies) with automatic binary verification.
4. **Classification-Dependent Score Emphasis:**
   - **ENGINE-ASSISTED:** Prominent amber badge, **38px bold** engine-assistance score, relevant progress bar, and secondary human-like score.
   - **HUMAN-LIKE:** Prominent green badge, **38px bold** human-like score, relevant progress bar, and secondary engine-assistance score.
5. **Key Behavioral Metrics:** Instant computation of Mean Centipawn Loss, Top-1 Engine Agreement, and Blunder Rate.
6. **Exploratory Data Analysis (EDA Charts):** Move quality breakdown, move-by-move centipawn trajectory, centipawn loss histogram, and complete 13-feature tabular comparison.
7. **Model Evaluation Tab:** Out-of-fold 5-fold grouped cross-validation comparison table, Brier score calibration, and feature importances.

---

## Running the Command-Line Pipelines

### 1. Single-Game Move Extraction (Version 0)
```powershell
.\.venv\Scripts\python.exe main.py data/sample.pgn --output output/moves.csv
```

### 2. Single-Game Engine Analysis (Version 1)
```powershell
.\.venv\Scripts\python.exe main.py data/sample.pgn `
    --engine-analysis `
    --stockfish-path "C:\stockfish\stockfish.exe" `
    --depth 10 `
    --engine-output output/sample_moves_engine_analysis.csv
```

### 3. Multi-Game Batch Processing (Version 2)
```powershell
.\.venv\Scripts\python.exe batch_pipeline.py --stockfish-path "C:\stockfish\stockfish.exe" --depth 10
```

### 4. Model Training & Cross-Validation (Version 3)
```powershell
.\.venv\Scripts\python.exe main.py --train-model
```

### 5. CLI Game Prediction (Version 3)
```powershell
.\.venv\Scripts\python.exe main.py `
    --predict data/raw/unlabeled/thegreatimu_game.pgn `
    --stockfish-path "C:\stockfish\stockfish.exe" `
    --depth 10 `
    --model-type lr
```

---

## Running the Test Suite

The project includes an automated test suite covering PGN parsing, Stockfish UCI communication, tactical feature extraction, grouped cross-validation, and dashboard UI logic.

Run the complete test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Run with verbose test reporting:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Dataset Attribution & Licensing

### Benchmark Dataset: ChessFraud
This project utilizes the **ChessFraud** controlled-tournament benchmark dataset for supervised model training and evaluation:
- **Paper:** Artem Linich, Daniil Oganesian, et al. (2026), *"ChessFraud: Exploring the Capabilities of Human-Aligned Models for Cheating Detection in Online Chess"*, ACM SIGKDD 2026 Datasets and Benchmarks Track.
- **Repository:** https://huggingface.co/datasets/artemlepin/chess-fraud
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Benchmark Composition:** 505 controlled blitz matches (1,010 player-game samples, 38,510 move rows) played by 49 participants under a logged Stockfish 15 hint protocol (40.3% assisted player-games, 59.7% fair player-games).

### Project Code License
The project codebase is licensed under the [MIT License](LICENSE).

---

## Scientific Disclaimers & Methodological Limitations

1. **Similarity Estimates, Not Definitive Proof:**
   These scores estimate behavioral similarity to engine-assisted play based on statistical concordance with Stockfish evaluations. They are **not verified real-world probabilities** and **cannot prove that a player used an engine or engaged in unethical conduct**. Results can differ between models and configurations.
2. **Operational Decision Boundary:**
   The 50% classification threshold is an operational software boundary for demonstration, not a scientifically established threshold for fair-play adjudication.
3. **Natural Engine Agreement:**
   Opening preparation, forced combinations, endgames, and simplified tactical exchanges naturally produce high engine agreement in human master play without computer assistance.
4. **Controlled Benchmark Scope:**
   The training benchmark reflects 5-minute blitz games played by ~2000 Elo players under a specific Stockfish 15 hint protocol. It does not generalize unconditionally to classical time controls, bullet games, or different rating cohorts.
5. **Version 5 Research Status:**
   Version 5 (position-complexity features) is an offline experimental research branch. It is preserved in the repository for ongoing investigation and is intentionally separated from the main user-facing dashboard.
