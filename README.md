# EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection

> **Research Question**: *"Do EDA-driven behavioral features provide useful predictive information for distinguishing human from AI chess play compared with simpler baselines?"*  
> **Empirical Answer**: **Yes.** The validated 10-feature behavioral model lifts ROC-AUC from **0.8312 to 0.9008** (+8.4% relative gain) and elevates PR-AUC from **0.2905 to 0.4780** (**+64.3% relative lift**) over conventional timing baselines on held-out unseen games, while preserving strong discrimination (**0.8639 ROC-AUC**) across unseen players.

---

## 1. System Overview

This application serves as an interactive demonstration of our empirical research pipeline. It processes chess games from raw PGN format move-by-move without lookahead, extracts 10 validated behavioral features, and scores whether an observed player's move pattern is more consistent with human or AI deliberation physics.

### Core Capabilities
- **End-to-End PGN Ingestion**: Accepts raw PGN streams or files with `[%clk]` annotations.
- **Strict Zero-Lookahead Extraction**: Computes streaming behavioral signals at move $k$ using only history from moves $1 \dots k$.
- **Interactive Chessboard**: Fully playable SVG board with move scrubber, play/pause controls, and adjustable auto-playback speeds (1x, 2x, 5x).
- **Real-Time Behavioral Telemetry**: Live gauges for all 10 features, plus rolling time-series charts for AI probability, thinking time, CV volatility, and clock countdowns.
- **Adjustable Operating Threshold**: Interactive threshold slider ($\tau \in [0.10, 0.90]$) to adjust operational sensitivity.
- **Offline Demo Mode**: Bundled 1-click preset matches (Human Grandmaster game, AI Engine bot match, and Human-vs-Bot exhibition).

---

## 2. Project Directory Structure

```
Chess-AI-Detection/
│
├── data/
│   ├── raw/                           # Raw Lichess PGN files
│   ├── processed/                     # Clean move dataset & behavioral feature CSV
│   └── demo_games/                    # Bundled sample PGN matches for offline demo
│
├── docs/
│   ├── DATASET_REPORT.md              # Ingestion audit and source verification
│   ├── PREPROCESSING_REPORT.md        # Data cleaning, rules, and clock normalization
│   ├── EDA_FINDINGS.md                # Exploratory analysis and behavioral hypotheses
│   ├── FEATURE_CANDIDATES.md          # 35 candidate behavioral signals
│   ├── FEATURE_ENGINEERING_REPORT.md  # Temporal integrity, formulation, and 10-feature subset
│   ├── MODEL_EVALUATION_REPORT.md     # 19-section empirical model benchmarking report
│   ├── ERROR_ANALYSIS.md              # False positive / false negative failure modes
│   └── APPLICATION_REPORT.md          # Architecture, inference flow, and demo guide
│
├── models/
│   ├── best_model.pkl                 # Serialized EDA Behavioral LightGBM model
│   ├── feature_columns.json           # Exact 10-feature schema definition
│   └── model_metadata.json            # Model parameters, version & benchmark metrics
│
├── notebooks/
│   ├── 01_data_inspection.ipynb      # Inspection & verification
│   ├── 02_preprocessing.ipynb        # Data preparation & cleaning
│   ├── 03_eda.ipynb                  # Exploratory behavioral analysis
│   ├── 04_model_training.ipynb       # Group-aware model training
│   └── 05_model_evaluation.ipynb     # Model comparison & ablation
│
├── results/
│   ├── model_comparison.csv           # Baseline vs EDA model metrics
│   ├── ablation_results.csv           # Stepwise ablation across feature families
│   ├── feature_importance.csv         # Normalized Gain and Split importance
│   ├── confusion_matrix.png           # Confusion matrix at operating threshold
│   ├── roc_curve.png                  # Comparative ROC curves
│   ├── precision_recall_curve.png     # Precision-Recall curves under 27:1 imbalance
│   └── calibration_curve.png          # Reliability diagram and Brier score
│
├── src/
│   ├── inference_features.py          # Canonical streaming feature extraction module
│   ├── predict.py                     # Decoupled prediction & PGN scoring API
│   ├── feature_engineering.py         # Training-time feature construction pipeline
│   └── preprocess.py                  # PGN cleaning and clock parsing pipeline
│
├── app/
│   ├── backend/
│   │   └── main.py                    # FastAPI application & REST endpoints
│   └── frontend/
│       ├── index.html                 # Interactive dashboard UI
│       ├── style.css                  # Modern dark-mode styling & glassmorphism
│       └── app.js                     # Board rendering, Chart.js telemetry & controls
│
├── tests/
│   ├── test_model.py                  # Model loading & schema tests
│   ├── test_inference_features.py     # Feature calculation & zero-leakage tests
│   ├── test_edge_cases.py             # Missing clocks, short games, invalid PGN
│   ├── test_api.py                    # FastAPI endpoints & static file delivery
│   └── run_all_tests.py               # Master test runner
│
├── requirements.txt                   # Production dependencies
└── README.md                          # Project documentation
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- pip

### Step 1: Clone and Set Up Virtual Environment
```bash
git clone https://github.com/your-username/Chess-AI-Detection.git
cd Chess-AI-Detection

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Running the Application

### Launch the Combined Backend & Web Interface
```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
👉 **[http://localhost:8000](http://localhost:8000)**

The web interface will automatically load the **Human Master Game** demo preset on initial startup.

---

## 5. Using the Interactive Dashboard

1. **Quick Demo Presets**:
   - Click **Human Master Game** to observe natural human deliberation surges, deep think pauses, and low AI likelihood.
   - Click **AI Bot Match** to see machine pacing: near-zero hesitation, flat deliberation across phases, and high AI likelihood (>85%).
   - Click **Human vs. Bot Challenge** to compare human and bot move physics within the same match.
2. **Custom PGN Analysis**:
   - Click **Upload PGN File** to load any `.pgn` file from your computer.
   - Click **Paste PGN Text** to paste raw PGN notation directly.
3. **Interactive Board Navigation**:
   - Use the `<< First`, `< Prev`, `Next >`, `Last >>` buttons or the **Move Scrubber Slider** to jump to any move.
   - Click **▶ Play** to start auto-playback; adjust speed with the **1x / 2x / 5x** toggles.
   - Click any row in the **Move History Table** to jump the board and metrics directly to that ply.
   - Press **Left / Right Arrow Keys** or **Spacebar** on your keyboard to navigate moves.
4. **Inspecting Live Telemetry**:
   - Observe the **10 Behavioral Meters** update at every step.
   - Check the **Behavioral Signals Observed** panel to see plain-language, grounded explanations of observed timing patterns.
   - Inspect the **Interactive Charts** below the workbench to analyze probability trajectories and thinking-time distributions over the entire game.
5. **Adjusting Decision Sensitivity**:
   - Drag the **Decision Threshold Slider** ($\tau$) to adjust classification sensitivity.
   - Use the **Strict (0.85)** preset for cheating detection scenarios requiring a low false positive rate ($<1.0\%$).

---

## 6. Programmatic Python Usage

You can also score individual feature vectors or full PGN games programmatically:

```python
from src.predict import predict_player, analyze_pgn

# 1. Score a single 10-feature behavioral vector
sample_features = {
    "consecutive_fast_moves": 4.0,
    "premove_rate_10": 0.60,
    "cumulative_premove_rate": 0.50,
    "current_endgame_clock_ratio": 0.0,
    "phase_deliberation_ratio": 1.05,
    "move_time_cv_10": 0.18,
    "time_pressure_jitter": 0.0,
    "relative_move_time": 0.20,
    "time_spent_ratio": 0.02,
    "tank_move_count_10": 0.0
}
result = predict_player(sample_features, threshold=0.50)
print(result["prediction"])         # "AI-like behavior detected"
print(result["ai_probability"])      # e.g. 0.8842

# 2. Analyze a complete PGN game move-by-move
with open("data/demo_games/human_match.pgn", "r") as f:
    pgn_str = f.read()

analysis = analyze_pgn(pgn_str, threshold=0.50)
print("White Profile:", analysis["summary"]["white"])
print("Black Profile:", analysis["summary"]["black"])
```

---

## 7. Running the Test Suite

Execute the complete, automated test suite:

```bash
python tests/run_all_tests.py
```

This validates:
- Model loading and feature schema ordering
- Exact formula replication with zero future lookahead
- Edge case handling (missing clocks, short games, zero variance windows, invalid PGNs)
- REST API endpoint contracts and static asset delivery

---

## 8. Scientific Disclaimer & Limitations

> **Important**: This model produces a **statistical behavioral classification** based on move deliberation physics and clock timing. **It does NOT prove cheating or engine assistance.**

### Known Boundary Conditions:
- **Bullet Scrambles**: In 1+0 bullet time scrambles, human players queue multiple rapid pre-moves with near-zero latency, which can produce false positive AI indications.
- **Anthropomorphic Engines**: Bots configured with human-like randomized delays (e.g., Maia Chess) closely mirror human deliberation curves, yielding near-chance discrimination in paired holdout settings.
- For high-stakes applications, an operational threshold of **$\tau \ge 0.85$** is recommended to keep false positive rates below $1.0\%$.
