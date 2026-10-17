# Quality-Assurance Audit & Factual Validation: Versions 3 & 4 Baseline Models

**Date:** October 17, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project:** EDA_Project (Human–AI Chess Move Detection & Analysis)  
**Audited Components:** 
- Dataset provenance and parquet metrics ([`data/benchmarks/chess_fraud/full.parquet`](data/benchmarks/chess_fraud/full.parquet), [`chess_eda/ml/dataset.py`](chess_eda/ml/dataset.py))
- Feature extraction & engine consistency ([`chess_eda/ml/features.py`](chess_eda/ml/features.py))
- Group evaluation split & training ([`chess_eda/ml/train.py`](chess_eda/ml/train.py), [`models/chess_fraud_lr.joblib`](models/chess_fraud_lr.joblib), [`models/chess_fraud_rf.joblib`](models/chess_fraud_rf.joblib), [`models/model_metadata.json`](models/model_metadata.json))
- Inference pipeline & dashboard ([`chess_eda/ml/predict.py`](chess_eda/ml/predict.py), [`chess_eda/dashboard.py`](chess_eda/dashboard.py), [`app.py`](app.py))
- Audited Game: `Engaliawad` (758) vs `TheGreatIMU` (962), Rapid, 22 moves / 44 plies ([`data/raw/unlabeled/thegreatimu_game.pgn`](data/raw/unlabeled/thegreatimu_game.pgn))

---

## Executive Summary

This report establishes the factual accuracy of the Version 3 and Version 4 audit in `EDA_Project`. All statistics and mathematical claims below were directly calculated from [`data/benchmarks/chess_fraud/full.parquet`](data/benchmarks/chess_fraud/full.parquet), [`models/model_metadata.json`](models/model_metadata.json), and the audited game PGN.

### Key Audit Conclusions:
1. **Resolution of Conflicting Dataset Statistics:**
   - The controlled ChessFraud benchmark file (`full.parquet`) contains **38,510 move rows**, **505 physical chess matches**, and **1,010 player-game samples** (each match recorded from White `_w` and Black `_b` perspectives).
   - Prior claims of 649 games and 1,298 player-game samples were incorrect. The true count is **505 physical games** and **1,010 player-game observations across 49 unique players**.
   - The dataset is **moderately imbalanced**: 603 Fair games (**59.70%**) vs. 407 Assisted games (**40.30%**), not 50/50 balanced.
   - The time control is strictly **`300+0` (5-minute blitz)** across all rows (100%), not 3-minute blitz.
   - The feature schema consists of **exactly 13 features** in [`ML_FEATURE_COLUMNS`](chess_eda/ml/features.py#L20-L34), not 10.

2. **Evaluation Split Audit:**
   - The evaluation split uses `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped strictly by **`player_id`**, allocating 771 games (39 players) to train and 239 games (10 players) to test.
   - **Player ID leakage is strictly zero** (0 overlapping players).
   - **Game-level physical overlap exists:** Because the split grouped by individual `player_id` rather than base match ID, **187 physical matches** have one player in train and the opposing player in test.

3. **ChessCheatDetector.com Methodology & Context:**
   - **Published Methodology:** ChessCheatDetector does not merely equate raw accuracy with cheating; it publishes an algorithm incorporating **position complexity estimation** based on research by Biswas & Kenneth Regan (evaluating Stockfish move-ranking stability via Kendall tau distance across search depths), per-move margin analysis, opening/trivial filtering, and a custom ML model.
   - **Game Dynamics:** In this 22-move game, White blundered an entire minor piece, a queen-rook fork, and a full rook by move 9 (`6. Qe2? Bb4+ 7. c3? Nxc3 8. Nxc3? Bxc3+ 9. Bd2 Bxa1`). From move 9 onwards, Black played with an overwhelming $+9.0$ advantage.
   - **What Remains Unknown:** We do not have ChessCheatDetector's proprietary weights, search depth, or move filters for this specific run, nor do we possess independent verified ground truth on whether `TheGreatIMU` used an engine. We cannot assume 98% is correct ground truth, nor can we dismiss it without internal access.

4. **Reproducibility & Benchmark Limitations on `TheGreatIMU`:**
   - The dashboard scores reproduce down to the decimal: Random Forest **37.96%** ($\rightarrow$ 38.0%), Logistic Regression **42.80%** ($\rightarrow$ 42.8%).
   - Black's play had a high median centipawn loss (**36.00 cp** vs. benchmark assisted mean of **4.02 cp**) and a low excellent-move rate (**42.11%** vs. benchmark assisted mean of **83.01%**), pulling both models toward "human-like".
   - The benchmark model establishes that Black's play does not resemble the dense, master-level assistance of the ChessFraud cohort. However, it cannot establish whether Black used an engine selectively for 1–2 tactical moves.

---

## 1. Direct Verification of Dataset Statistics

Computed directly from [`data/benchmarks/chess_fraud/full.parquet`](data/benchmarks/chess_fraud/full.parquet):

### 1.1 Structural Counts
| Metric | Parquet Computed Value | Status & Conflict Resolution |
|---|---|---|
| **Total Rows (Moves)** | **38,510** | Matches [`chess_eda/ml/dataset.py`](chess_eda/ml/dataset.py#L79). |
| **Unique Base Games (Physical Matches)** | **505** | Confirms 505 matches (each with `_w` and `_b` records). Resolves conflict: **505 is correct**, 649 was erroneous. |
| **Unique `game_id` Records** | **1,010** | Exactly $505 \times 2 = 1,010$ player-games. Resolves conflict: **1,010 is correct**, 1,298 was erroneous. |
| **Unique `player_id` Count** | **49** | Exactly 49 unique tournament participants. |
| **Time Control** | **`300+0` (5 min blitz)** | 38,510 / 38,510 rows (100.0%). Resolves conflict: **5-minute blitz**, not 3-minute blitz. |
| **Player Elo Distribution** | Mean: **1964.9** ± 325.0, Median: **1973.0**, Min: **1147**, Max: **2592** | Strong club to master cohort. |

### 1.2 Class Distribution & Imbalance
| Level | Label Column | Fair (0 / False) | Assisted (1 / True) | Class Balance |
|---|---|---|---|---|
| **Player-Game Level** | `is_cheating_player_game` | **603** (59.70%) | **407** (40.30%) | **Moderately Imbalanced (60/40)** |
| **Move Level (Game Target)**| `is_cheating_player_game` | **21,765** (56.52%) | **16,745** (43.48%) | Imbalanced (57/43) |
| **Move Level (Engine Hint)**| `is_cheating_move` | **29,105** (75.58%) | **9,405** (24.42%) | Strongly Imbalanced (76/24) |

*Resolution of Balance Conflict:* The dataset is **moderately imbalanced** (59.7% fair vs. 40.3% assisted at the player-game level). The majority-class baseline accuracy is **59.70%** overall and **61.51%** on the held-out test split. The prior claim of a "balanced 50/50 prior" was factually incorrect.

---

## 2. Evaluation Split & Leakage Audit

Inspected in [`chess_eda/ml/train.py`](chess_eda/ml/train.py#L130-L145) and [`models/model_metadata.json`](models/model_metadata.json#L5-L13):

### 2.1 Grouping Specification
- **Split Strategy:** `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`
- **Group Key:** Strictly grouped by **`player_id`**, NOT `game_id`.
- **Partition Sizes:**
  - Training partition: **771 player-games** across **39 unique players**.
  - Testing partition: **239 player-games** across **10 unique players**.
- **Player-Level Leakage:** **Zero player leakage** (`set(train_players).intersection(set(test_players)) == set()`).

### 2.2 Physical Game Overlap
- Because the split grouped by individual `player_id` rather than base match ID, head-to-head matches between a train-set player and a test-set player were divided:
  - White player record (`<match>_w`) assigned to training.
  - Black player record (`<match>_b`) assigned to testing.
- Out of 505 physical matches, **187 physical matches were shared across the split** (present in train from one player's perspective and in test from the other's).
- *Implication:* While player identities did not overlap, board positions and game flow were shared across 187 paired observations.

### 2.3 Correction Implemented: 5-Fold Stratified Group Cross-Validation
To completely eliminate physical game leakage, the evaluation methodology was upgraded to **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`** grouped strictly by `physical_game_id`:
- **Physical Game Leakage:** **Strictly 0 physical matches shared across training and validation within any fold**.
- **Out-of-Fold (OOF) Metrics (1,010 player-games):**
  - **Logistic Regression:** Accuracy: **70.99%** | Precision: **66.57%** | Recall: **56.27%** | F1: **60.99%** | ROC-AUC: **0.7559** | PR-AUC: **0.6892**
  - **Random Forest:** Accuracy: **70.69%** | Precision: **66.57%** | Recall: **54.79%** | F1: **60.11%** | ROC-AUC: **0.7437** | PR-AUC: **0.6628**
- **Inference Models:** Final production models were fitted on the full benchmark (all 1,010 player-games) to maximize feature coverage for live inference.
- **Remaining Evaluation Limitation:** While physical matches are strictly isolated without leakage, the 49 tournament participants played in multiple matches across the event, meaning player identities occur across folds. This constitutes match-independent rather than fully player-independent cross-validation.

---

## 3. Evaluation of ChessCheatDetector Claims

### 3.1 Published Methodology
ChessCheatDetector.com documents the following architectural pipeline:
1. **Server-Side Stockfish Evaluation:** Processes games deeply on backend infrastructure rather than lightweight browser engines.
2. **Position Complexity via Biswas & Regan (2015):** Measures decision difficulty by tracking move-ranking stability across incremental search depths with cleared hash tables. High ranking volatility across depths indicates high position complexity (non-obvious moves); stable rankings indicate low complexity.
3. **Move Filtering:** Filters out opening book plies, forced moves, and trivial positions where victory is mathematically guaranteed.
4. **Machine Learning Classifier:** Trains a model to evaluate patterns of engine agreement specifically conditioned on position complexity, producing per-move suspicion scores and an aggregate engine-assistance probability.
5. **Disclaimer:** Formally states that outputs are statistical anomaly scores and do not constitute absolute proof of cheating.

### 3.2 Inferences from the Audited Match
- In `thegreatimu_game.pgn`, White hung pieces on consecutive turns (`6. Qe2?`, `7. c3?`, `8. Nxc3?`, `9. Bd2 Bxa1`).
- From move 9 to 22, Black played with an overwhelming $+9.0$ evaluation advantage.
- Black's win-probability degradation was negligible: `mean_norm_cp_loss = 0.00737` (under 0.74% win probability lost per move), yielding a nominal move accuracy of ~98–99%.
- In runaway games with only 12 evaluated moves, if a tool's triviality filter does not fully exclude simple attacking conversions after catastrophic blunders, the resulting engine correlation metric can surge toward 98%.

### 3.3 What Remains Unknown
- We do not have access to ChessCheatDetector's proprietary model weights, exact search depth, or move-filter thresholds used on this game.
- We do not possess verified real-world ground truth (e.g. system logs or video capture) establishing whether `TheGreatIMU` used engine assistance.
- Therefore, we cannot verify whether 98% represents a true detection of engine assistance or an artifact of high accuracy in a trivial position. It cannot be treated as ground truth, nor can it be dismissed without internal inspection.

---

## 4. Deep Investigation of `TheGreatIMU` Game

### 4.1 Match Details
- **File:** [`data/raw/unlabeled/thegreatimu_game.pgn`](data/raw/unlabeled/thegreatimu_game.pgn)
- **White:** `Engaliawad` (Elo 758)
- **Black:** `TheGreatIMU` (Elo 962)
- **Time Control:** 600s (10-minute Rapid)
- **Length:** 22 full moves (44 plies). Evaluated post-opening plies for Black: **12 moves** (plies 22, 24, ..., 44).

### 4.2 Side-by-Side Feature Comparison with Benchmark Distributions

Computed directly across all 13 features in [`ML_FEATURE_COLUMNS`](chess_eda/ml/features.py#L20-L34):

| Feature | TheGreatIMU (Black) | ChessFraud Fair Mean ± Std | ChessFraud Assisted Mean ± Std | Comparison Note |
|---|---|---|---|---|
| `mean_centipawn_loss` | **31.47 cp** | 24.80 ± 23.57 cp | 18.01 ± 16.67 cp | Higher loss than both benchmark cohorts |
| `std_centipawn_loss` | **26.25 cp** | 46.13 ± 47.03 cp | 36.40 ± 35.68 cp | Moderate volatility |
| `median_centipawn_loss` | **36.00 cp** | 7.33 ± 9.80 cp | 4.02 ± 6.37 cp | **5.02 $\sigma$ above Assisted median loss** |
| `mean_norm_cp_loss` | **0.00737** | 0.0163 ± 0.0160 | 0.0118 ± 0.0114 | Low win-prob loss (game already won at move 9) |
| `std_norm_cp_loss` | **0.01248** | 0.0293 ± 0.0306 | 0.0233 ± 0.0242 | Stable winning evaluation |
| `blunder_rate` ($>200$ cp) | **0.000** | 0.0234 ± 0.0362 | 0.0144 ± 0.0287 | 0 blunders while up a rook |
| `mistake_rate` ($100-200$ cp)| **0.000** | 0.0370 ± 0.0497 | 0.0231 ± 0.0358 | 0 mistakes |
| `inaccuracy_rate` ($50-100$ cp)| **0.1053** | 0.0686 ± 0.0698 | 0.0503 ± 0.0554 | Higher inaccuracy rate than Assisted |
| `excellent_rate` ($\le 25$ cp) | **0.4211** | 0.7791 ± 0.1775 | 0.8301 ± 0.1300 | **3.15 $\sigma$ below Assisted mean** |
| `top1_match_rate` | **0.5833** | 0.4337 ± 0.1275 | 0.5501 ± 0.1361 | 7 / 12 post-opening moves matched |
| `used_mean_cp_loss` | **34.33 cp** | 30.37 ± 31.01 cp | 21.33 ± 21.21 cp | Post-opening loss higher than Assisted |
| `used_blunder_rate` ($>100$ cp)| **0.000** | 0.0779 ± 0.0975 | 0.0475 ± 0.0713 | 0 post-opening blunders |
| `move_count` | **22** | 36.09 ± 13.12 | 41.14 ± 15.18 | Short game |

### 4.3 Exact Mathematical Attribution of Model Predictions

Running [`predict_game`](chess_eda/ml/predict.py#L33-L158) confirms:
- **Random Forest:** `37.96%` $\rightarrow$ Displayed as **38.0%** (Human-like)
- **Logistic Regression:** `42.80%` $\rightarrow$ Displayed as **42.8%** (Human-like)

#### Exact Logistic Regression Step-by-Step Breakdown:
Formula: $z = w^T X_{\text{scaled}} + b$, where $b = -0.4650$:
- `top1_match_rate` (scaled $= +0.7241$, coef $= +0.9814$): **$+0.7106$**
- `median_centipawn_loss` (scaled $= +3.3082$, coef $= +0.0897$): **$+0.2967$**
- `blunder_rate` (scaled $= -0.5883$, coef $= -0.2551$): **$+0.1501$**
- `mean_norm_cp_loss` (scaled $= -0.4926$, coef $= -0.2512$): **$+0.1237$**
- `mistake_rate` (scaled $= -0.6911$, coef $= -0.1741$): **$+0.1203$**
- `mean_centipawn_loss` (scaled $= +0.4196$, coef $= +0.2325$): **$+0.0975$**
- `used_mean_cp_loss` (scaled $= +0.2505$, coef $= +0.2115$): **$+0.0530$**
- `std_centipawn_loss` (scaled $= -0.3689$, coef $= -0.0722$): **$+0.0266$**
- `inaccuracy_rate` (scaled $= +0.6779$, coef $= +0.0176$): **$+0.0119$**
- `std_norm_cp_loss` (scaled $= -0.5055$, coef $= +0.1154$): **$-0.0583$**
- `used_blunder_rate` (scaled $= -0.7339$, coef $= +0.2427$): **$-0.1781$**
- `move_count` (scaled $= -1.1379$, coef $= +0.4624$): **$-0.5262$**
- `excellent_rate` (scaled $= -2.2989$, coef $= +0.2840$): **$-0.6529$**
- **Intercept:** **$-0.4650$**
- **Total Log-Odds ($z$):** **$-0.2900$**
- **Probability:** $\sigma(-0.2900) = \frac{1}{1 + e^{0.2900}} = \mathbf{0.4280}$ (**42.80%**).

---

## 5. Scope of Benchmark Findings

### 5.1 What the Current Benchmark Establishes
- The trained models assess concordance with the **ChessFraud controlled tournament**, where ~2000-Elo players had continuous access to Stockfish 15 recommendations.
- Black’s play in this game diverged substantially from the assisted benchmark: Black had a high median centipawn loss (36.0 cp) and an excellent-move rate of only 42.1% (compared to 83.0% for assisted benchmark games).
- The models correctly determined that Black's statistical profile does not match that of a 2000-Elo player executing dense engine suggestions.

### 5.2 What the Current Benchmark Cannot Establish
- **Selective Engine Use:** The benchmark cannot rule out intermittent assistance (e.g., using an engine on plies 14–18 to execute the checkmate).
- **Ground Truth for Amateur Online Games:** Models trained on 2000-Elo blitz games under a full-game hint protocol cannot reliably establish whether an amateur 960-Elo rapid player was assisted in an asymmetric game where the opponent hung pieces early.

---

## 6. Regression Testing & Pipeline Health

All 146 unit and integration tests continue to pass without regression:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```
**Test Result:**
```text
146 passed in 46.93s
```
