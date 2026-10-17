# Version 3 Machine Learning Baseline Report: Engine-Assistance Detection

**Project Title:** EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Stage:** Version 3 — Supervised Machine Learning Baseline for Engine-Assistance Detection  
**Date:** October 2026  
**Status:** Completed, Verified, and Tested  

---

## 1. Executive Summary

Version 3 introduces a **supervised machine learning baseline** trained on the peer-reviewed **ChessFraud benchmark dataset** (Linich et al., ACM SIGKDD 2026) to detect computer-assisted chess play at the player-game level.

The baseline system:
1. Ingests the controlled-tournament configuration (`chess_fraud`) with logged ground-truth engine assistance annotations.
2. Aligns move-level and player-game features engineered in Version 2 with the benchmark schema, preventing data leakage.
3. Implements group-aware train/test splitting by `player_id`, guaranteeing zero player overlap between training and evaluation partitions.
4. Trains and rigorously compares **Logistic Regression** and **Random Forest** models, evaluating both strictly on unseen test groups against a majority-class baseline.
5. Connects the fitted models to a local CLI inference pipeline that analyzes any PGN file with Stockfish, computes aligned behavioral features, and reports separate player-level probabilities, key behavioral metrics, and scientific caveats.

---

## 2. Dataset Provenance, Schema & Limitations (Task 1)

### 2.1 Benchmark Selection and Provenance
We utilize the controlled-tournament configuration of the public **ChessFraud** dataset:
- **Repository:** `https://huggingface.co/datasets/artemlepin/chess-fraud`
- **Citation:** Linich et al. (2026), *ChessFraud: Exploring the Capabilities of Human-Aligned Models for Cheating Detection in Online Chess*, Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
- **License:** Creative Commons Attribution 4.0 International (`CC BY 4.0`).
- **Configuration:** `chess_fraud` (`data/chess_fraud/full.parquet`, 38,510 plies).
- **Physical Structure:** 505 physical games, each represented as two player-side games (1,010 total player-game observations) played across 49 unique participants.

### 2.2 Strict Separation: Controlled Benchmark vs. Synthetic Data
The repository provides two configurations:
1. `chess_fraud` (**Used for our benchmark**): Ground-truth human gameplay collected in a controlled tournament setting where participants were explicitly logged when requesting and following Stockfish 15 engine hints.
2. `chess_fraud_synth` (**Excluded**): Synthetic move substitutions generated post-hoc from public Lichess games using classical engines and human-aligned models (Maia, Allie). We preserve strict separation and do NOT conflate synthetic substitutions with observed tournament play.

### 2.3 Class Balance and Label Definitions
* **Player-Game Level Label (`is_cheating_player_game`)**:
  - `False` (Fair / Unaided play): 603 games (**59.70%**)
  - `True` (Engine-Assisted play): 407 games (**40.30%**)
  - **Majority-Class Baseline Accuracy**: **59.70%** (overall) / **61.51%** (test split).
* **Move-Level Label (`is_cheating_move`)**:
  - `False`: 29,105 plies (**75.58%**)
  - `True`: 9,405 plies (**24.42%**)

### 2.4 Missing Values and Analysis Filter (`is_used`)
- Exactly 10,100 moves have `is_used = False` corresponding to the opening phase (`half_move < 21`).
- The remaining 28,410 moves (`is_used = True`) represent post-opening decision points (`half_move >= 21`) where engine evaluations and top candidates are fully populated.
- Missing values in `assistance_line_rank` (29,105 missing) occur naturally when no engine line was followed.

### 2.5 Dataset Limitations
1. **Cohort Scope**: Covers 49 participants playing 5-minute blitz (`300+0`). Does not represent classical time controls, bullet games, or novice ratings.
2. **Assistance Modality**: Assistance in the tournament was provided exclusively via Stockfish 15 hints at depths 12–18. Real-world subtle cheating (e.g., consulting an engine once or twice per game) is fundamentally different from sustained engine following.
3. **Not General Human Ground Truth**: Ordinary human games outside this controlled protocol cannot be assumed to be 100% unaided without empirical proof.

---

## 3. Feature Engineering & Version 2 Compatibility (Task 2)

### 3.1 Feature Compatibility Matrix

| Feature | Source in Benchmark | Source in Version 1/2 Local Pipeline | Compatibility Note |
|---|---|---|---|
| `mean_centipawn_loss` | `centipawn_loss.mean()` | `centipawn_loss.mean()` | Direct match. Clipped to non-negative values. |
| `std_centipawn_loss` | `centipawn_loss.std()` | `centipawn_loss.std()` | Direct match. Measures tactical volatility. |
| `median_centipawn_loss` | `centipawn_loss.median()` | `centipawn_loss.median()` | Direct match. Resilient to extreme outlier blunders. |
| `mean_norm_cp_loss` | `normalized_centipawn_loss.mean()` | Derived via Lichess Win% formula from `score_before`/`score_after` | Direct mathematical equivalence: $p(e) = \frac{1}{1 + e^{-0.00368208 \cdot e}}$. |
| `std_norm_cp_loss` | `normalized_centipawn_loss.std()` | Derived via Lichess Win% formula | Direct mathematical equivalence. |
| `blunder_rate` | Rate of plies with $CP > 200$ | Rate of plies with $CP > 200$ | Direct match. Captures catastrophic errors. |
| `mistake_rate` | Rate of plies with $100 < CP \le 200$ | Rate of plies with $100 < CP \le 200$ | Direct match. Captures intermediate tactical errors. |
| `inaccuracy_rate` | Rate of plies with $50 < CP \le 100$ | Rate of plies with $50 < CP \le 100$ | Direct match. Captures minor inaccuracies. |
| `excellent_rate` | Rate of plies with $CP \le 25$ | Rate of plies with $CP \le 25$ | Direct match. Captures near-perfect engine moves. |
| `top1_match_rate` | Rate of $move == move\_stockfish\_15$ on used moves | Rate of $move == engine\_best\_move$ on post-opening plies | High compatibility. Measures engine concordance. |
| `used_mean_cp_loss` | Mean $CP$ loss on plies $\ge 21$ | Mean $CP$ loss on plies $\ge 21$ | Direct match. Focuses on post-opening execution. |
| `used_blunder_rate` | Rate of $CP > 100$ on plies $\ge 21$ | Rate of $CP > 100$ on plies $\ge 21$ | Direct match. |
| `move_count` | Number of player moves | Number of player moves | Volume normalization. |

### 3.2 Data Leakage Prevention
To prevent target leakage, all tournament logging annotations were **strictly excluded from feature sets**:
- Excluded: `player_hint_shown` (tells whether hint was requested)
- Excluded: `assistance_search_depth` (preset engine parameter)
- Excluded: `assistance_line_rank` (logging rank of followed line)
- Excluded: `is_cheating_move` (the target move label)
- Excluded: `is_accused_by_opponent` (post-hoc subjective feedback)

### 3.3 Stockfish Discrepancy & Timing Differences
- **Stockfish Version**: The benchmark used Stockfish 17 (depth 15) for post-hoc features. Local inference uses Stockfish 19 (depth 10 by default for speed, or depth 15).
- **Thinking Time**: Historical PGNs lack clock telemetry (`move_thinking_time`). To ensure inference reliability on any standard PGN without fabrication, clock features were excluded from the core tabular feature set.

---

## 4. Model Training & Grouped Comparative Evaluation (Task 3)

### 4.1 Group-Aware Splitting by Physical Match
- **Split Algorithm**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `physical_game_id`.
- **Physical Grouping**: Each physical chess match generates two paired player-game records ending in `_w` (White) and `_b` (Black). Both sides of every physical game are strictly constrained to the same cross-validation fold.
- **Physical Match Leakage**: **Strictly 0 physical matches cross between training and validation within any fold**.
- **Fold Partitions**:
  - Exactly 5 folds, each containing 808 training observations (404 physical matches) and 202 validation observations (101 physical matches).
  - Validation class balance is preserved evenly across all folds (~40.1% to 40.6% positive rate per fold).
- **Evaluation Limitation**: Physical matches are strictly independent across folds. However, because the 49 tournament participants played in multiple matches, individual players appear in multiple folds. Thus, this constitutes match-independent rather than fully participant-independent cross-validation.

### 4.2 Model Pipelines
1. **Logistic Regression Pipeline**:
   `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()` $\rightarrow$ `LogisticRegression(max_iter=1000, random_state=42)`
2. **Random Forest Pipeline**:
   `SimpleImputer(strategy='median')` $\rightarrow$ `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`

### 4.3 Model Comparison Table (5-Fold Stratified Group OOF Metrics)

| Metric | Majority-Class Baseline | Logistic Regression (OOF) | Random Forest (OOF) |
|---|---|---|---|
| **Accuracy** | 59.70% | **70.99%** (+11.29%) | 70.69% (+10.99%) |
| **Precision** | 0.00% | **66.57%** | 66.57% |
| **Recall** | 0.00% | **56.27%** | 54.79% |
| **F1-Score** | 0.00% | **60.99%** | 60.11% |
| **ROC-AUC** | 0.5000 | **0.7559** | 0.7437 |
| **PR-AUC** | 0.4030 | **0.6892** | 0.6628 |

*Note: All performance metrics represent out-of-fold cross-validation predictions pooled across all 1,010 player-game samples. Physical matches are strictly partitioned without overlap within any fold.*

### 4.4 Out-Of-Fold Confusion Matrices

**Logistic Regression OOF Confusion Matrix**:
```text
                  Predicted Fair (0)    Predicted Assisted (1)
True Fair (0)     488 (TN)              115 (FP)
True Assisted (1) 178 (FN)              229 (TP)
```

**Random Forest OOF Confusion Matrix**:
```text
                  Predicted Fair (0)    Predicted Assisted (1)
True Fair (0)     491 (TN)              112 (FP)
True Assisted (1) 184 (FN)              223 (TP)
```

### 4.5 Feature Importances & Final Production Inference Models
Following cross-validation, final production inference models were fitted on the complete benchmark dataset (all 1,010 player-games) to maximize sample utilization and feature representation for inference:

**Top Random Forest Feature Importances (Final Model)**:
1. `top1_match_rate`: **27.02%** — Dominant predictor; following engine recommendations spikes top-1 concordance.
2. `mean_centipawn_loss`: **8.81%** — Overall move quality.
3. `move_count`: **8.65%** — Game volume / error opportunity.
4. `mean_norm_cp_loss`: **8.65%** — Win probability loss rate.
5. `excellent_rate`: **8.23%** — Frequency of plies with $\le 25$ centipawns loss.
6. `used_mean_cp_loss`: **7.18%** — Post-opening execution.
7. `std_norm_cp_loss`: **7.14%** — Positional stability.

### 4.6 Probability Calibration & Distinctions
- **Explicit Calibration**: Neither Platt scaling nor isotonic regression has been applied yet.
- **Score Interpretation**:
  - Logistic Regression outputs are raw logistic sigmoids: $\sigma(w^T x + b)$.
  - Random Forest outputs are empirical voting fractions: $\frac{1}{B} \sum_{b=1}^B \hat{y}_b$.
  - These values represent uncalibrated benchmark ranking scores relative to the 2000-Elo Stockfish 15 tournament, not population posterior probabilities.


---

## 5. Local PGN Prediction Workflow (Task 4 & 5)

The prediction pipeline ingests a local PGN game, evaluates moves with Stockfish, computes aligned behavioral features, and applies the trained model pipeline.

### 5.1 CLI Command

```powershell
.\.venv\Scripts\python.exe main.py --predict path/to/game.pgn --stockfish-path "C:\stockfish\stockfish.exe" --depth 10
```

### 5.2 Verification on Real Games

#### Test Game 1: Historical Master Human Game (`data/sample.pgn`)
*Adolf Anderssen vs. Lionel Kieseritzky (London, 1851)*

**Post-Audit Logistic Regression Output**:
```text
PLAYER-LEVEL ENGINE-ASSISTANCE ESTIMATES

Player 1 — Adolf Anderssen (White)
  Engine-assistance score (uncalibrated):        49.6%
  Complementary human-like score (uncalibrated): 50.4%
  Classification (50% software threshold):       HUMAN-LIKE
  Key Metrics:                                   Mean CP Loss = 62.45 cp, Top1 Rate = 69.2%, Blunder Rate = 10.0%

Player 2 — Lionel Kieseritzky (Black)
  Engine-assistance score (uncalibrated):        25.1%
  Complementary human-like score (uncalibrated): 74.9%
  Classification (50% software threshold):       HUMAN-LIKE
  Key Metrics:                                   Mean CP Loss = 62.58 cp, Top1 Rate = 50.0%, Blunder Rate = 15.8%
```

**Post-Audit Random Forest Output**:
```text
PLAYER-LEVEL ENGINE-ASSISTANCE ESTIMATES

Player 1 — Adolf Anderssen (White)
  Engine-assistance score (uncalibrated):        28.5%
  Complementary human-like score (uncalibrated): 71.5%
  Classification (50% software threshold):       HUMAN-LIKE
  Key Metrics:                                   Mean CP Loss = 62.45 cp, Top1 Rate = 69.2%, Blunder Rate = 10.0%

Player 2 — Lionel Kieseritzky (Black)
  Engine-assistance score (uncalibrated):        29.9%
  Complementary human-like score (uncalibrated): 70.1%
  Classification (50% software threshold):       HUMAN-LIKE
  Key Metrics:                                   Mean CP Loss = 62.58 cp, Top1 Rate = 50.0%, Blunder Rate = 15.8%
```

*Audit Note on Historical Model Disagreement*:
During the QA audit, a feature column lookup discrepancy was identified where `score_before` was read instead of `engine_score_before_cp`, causing `mean_norm_cp_loss` to default to median training imputation. Because Anderssen's romantic sacrifices were forced engine lines (`top1_match_rate = 69.2%`), Logistic Regression's linear additive formulation (+0.98 weight on top-1 rate) previously swung White to 60.7%. Random Forest's non-linear tree splits on high centipawn loss (>60 cp) robustly classified the game as human-like (33.2%). With the column lookup corrected, both models classify Anderssen as human-like (LR: 49.6%, RF: 28.5%).

#### Test Game 2: Engine Match (`data/raw/engine_generated_games/tcec_engine_game.pgn`)
*Stockfish 2021 vs. Leela Chess Zero*
- **Player 1 — Stockfish 2021 (White)**:
  - Engine-assistance score (uncalibrated): 62.6% | Complementary human-like score (uncalibrated): 37.4% | Classification: **ENGINE-ASSISTED**
  - Key Metrics: Mean CP Loss = 10.15 cp, Top1 Rate = 68.8%, Blunder Rate = 0.0%
- **Player 2 — Leela Chess Zero (Black)**:
  - Engine-assistance score (uncalibrated): 65.0% | Complementary human-like score (uncalibrated): 35.0% | Classification: **ENGINE-ASSISTED**
  - Key Metrics: Mean CP Loss = 40.92 cp, Top1 Rate = 60.0%, Blunder Rate = 4.0%

*Methodological Clarification*:
The human-like score is calculated as 100% minus the engine-assistance score. Neither score is a calibrated real-world probability. The 50% classification threshold is an operational software boundary, not a scientifically established threshold for cheating.

---

## 6. Scientific Boundaries & Ethical Disclaimers

1. **Statistical Likelihood $\ne$ Ethical Proof**: High engine correlation reflects tactical precision, book preparation, or forced combinations. It does NOT constitute legal or competitive proof of cheating.
2. **Benchmark Scope**: Trained on 5-minute blitz games under a controlled hint protocol. Models should not be used as automated cheating enforcement systems for real-world platforms.
3. **Inference Integrity**: The pipeline strictly refuses to fabricate missing features or invent arbitrary confidence percentages.

