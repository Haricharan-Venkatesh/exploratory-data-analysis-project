# Version 5 Technical Report: Position-Complexity-Aware Feature Engineering

**Project Title:** EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Stage:** Version 5 — Position Complexity Investigation & Controlled Experiment  
**Date:** October 2026  
**Status:** Completed, Verified, and Preserved  

---

## 1. Executive Summary

Version 5 investigates whether augmenting supervised baseline models with **position-complexity-aware behavioral features** improves engine-assistance detection compared to aggregate move-level metrics alone.

### Key Conclusions:
1. **Benchmark Schema Reality**:
   - The public ChessFraud benchmark (`data/benchmarks/chess_fraud/full.parquet`, 38,510 plies) contains multi-depth engine evaluations (`move_stockfish_1`, `move_stockfish_9`, `move_stockfish_15`) and position evaluations (`eval_before`, `eval_after`), but **does NOT contain second-best move evaluations** (`best_vs_second_cp_gap`).
   - Re-evaluating 38,510 positions with MultiPV Stockfish would require hours of heavy computation. In compliance with rigorous ML integrity standards, this benchmark limitation is documented rather than bypassed through fabrication.
2. **Empirical Discriminatory Power of Complexity**:
   - In obvious/shallow positions where Stockfish depth 1 agrees with depth 15, **both human and engine-assisted players find the engine move over 61% of the time** (Fair: 61.59%, Assisted: 69.85%, difference: +8.26%).
   - In non-obvious/depth-unstable positions where depth 1 disagrees with depth 15, **fair human accuracy plunges to 27.70%**, whereas engine-assisted players maintain **42.08% accuracy** (difference: **+14.38%**, a **+51.9% relative increase**).
   - In close/critical positions ($|eval| \le 200$ cp), assisted players achieve a **+14.31%** higher top-1 concordance and commit **-12.24 cp** fewer errors than fair players.
3. **Controlled Experimental Results (5-Fold StratifiedGroupKFold on Physical Matches)**:
   - Augmenting models from 13 to 18 features improves **Random Forest ROC-AUC from 0.7437 to 0.7614 (+0.0177)** and **PR-AUC from 0.6628 to 0.6938 (+0.0310)**.
   - Precision increases from 66.57% to **67.77%** for Random Forest and **67.56%** for Logistic Regression, reducing false positives across unseen matches.
   - In feature importance rankings, `unstable_top1_match_rate` emerged as the **#1 dominant feature (16.16% importance)**, surpassing even general `top1_match_rate` (14.09%).
4. **Architectural Preservation**:
   - Production models (`models/chess_fraud_rf.joblib`, `models/chess_fraud_lr.joblib`) and baseline features remain completely untouched and backward-compatible.
   - Experimental pipelines are saved separately in `models/experimental/`.

---

## 2. Investigation of Available Data (Task 1)

### 2.1 Benchmark Schema Inspection
The raw benchmark dataset was inspected directly from `data/benchmarks/chess_fraud/full.parquet`:
- **Shape**: 38,510 rows $\times$ 33 columns.
- **Physical Structure**: 505 physical chess matches $\times$ 2 player-game sides = 1,010 player-game observations.

| Column | Type | Null Count | Description & Availability |
|---|---|---|---|
| `position_before` | `str` | 0 | Full FEN of position prior to player move. |
| `eval_before` | `int64` | 0 | Stockfish 17 centipawn evaluation before move. |
| `eval_after` | `int64` | 0 | Stockfish 17 centipawn evaluation after move. |
| `centipawn_loss` | `int64` | 0 | Move centipawn loss relative to engine choice. |
| `normalized_centipawn_loss`| `float64` | 0 | Lichess win-percentage difference. |
| `move_stockfish_1` | `str` | 10,100 | Best move at search depth 1 (post-opening only). |
| `move_stockfish_9` | `str` | 10,100 | Best move at search depth 9 (post-opening only). |
| `move_stockfish_15` | `str` | 10,100 | Best move at search depth 15 (post-opening only). |
| `is_used` | `bool` | 0 | Filter for post-opening decision points (`half_move >= 21`). |

### 2.2 Status of `best_vs_second_cp_gap`
- In the local engine analysis pipeline (`chess_eda/engine_analysis.py`), Stockfish is queried with `MultiPV=5`, allowing the computation of `best_vs_second_cp_gap = abs(score_best - score_second)`.
- However, in `chess_fraud/full.parquet`, **there are no MultiPV evaluations or candidate gap scores**.
- **Scientific Limitation Documented**: Because candidate gap data cannot be derived from the benchmark without re-running engine searches across all 38,510 positions, `best_vs_second_cp_gap` cannot be computed consistently across both training and inference. It is therefore excluded from training feature vectors to prevent data leakage or training/inference divergence.

### 2.3 Move Classification by Search Depth Stability
Comparing `move_stockfish_1` and `move_stockfish_15` provides an empirical measure of tactical obviousness:
- **Depth-Stable Moves (`move_stockfish_1 == move_stockfish_15`)**: 46.34% of used positions. In these positions, the immediate shallow evaluation finds the same move as the deep search.
- **Depth-Unstable / Hard Moves (`move_stockfish_1 != move_stockfish_15`)**: 53.66% of used positions. These positions represent complex positions where shallow tactical heuristics fail or deeper calculation is required to find the engine's choice.

---

## 3. Candidate Feature Design (Task 2)

We engineered five position-complexity features that capture tactical difficulty and positional criticality:

### 3.1 Feature Definitions

1. **`unstable_top1_match_rate` (Hard Move Concordance)**:
   $$\text{Rate of } (move\_player == move\_stockfish\_15) \text{ when } move\_stockfish\_1 \ne move\_stockfish\_15$$
   Measures accuracy restricted strictly to positions requiring deep calculation.
2. **`stable_top1_match_rate` (Obvious Move Concordance)**:
   $$\text{Rate of } (move\_player == move\_stockfish\_15) \text{ when } move\_stockfish\_1 == move\_stockfish\_15$$
   Measures baseline accuracy on simple recaptures, straightforward tactics, and obvious moves.
3. **`depth_drop_rate` (Complexity Degradation Gap)**:
   $$\text{depth\_drop\_rate} = \text{stable\_top1\_match\_rate} - \text{unstable\_top1\_match\_rate}$$
   Quantifies how much a player's accuracy degrades when positions transition from obvious to complex. Human players experience a steep performance cliff (-33.89%), while engine-assisted players maintain higher precision across difficulties (-27.77%).
4. **`critical_top1_match_rate` (Tense Position Concordance)**:
   $$\text{Rate of } (move\_player == move\_stockfish\_15) \text{ when } |eval\_before| \le 200 \text{ cp}$$
   Restricts concordance measurement to balanced or critical positions, removing dilution from blowout "garbage-time" positions ($|eval| > 200$ cp) where almost any move preserves winning/losing advantage.
5. **`critical_mean_cp_loss` (Tense Position Loss Magnitude)**:
   $$\text{Mean centipawn loss when } |eval\_before| \le 200 \text{ cp}$$
   Evaluates error severity specifically when game balance is undecided.

### 3.2 Training & Inference Consistency
- **Benchmark Calculation**: Derived directly from `eval_before`, `move_stockfish_1`, `move_stockfish_15`, and `centipawn_loss`.
- **Inference Calculation**: Any new PGN evaluated with Stockfish produces `engine_score_before_cp` for every move (allowing critical position filtering) and can query depth 1 and depth $D$ for depth-stability comparison. When depth-1 telemetry is not available, median training imputation ensures stable prediction.

---

## 4. Controlled Experimental Evaluation (Task 3)

### 4.1 Methodology
- **Validation Scheme**: 5-Fold Stratified Group Cross-Validation (`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Grouping**: Grouped strictly by `physical_game_id`. Both White and Black records of each physical chess match are constrained to the same fold (zero match leakage).
- **Comparison**: Exactly identical fold partitions were used for the baseline 13-feature models and experimental models.

### 4.2 Out-of-Fold Cross-Validation Metrics

| Configuration | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Fold Std |
|---|---|---|---|---|---|---|---|---|
| **Baseline (13 feats)** | Logistic Regression | 70.99% | 66.57% | **56.27%** | 60.99% | 0.7559 | 0.6892 | 0.0195 |
| **Augmented (18 feats)** | Logistic Regression | **71.39%** (+0.40%) | **67.56%** (+0.99%) | 55.77% (-0.50%) | **61.10%** (+0.11%) | **0.7605** (+0.0046) | **0.6983** (+0.0091) | **0.0158** |
| **Baseline (13 feats)** | Random Forest | 70.69% | 66.57% | 54.79% | 60.11% | 0.7437 | 0.6628 | **0.0164** |
| **Augmented (18 feats)** | Random Forest | **71.39%** (+0.70%) | **67.77%** (+1.20%) | **55.28%** (+0.49%) | **60.89%** (+0.78%) | **0.7614** (+0.0177) | **0.6938** (+0.0310) | 0.0199 |

### 4.3 Confusion Matrix Analysis (1,010 Player-Games)

**Logistic Regression**:
- *Baseline*: TN = 488, FP = 115, FN = 178, TP = 229
- *Augmented (18 feats)*: TN = 494, FP = 109, FN = 180, TP = 227
- **Impact**: 6 fewer false positives (fair players incorrectly classified as assisted).

**Random Forest**:
- *Baseline*: TN = 491, FP = 112, FN = 184, TP = 223
- *Augmented (18 feats)*: TN = 496, FP = 107, FN = 182, TP = 225
- **Impact**: 5 fewer false positives and 2 additional true positives.

### 4.4 Ablation Study (Random Forest)

| Feature Set | Features | Accuracy | ROC-AUC | PR-AUC | Key Observation |
|---|---|---|---|---|---|
| Baseline | 13 | 70.69% | 0.7437 | 0.6628 | Initial benchmark baseline. |
| Sub1: Depth Stability Only | 16 | **72.08%** | 0.7533 | 0.6863 | Strongest pure accuracy gain (+1.39%). |
| Sub2: Critical Positions Only | 15 | 70.69% | 0.7509 | 0.6751 | Improves probability ranking without raw accuracy shift. |
| Sub3: Unstable + Critical Top1 | 15 | 71.19% | 0.7558 | 0.6812 | Focused top-1 concordances. |
| **All 5 Augmented** | 18 | 71.39% | **0.7614** | **0.6938** | Highest overall ranking quality (ROC-AUC & PR-AUC). |

### 4.5 Feature Importance & Coefficients in Augmented Models

**Top Random Forest Feature Importances (18 Features)**:
1. `unstable_top1_match_rate`: **16.16%** (#1 feature across entire forest)
2. `top1_match_rate`: **14.09%** (#2 feature)
3. `critical_top1_match_rate`: **7.78%** (#3 feature)
4. `stable_top1_match_rate`: **7.45%** (#4 feature)
5. `excellent_rate`: **6.59%**
6. `move_count`: **6.51%**
7. `mean_centipawn_loss`: **5.33%**
8. `depth_drop_rate`: **5.22%**

**Key Logistic Regression Coefficients**:
- `unstable_top1_match_rate`: **+0.5125** (strongest positive rate coefficient, exceeding general top-1 at +0.2717)
- `stable_top1_match_rate`: **+0.3289**
- `depth_drop_rate`: **-0.1590** (larger drop from simple to hard moves reduces engine-assistance score)
- `critical_top1_match_rate`: **+0.1136**

---

## 5. Architectural Preservation & Isolation (Task 4)

To protect existing working functionality:
1. **Production Models Untouched**: `models/chess_fraud_rf.joblib` and `models/chess_fraud_lr.joblib` continue serving baseline inference.
2. **Experimental Models Isolated**:
   - `models/experimental/chess_fraud_rf_v5_exp.joblib`
   - `models/experimental/chess_fraud_lr_v5_exp.joblib`
   - `models/experimental/v5_experiment_metadata.json`
3. **Dashboard Integration**: Tab 3 of `app.py` displays the experimental position-complexity findings alongside baseline metrics, with transparent disclosures regarding the absence of `best_vs_second_cp_gap` in the benchmark.

---

## 6. Scientific Limitations & Ethical Disclaimers (Task 5)

1. **Benchmark Scope**: The ChessFraud dataset represents 5-minute blitz games (`300+0`) among 49 club players (~2000 Elo) using Stockfish 15 hints. These findings do not generalize directly to classical games, bullet time controls, or grandmaster play.
2. **Candidate Gap Absence**: The lack of MultiPV evaluations in the public benchmark prevents testing candidate evaluation gaps (`best_vs_second_cp_gap`) without costly recomputation.
3. **Decision Boundary Semantics**: The 50% classification threshold remains an operational software boundary. Scores are uncalibrated empirical ranking scores, not legal or competitive proof of cheating.

---

## 7. Changed Files Summary

| File | Purpose |
|---|---|
| `chess_eda/ml/features.py` | Defined `V5_COMPLEXITY_FEATURE_COLUMNS`, `V5_ALL_FEATURE_COLUMNS`; updated feature extractors with `include_complexity`. |
| `chess_eda/ml/v5_experiments.py` | Complete controlled 5-fold cross-validation runner, ablation evaluation, and experimental model serializer. |
| `chess_eda/ml/__init__.py` | Exported V5 complexity constants and experiment runner. |
| `app.py` | Added comparative Version 5 experimental evaluation section to Tab 3. |
| `models/experimental/*` | Serialized experimental pipelines and experiment metadata. |
| `reports/v5_position_complexity_findings.md` | Comprehensive scientific and experimental findings report. |
| `tests/test_v5_complexity.py` | Integration tests verifying V5 feature computation, physical match grouping, and pipeline execution. |
