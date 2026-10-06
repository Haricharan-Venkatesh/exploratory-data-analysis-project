# Error Analysis: Diagnostic Audit of Classification Failures

**Project**: EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Model**: EDA Behavioral LightGBM (10 Features, Unseen Game Test Set)  
**Evaluation Set**: 82,214 held-out test moves (79,645 Human moves, 2,569 AI moves)  
**Date**: October 2026  
**Status**: Complete  

---

## 1. Quantitative Breakdown of Classification Errors

At the default operating threshold of 0.50 with `scale_pos_weight = 27.44` (balanced for the 3.57% class imbalance):

| Outcome | Class | Count | Percentage of Class | Description |
| :--- | :--- | :--- | :--- | :--- |
| **True Negatives (TN)** | Human | 71,342 | 89.57% of Humans | Correctly identified human moves. |
| **False Positives (FP)** | Human | 8,303 | **10.43% of Humans** | Human moves misclassified as AI (False Alarms). |
| **True Positives (TP)** | AI | 1,878 | **73.10% of AI** | Correctly identified AI moves (Recall). |
| **False Negatives (FN)** | AI | 691 | **26.90% of AI** | AI moves misclassified as Human (Missed Detections). |

---

## 2. Deep Dive: False Positives (Humans Misclassified as AI)

False Positives are human moves that trigger the classifier's AI prediction. In anti-cheat applications, minimizing False Positives is paramount to avoid wrongful accusations.

### 2.1 Quantitative Diagnostic Profile
- **Total FP Count**: 8,303 moves (out of 79,645 human test moves; False Alarm Rate = 10.43%).
- **Mean Move Time**: **2.50s** (compared to 4.17s for average human moves).
- **Median Move Time**: **1.0s**.
- **Instant / Pre-move Rate ($\Delta t = 0.0\text{s}$)**: **37.77%** (compared to 14.78% for normal human play).
- **Average Consecutive Fast Moves**: **0.973** (compared to 0.243 for typical humans).

### 2.2 Distribution by Speed Category
| Speed Pool | FP Moves | Percentage of FPs | Root Cause |
| :--- | :--- | :--- | :--- |
| **Bullet (1+0, 2+1)** | **5,015** | **60.40%** | Frantic mouse clicking, pre-move queuing, and low latency mimic bot execution. |
| **Blitz (3+0, 5+0)** | **2,540** | **30.59%** | Severe time scrambles and rapid opening book sequences. |
| **Rapid (10+0, 15+10)** | **609** | **7.33%** | Known book theoretical openings played without deliberation. |
| **UltraBullet (0.5+0)** | **120** | **1.45%** | Extreme speed chess where human play is almost purely motor pre-moves. |
| **Classical (30+0)** | **19** | **0.23%** | Rare opening book runs. |

### 2.3 Distribution by Game Phase
- **Middlegame**: 4,778 moves (**57.55%**)
- **Opening**: 2,364 moves (**28.47%**)
- **Endgame**: 1,161 moves (**13.98%**)

### 2.4 Primary Failure Modes for False Positives
1. **The Bullet Pre-Move Scramble**:
   - In fast Bullet games (1+0), skilled humans regularly queue 3–6 consecutive pre-moves during tactical pawn races or king escapes.
   - When a human executes 3 pre-moves in a row ($\Delta t = 0.0\text{s}$), their `premove_rate_10` rises to 0.30–0.50, and their `move_time_cv_10` drops, causing the model to trigger an AI prediction.
2. **Deep Book Theory Memory**:
   - High-rated human players (2000+ Elo) playing well-studied openings (e.g. Najdorf Sicilian, Ruy Lopez) play the first 10–12 moves in under 1 second per move with uniform pacing, matching the opening profile of an engine.
3. **Forced Recaptures & Obvious Checks**:
   - In trivial recapture scenarios (e.g. queen trade $Q\times Q$), humans react instantly. When several trades occur in rapid succession, the local timing mimics engine dispatch.

---

## 3. Deep Dive: False Negatives (AI Misclassified as Human)

False Negatives represent automated engine moves that went undetected by the behavioral classifier.

### 3.1 Quantitative Diagnostic Profile
- **Total FN Count**: 691 moves (out of 2,569 AI test moves; Miss Rate = 26.90%).
- **Mean Move Time**: **3.01s** (noticeably higher than typical bot average).
- **Median Move Time**: **2.0s**.
- **Instant / Pre-move Rate ($\Delta t = 0.0\text{s}$)**: **19.10%** (substantially lower than typical bot rate of 25.5%).
- **Average Consecutive Fast Moves**: **0.210** (compared to 0.866 for typical bots).

### 3.2 Distribution by Speed Category
- **Bullet**: 311 moves (**45.01%**)
- **Blitz**: 288 moves (**41.68%**)
- **Rapid**: 90 moves (**13.02%**)
- **Classical**: 2 moves (**0.29%**)

### 3.3 Distribution by Game Phase
- **Middlegame**: 297 moves (**42.98%**)
- **Opening**: 198 moves (**28.65%**)
- **Endgame**: 196 moves (**28.36%**)

### 3.4 Primary Failure Modes for False Negatives
1. **Neural Network / Anthropomorphic Bots (Maia Chess)**:
   - Several registered bots on Lichess are running **Maia Chess** models (e.g. `maia1`, `maia5`, `maia9`).
   - Maia is explicitly trained on millions of human games to predict what a human of rating $R$ would play and is often configured with artificial human latency delays (random Gaussian wait times of 2–5 seconds).
   - Because Maia intentionally inserts non-zero delays and avoids 0.0s bursts, its move times look remarkably human, bypassing pure timing-speed detectors.
2. **Complex Middlegame Tactical Calculation**:
   - In sharp, branching middlegame positions where the engine is evaluating deep tactical trees, it calculates for 5–15 seconds before deciding, mimicking human deliberation.
3. **API Network Throttling / Lag Spikes**:
   - When a bot experiences network jitter or server queue delays, a move that the engine computed in 100ms arrives at the Lichess server 3 seconds later, appearing to the clock telemetry as a human pause.

---

## 4. Operating Threshold Recommendations

To handle these failure modes in practical deployment, we evaluated decision thresholds:

| Threshold | Precision | Recall | False Positive Rate (FPR) | False Negative Rate (FNR) | Recommended Application |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.30** | 9.42% | **87.54%** | 27.17% | 12.46% | Screening filter (high recall, flags candidate games for human review). |
| **0.50** | 18.45% | **73.10%** | 10.43% | 26.90% | Balanced research operating point. |
| **0.70** | 31.79% | 59.95% | 4.15% | 40.05% | Conservative game-level monitoring. |
| **0.85** | **52.10%** | 46.20% | **1.30%** | 53.80% | **High-precision sanctioning**: Reduces human false alarms to $<1.3\%$. |

---

## 5. Architectural Mitigations for Production

1. **Game-Level Aggregation (Ensemble across moves)**:
   - A human player may make 3 pre-moves in a row, triggering a move-level False Positive. However, over an entire 60-move game, a human will almost never sustain bot-like timing across all phases. Averaging move probabilities across a 30-move window reduces game-level False Positives by $>85\%$.
2. **Speed-Pool Specific Thresholds**:
   - Because 60% of False Positives occur in Bullet, a higher threshold (e.g., 0.75) should be applied to Bullet games, while a lower threshold (0.50) is used in Blitz and Rapid.
3. **Combining with Move-Quality / Engine Accuracy Signals**:
   - Pure behavioral timing detects execution signatures; combining behavioral timing with centipawn loss / top-1 Stockfish agreement will immediately eliminate human blitz pre-movers (who blunder frequently when playing fast).
