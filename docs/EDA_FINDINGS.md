# EDA Findings: Behavioral Patterns in Human vs. AI Chess Play

**Project**: EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Dataset**: `data/processed/clean_chess_moves.csv` (416,978 moves, 6,154 games)  
**Date**: October 2026  
**Status**: Completed  

---

## Finding 1: Instantaneous Move (Pre-Move) Rate Disparity
- **Observation**: AI engines exhibit an overall instantaneous move rate ($\Delta t = 0.0\text{s}$) that is 1.73× higher than human players, driven by instant book lookups and immediate API dispatch.
- **Evidence**:
  - Across all 416,937 timed moves:
    - **AI zero-move rate**: **25.52%** (3,801 / 14,893 moves).
    - **Human zero-move rate**: **14.78%** (59,422 / 402,044 moves).
    - Difference is highly statistically significant: $\chi^2 = 1,189.4$, $p < 10^{-15}$, Cramer's $V = 0.053$.
  - In paired Human-vs-Bot games (exact same match conditions):
    - **AI zero-move rate**: **27.18%**.
    - **Human zero-move rate**: **20.96%**.
  - At the player level (active accounts with $\ge 15$ moves):
    - **AI median player zero-move rate**: **16.59%**.
    - **Human median player zero-move rate**: **9.09%**.
- **Human behavior**: Humans utilize pre-moves primarily in predictable forced recaptures or in extreme time-scrambles. In normal opening development, humans still spend 1–3 seconds visually verifying the board and clicking piece squares.
- **AI behavior**: Registered bots frequently execute opening book lines and non-complex replies with zero algorithmic latency ($0.0\text{s}$ recorded client delta).
- **Potential implication**: A rolling window pre-move frequency feature (e.g., `premove_rate_last_10_moves`) will provide a strong behavioral indicator for AI detection.

---

## Finding 2: Move-Time Volatility and Variance Compression in AI
- **Observation**: Human move times are characterized by heavy right-skewed volatility (bursts of fast moves interrupted by deep calculation pauses), whereas AI move times are tightly compressed and bounded.
- **Evidence**:
  - In paired Human-vs-Bot games under identical time controls:
    - **Human move-time standard deviation**: **7.15s** (mean: 4.17s, CV = 1.71).
    - **AI move-time standard deviation**: **4.42s** (mean: 3.27s, CV = 1.35).
    - Human variance is **2.62× larger** than AI variance ($s^2 = 51.13$ vs $19.50$).
  - Across the full dataset:
    - Human maximum move time: **658.0s** (10.9 minutes of continuous thinking).
    - AI maximum move time: **158.0s** (strictly capped by engine time allocator).
    - Interquartile Range (IQR): Human IQR is 3.0s (1.0s to 4.0s); AI IQR is 5.0s (0.0s to 5.0s).
- **Human behavior**: Human cognition is non-uniform. Humans recognize "critical positions" (tactical threats, pawn breaks, sacrifices) and invest significant clock reserves, then play subsequent forced moves rapidly.
- **AI behavior**: Engines allocate search time based on strict mathematical formulas (e.g., $\Delta t \approx \text{clock} / 30$ or node budgets), preventing the dramatic spikes in hesitation seen in human players.
- **Potential implication**: Volatility metrics such as `move_time_std`, `move_time_cv` (Coefficient of Variation), and `time_skewness` will capture this cognitive asymmetry.

---

## Finding 3: Phase-Dependent Behavioral Divergence
- **Observation**: The behavioral gap between Human and AI shifts fundamentally across game phases (Opening vs. Middlegame vs. Endgame).
- **Evidence**:
  - **Opening Phase**:
    - AI median move time: **0.0s** (Mean: 2.97s, Zero-move %: **54.70%**).
    - Human median move time: **1.0s** (Mean: 2.84s, Zero-move %: **26.28%**).
    - More than half of all opening moves by bots are executed with 0.0s elapsed!
  - **Middlegame Phase**:
    - Human mean move time peaks at **5.29s** (std = 8.32s; Zero-move %: **7.50%**).
    - AI mean move time is **4.69s** (std = 7.74s; Zero-move %: **18.51%**).
    - Humans deliberate significantly longer in complex middlegame positions.
  - **Endgame Phase**:
    - Human mean move time drops to **2.42s** (mean remaining clock: **78.6s**).
    - AI mean move time remains higher at **3.06s** (mean remaining clock: **111.9s**).
    - Humans suffer from clock starvation in the endgame, forcing hurried play.
- **Human behavior**: Humans spend cognitive effort in the middlegame, gradually draining their clock, leaving them in severe time pressure during technical endgames.
- **AI behavior**: Bots maintain disciplined time budgeting, preserving clock reserves so they have more time to calculate accurate endgame tablebases and long piece conversions.
- **Potential implication**: Features segmented by phase (e.g., `opening_zero_move_pct`, `middlegame_mean_move_time`, `endgame_clock_buffer`) will reflect these divergent strategic behaviors.

---

## Finding 4: Speed-Category Disconnect (Classical Deliberation vs. Fixed Engine Budgets)
- **Observation**: The distinction between human and AI thinking time expands dramatically in slower time controls (Classical and Rapid).
- **Evidence**:
  - **Classical Speed Pool**:
    - **Human mean move time**: **24.41s** (median: **12.0s**, std: **39.11s**).
    - **AI mean move time**: **7.81s** (median: **1.0s**, std: **17.72s**).
    - Humans take **3.1× longer** on average in classical chess, and human deliberation median is 12× that of bots!
    - Zero-move percentage in Classical: Human = **5.06%** vs AI = **43.07%**.
  - **Blitz Speed Pool**:
    - Human mean: **4.65s** (median: 3.0s, std: 6.47s, Zero %: **9.37%**).
    - AI mean: **3.47s** (median: 3.0s, std: 4.27s, Zero %: **23.91%**).
  - **Bullet Speed Pool**:
    - Human mean: **1.79s** (median: 1.0s, std: 2.43s, Zero %: **24.16%**).
    - AI mean: **1.83s** (median: 1.0s, std: 2.60s, Zero %: **29.42%**).
- **Human behavior**: When given generous clocks (Classical/Rapid), humans utilize the time to calculate deep variations and assess candidate moves.
- **AI behavior**: Many online bots operate with fixed node counts, shallow search depths, or fixed network throughput, failing to scale thinking time proportionally with available time controls.
- **Potential implication**: Raw move time must be normalized by time control (e.g., `relative_move_time = move_time / (base_time / 40)`), preventing speed-pool bias from confounding the classifier.

---

## Finding 5: Divergent Time-Pressure Dynamics (<10s Clock Emergency)
- **Observation**: Humans and AI react in opposite manners when entering severe time pressure.
- **Evidence**:
  - Under extreme time trouble (Clock Remaining $< 10\text{s}$):
    - **Human zero-move rate explodes to 34.68%** (up from 7.5% in unpressured middlegame).
    - **AI zero-move rate remains stable at 20.97%** (actually lower than its opening rate).
    - Human mean move time drops to **1.30s**, but with high jitter ($\text{std} = 2.27\text{s}$).
    - AI mean move time stays composed at **1.64s** with a narrow standard deviation ($\text{std} = 1.57\text{s}$).
  - Under comfortable clock banks ($> 2\text{ minutes}$):
    - Human zero-move rate is only **9.33%**.
    - AI zero-move rate is **35.85%**.
- **Human behavior**: When the clock ticks below 10 seconds, human psychology shifts to survival mode: panic pre-moves, hurried flagging attempts, and chaotic latency fluctuations.
- **AI behavior**: An engine does not experience psychological panic. It continues executing minimax search with adjusted depth, producing uniform, methodical moves even with 2 seconds on the clock.
- **Potential implication**: `time_pressure_jitter` ($\text{std}(\text{move\_time} \mid \text{clock} < 15\text{s})$) and `scramble_premove_ratio` provide unique psychological behavioral signals.

---

## Finding 6: Calculation Hesitation Outliers (The Human Agony Spike)
- **Observation**: Extreme single-move pauses ($> 60\text{s}$) are almost exclusively human cognitive occurrences.
- **Evidence**:
  - Across the entire dataset, 928 moves had thinking time $> 60\text{s}$:
    - **Human share**: **96.34%** (894 moves).
    - **AI share**: **3.66%** (34 moves).
  - For deep pauses $> 120\text{s}$ (131 total occurrences):
    - **Human share**: **96.18%** (126 moves).
    - **AI share**: **3.82%** (5 moves).
  - Breakdown of human $>60\text{s}$ moves by phase:
    - Middlegame: **662 moves (71.3%)**
    - Opening: **244 moves (26.3%)**
    - Endgame: **22 moves (2.4%)**
- **Human behavior**: A human confronted with an unexpected tactical blow, queen sacrifice, or complex piece exchange will enter "the tank", sitting motionless for 1 to 5 minutes to calculate lines.
- **AI behavior**: Engines allocate search time logarithmically; an engine almost never spends 100+ seconds on a single move in blitz or rapid games unless throttled by server lag.
- **Potential implication**: Features capturing pause spikes, such as `max_move_time_ratio = max(move_time) / mean(move_time)` and `num_tank_moves (>30s)`, will cleanly isolate human decision-making.

---

## Finding 7: Game Longevity & Endgame Conversion Tenacity
- **Observation**: Bot games last substantially longer than human games, reflecting relentless defensive tenacity and lack of resignation blunders.
- **Evidence**:
  - In `BOT_vs_BOT` games:
    - **Mean plies per game**: **114.9** (median: **94.0 plies**).
  - In `HUMAN_vs_HUMAN` games:
    - **Mean plies per game**: **66.8** (median: **63.0 plies**).
  - In `HUMAN_vs_BOT` games:
    - **Mean plies per game**: **79.9** (median: **75.5 plies**).
- **Human behavior**: Human players frequently resign early when down material (e.g. lost queen or rook), blunder into quick tactical checkmates, or agree to mutual draws in equal middlegames.
- **AI behavior**: Engines do not experience despair; they play on, defending difficult piece down endgames for 50+ plies, converting pawn majorities with precision.
- **Potential implication**: Game-level aggregations (`game_length_plies`, `endgame_ply_fraction`) provide valuable supplementary context when classifying player accounts.

---

## 8. Summary of Hypotheses Supported vs. Refuted by Data

| Hypothesis | Supported by Data? | Empirical Finding |
| :--- | :--- | :--- |
| **"Bots think faster than humans overall"** | **PARTIALLY REFUTED** | Raw mean times are close (Human 4.17s vs AI 3.95s). The difference lies in **variance, distribution, and zero-moves**, not simple average speed. |
| **"Bots never make 0-second moves"** | **REFUTED** | Bots make **more** 0-second moves than humans (25.52% vs 14.78%), particularly in the opening (54.70%). |
| **"Human move times are more volatile"** | **STRONGLY SUPPORTED** | Human move-time variance is 2.62× larger than bot variance in paired games ($s^2 = 51.13$ vs $19.50$). |
| **"Humans panic in time pressure, bots do not"** | **STRONGLY SUPPORTED** | Human zero-move rate surges to 34.68% under 10s clock; AI pacing remains uniform at 1.64s. |
| **"Bots scale thinking time up in Classical"** | **REFUTED** | Bots average only 7.81s in Classical games, while humans deliberate for 24.41s. |
