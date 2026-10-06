# Feature Dictionary: Behavioral Chess Move Features

**Project**: EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Primary Dataset**: `data/processed/chess_behavioral_features.csv`  
**Date**: October 2026  
**Status**: Complete  

---

## 1. Overview of Dataset Columns

The dataset `data/processed/chess_behavioral_features.csv` contains 50 total columns structured as:
- **13 Identifier & Metadata Columns**: Preserved for tracking and slicing (strictly excluded from modeling).
- **2 Target Columns**: `label` (`HUMAN` / `AI`) and `target` (`0` / `1`).
- **3 Context Control Flags**: Rating differences, normalized move numbers.
- **7 Context Categorical One-Hot Flags**: Phase and speed pool indicators.
- **25 Behavioral Move-Time & Clock Features**: The core behavioral feature set.

---

## 2. Behavioral Feature Specifications

### 1. `relative_move_time`
- **Definition**: Thinking time on the current move scaled by the standard move-time budget unit of the game ($T_{\text{base}} / 40$).
- **Formula**: $\text{RMT}_k = \frac{\Delta t_k}{\max(1.0, T_{\text{base}} / 40.0)}$
- **Input columns**: `move_time`, `time_control`
- **Granularity**: Move-level (Level A)
- **Uses future information?**: **No**. $T_{\text{base}}$ is set before match start; $\Delta t_k$ is the elapsed time on current move.
- **Missing-value behavior**: 0 missing. (When raw move time is uncomputable, imputed to 0.0 with flag).
- **Behavioral interpretation**: Scale-invariant measure of relative cognitive effort. Humans spend higher relative budget units in critical positions.

---

### 2. `time_spent_ratio`
- **Definition**: Fraction of the player's available clock reserves consumed on the current move.
- **Formula**: $\text{TSR}_k = \min\left(1.0, \frac{\Delta t_k}{\max(1.0, C_{k-1} + I)}\right)$ (for move 1: $\Delta t_1 / \max(1.0, T_{\text{base}})$)
- **Input columns**: `move_time`, `clock_after_move`, `time_control`
- **Granularity**: Move-level (Level A)
- **Uses future information?**: **No**. Uses only previous clock $C_{k-1}$, increment $I$, and current move time $\Delta t_k$.
- **Missing-value behavior**: 0 missing. Bounded strictly in $[0.0, 1.0]$.
- **Behavioral interpretation**: Captures how deeply a player is willing to deplete their clock on a single decision. Humans display high single-move bank depletion spikes ($>0.20$), whereas bots maintain regulated search budgets.

---

### 3. `is_premove`
- **Definition**: Binary indicator of whether the move was executed with $0.0\text{s}$ elapsed recorded time.
- **Formula**: $\mathbb{I}(\Delta t_k == 0.0)$
- **Input columns**: `move_time`
- **Granularity**: Move-level (Level A)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Instant dispatch indicator. Bots execute 0.0s moves at nearly double the baseline rate of humans.

---

### 4. `tank_move_indicator`
- **Definition**: Binary flag indicating whether the move was a deep calculation pause ($>30\text{s}$).
- **Formula**: $\mathbb{I}(\Delta t_k > 30.0)$
- **Input columns**: `move_time`
- **Granularity**: Move-level (Level A)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Deep pause marker. 96.3% of moves with $>60\text{s}$ deliberation belong to humans; deep tanks strongly increase human probability.

---

### 5. `clock_remaining_ratio`
- **Definition**: Player's remaining clock time after completing the move, relative to starting base time.
- **Formula**: $\text{CRR}_k = \frac{C_k}{\max(1.0, T_{\text{base}})}$
- **Input columns**: `clock_after_move`, `time_control`
- **Granularity**: Move-level (Level A)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Tracks clock conservation across the game. Bots systematically maintain higher clock buffers into later plies.

---

### 6. `move_time_cv_10`
- **Definition**: Rolling coefficient of variation of move time over the player's last 10 moves within that game.
- **Formula**: $\text{CV}_k = \frac{\sigma_W}{\mu_W}$ if $\mu_W > 0.05$, else $0.0$
- **Input columns**: `move_time`
- **Granularity**: Rolling window ($W=10$ moves of active player; Level B)
- **Uses future information?**: **No**. Evaluated over window $[\max(1, k-9), k]$.
- **Missing-value behavior**: 0 missing. Handles $\mu=0$ and $N=1$ by falling back to 0.0.
- **Behavioral interpretation**: Direct measure of cognitive volatility. Humans oscillate between fast recaptures and deep tanks (high CV); bots exhibit uniform, formulaic pacing (low CV).

---

### 7. `move_time_mean_10`
- **Definition**: Rolling average move time over the player's last 10 moves.
- **Formula**: $\frac{1}{|W|} \sum_{i \in W} \Delta t_i$
- **Input columns**: `move_time`
- **Granularity**: Rolling window ($W=10$; Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Short-term tempo. Reflects local pacing regime.

---

### 8. `move_time_std_10`
- **Definition**: Rolling sample standard deviation of move time over the player's last 10 moves.
- **Formula**: $\sqrt{\frac{1}{|W|-1} \sum_{i \in W} (\Delta t_i - \mu_W)^2}$
- **Input columns**: `move_time`
- **Granularity**: Rolling window ($W=10$; Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing. Defaulted to 0.0 for $N=1$.
- **Behavioral interpretation**: Absolute local dispersion in thinking time.

---

### 9. `premove_rate_10`
- **Definition**: Rolling fraction of moves with $\Delta t == 0.0\text{s}$ over the player's last 10 moves.
- **Formula**: $\frac{1}{|W|} \sum_{i \in W} \mathbb{I}(\Delta t_i == 0.0)$
- **Input columns**: `move_time`
- **Granularity**: Rolling window ($W=10$; Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing. Bounded in $[0.0, 1.0]$.
- **Behavioral interpretation**: Local burstiness of automated/pre-move responses. Strongly elevated in bot opening play.

---

### 10. `tank_move_count_10`
- **Definition**: Rolling count of moves with $\Delta t > 30\text{s}$ over the player's last 10 moves.
- **Formula**: $\sum_{i \in W} \mathbb{I}(\Delta t_i > 30.0)$
- **Input columns**: `move_time`
- **Granularity**: Rolling window ($W=10$; Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing. Integer in $\{0, \dots, 10\}$.
- **Behavioral interpretation**: Clustering of intense human tactical deliberation.

---

### 11. `move_time_skewness_10`
- **Definition**: Rolling sample skewness of move time over the player's last 10 moves (requiring $\ge 4$ moves).
- **Formula**: Fisher-Pearson sample skewness clipped to $[-5.0, 5.0]$
- **Input columns**: `move_time`
- **Granularity**: Rolling window ($W=10$; Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing. Falls back to 0.0 when $N < 4$ or variance is zero.
- **Behavioral interpretation**: Quantifies the asymmetry of the thinking distribution. Positive skew indicates a baseline of fast moves with occasional long tanks.

---

### 12. `consecutive_fast_moves`
- **Definition**: Current count of consecutive moves executed with $\Delta t \le 0.5\text{s}$. Resets to 0 whenever $\Delta t > 0.5\text{s}$.
- **Formula**: $\text{CFM}_k = \begin{cases} \text{CFM}_{k-1} + 1 & \text{if } \Delta t_k \le 0.5\text{s} \\ 0 & \text{otherwise} \end{cases}$
- **Input columns**: `move_time`
- **Granularity**: Rolling streak (Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing. Integer $\ge 0$.
- **Behavioral interpretation**: **Highest single behavioral discriminator ($d = -0.686$)**. Bots regularly chain 5–15 sub-second moves from opening book or instant hash hits.

---

### 13. `in_time_pressure`
- **Definition**: Binary flag indicating whether the player's remaining clock is currently below 15 seconds.
- **Formula**: $\mathbb{I}(C_k < 15.0)$
- **Input columns**: `clock_after_move`
- **Granularity**: Move-level condition (Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Indicates entry into critical time scramble conditions.

---

### 14. `time_pressure_jitter`
- **Definition**: Standard deviation of move times strictly during the player's low-clock period ($C < 15\text{s}$) up to move $k$.
- **Formula**: $\sigma(\Delta t \mid C_i < 15\text{s}, i \le k)$ over last 5 time-trouble moves.
- **Input columns**: `move_time`, `clock_after_move`
- **Granularity**: Conditional rolling window (Level B)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing. Falls back to 0.0 when not in time pressure or on first pressure move.
- **Behavioral interpretation**: Motor/mental panic signature. Humans exhibit erratic jitter in time scrambles; bots remain composed.

---

### 15. `cumulative_mean_move_time`
- **Definition**: Cumulative average move time for the player in the current game from move 1 up to move $k$.
- **Formula**: $\frac{1}{k} \sum_{i=1}^k \Delta t_i$
- **Input columns**: `move_time`
- **Granularity**: Cumulative game-level (Level C)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Long-term pacing baseline.

---

### 16. `cumulative_move_time_std`
- **Definition**: Cumulative standard deviation of move time for the player in the current game from move 1 up to move $k$.
- **Formula**: $\sqrt{\frac{1}{k} \sum_{i=1}^k (\Delta t_i - \mu_{\le k})^2}$
- **Input columns**: `move_time`
- **Granularity**: Cumulative game-level (Level C)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Global timing volatility across the game session so far.

---

### 17. `cumulative_premove_rate`
- **Definition**: Cumulative fraction of moves executed with $\Delta t == 0.0\text{s}$ from move 1 up to move $k$.
- **Formula**: $\frac{1}{k} \sum_{i=1}^k \mathbb{I}(\Delta t_i == 0.0)$
- **Input columns**: `move_time`
- **Granularity**: Cumulative game-level (Level C)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing. Bounded in $[0.0, 1.0]$.
- **Behavioral interpretation**: Overall reliance on pre-moving in this game. Bots average 0.372 vs. humans 0.249 ($d = -0.505$).

---

### 18. `phase_deliberation_ratio`
- **Definition**: Ratio of cumulative Middlegame mean move time to completed Opening mean move time.
- **Formula**: $\text{PDR}_k = \begin{cases} 1.0 & \text{if phase is Opening} \\ \frac{\mu_{\text{mid}, \le k}}{\max(0.5, \mu_{\text{open}})} & \text{if phase is Middlegame or Endgame} \end{cases}$
- **Input columns**: `move_time`, `game_phase`
- **Granularity**: Cumulative phase-transition (Level C)
- **Uses future information?**: **No**. Uses only completed opening history and current middlegame moves.
- **Missing-value behavior**: 0 missing. Defaulted to 1.0 during opening.
- **Behavioral interpretation**: Quantifies the surge in cognitive deliberation when exiting opening book into middlegame tactical combat.

---

### 19. `is_endgame`
- **Definition**: Binary flag indicating whether the current board state satisfies the endgame definition.
- **Formula**: $\mathbb{I}(\text{game\_phase} == \text{'Endgame'})$
- **Input columns**: `game_phase`
- **Granularity**: Move-level state (Level C)
- **Uses future information?**: **No**.
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: Position phase indicator.

---

### 20. `current_endgame_clock_ratio`
- **Definition**: Player's remaining clock divided by base time, evaluated strictly during the Endgame phase (0.0 otherwise).
- **Formula**: $\mathbb{I}(\text{phase} == \text{Endgame}) \times \frac{C_k}{\max(1.0, T_{\text{base}})}$
- **Input columns**: `clock_after_move`, `game_phase`, `time_control`
- **Granularity**: Conditional move-level state (Level C)
- **Uses future information?**: **No**. (Temporally valid replacement for `endgame_clock_buffer`).
- **Missing-value behavior**: 0 missing.
- **Behavioral interpretation**: High clock reserves during endgame indicate superior robotic time management ($d = -0.475$).

---

## 3. Context Control Features

| Feature Name | Definition | Formula | Uses Future Info? |
| :--- | :--- | :--- | :--- |
| `move_number_normalized` | Game progression index | $\min(1.0, \text{move\_number} / 50.0)$ | No |
| `player_rating_normalized` | Standardized Elo | $(\text{Elo} - 1500) / 400.0$ | No |
| `opponent_rating_difference`| Skill differential | $\text{Elo}_{\text{player}} - \text{Elo}_{\text{opponent}}$ | No |
| `game_phase_opening` | Binary flag | $\mathbb{I}(\text{phase} == \text{'Opening'})$ | No |
| `game_phase_middlegame` | Binary flag | $\mathbb{I}(\text{phase} == \text{'Middlegame'})$ | No |
| `game_phase_endgame` | Binary flag | $\mathbb{I}(\text{phase} == \text{'Endgame'})$ | No |
| `speed_is_bullet` | Binary flag | $\mathbb{I}(\text{speed} == \text{'Bullet'})$ | No |
| `speed_is_blitz` | Binary flag | $\mathbb{I}(\text{speed} == \text{'Blitz'})$ | No |
| `speed_is_rapid` | Binary flag | $\mathbb{I}(\text{speed} == \text{'Rapid'})$ | No |
| `speed_is_classical` | Binary flag | $\mathbb{I}(\text{speed} == \text{'Classical'})$ | No |
| `move_time_missing` | Binary flag | $\mathbb{I}(\Delta t \text{ was NaN due to clock gifts})$ | No |
