# Feature Candidates: Behavioral Predictors for Human–AI Move Detection

**Project**: EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Stage**: Exploratory Data Analysis & Feature Ideation  
**Date**: October 2026  
**Status**: Preliminary Candidate Catalog (No final feature selection applied yet)  

---

## 1. Variables NOT Suitable as Predictive Behavioral Features

Before defining candidate features, we explicitly document variables that **must NOT be used as model inputs** to prevent target leakage, dataset bias, and memorization artifacts:

| Variable Name | Role in Dataset | Why It Is Unsuitable as a Model Feature |
| :--- | :--- | :--- |
| `game_id` | Metadata Identifier | Arbitrary platform session hash. Inclusion would allow tree models to memorize specific games, resulting in severe cross-validation leakage. |
| `player_id` | Player Identifier | Lichess username. A model trained on `player_id` would learn "maia1 is a bot" rather than detecting AI behavioral dynamics, destroying out-of-sample generalization to new players. |
| `opponent_id` | Player Identifier | Username of the opponent. Prone to memorization of player pairings and bot-testing accounts. |
| `label` | Target Ground Truth | The classification target (`HUMAN` vs `AI`). Using it as an input is trivial direct leakage. |
| `white_title` / `black_title` | Ground Truth Source | Platform badges (`BOT`, `GM`, `FM`, etc.) used to define `label`. Direct structural leakage. |
| `game_result` | Post-Game Outcome | Game result (`1-0`, `0-1`, `1/2-1/2`) is only known after the game terminates. Including it creates outcome leakage and survivorship bias. |
| `event` | Metadata String | Tournament and rating category strings (e.g. `Rated Blitz tournament https://...`). Contains platform-specific metadata unrelated to move physics. |

---

## 2. Promising Behavioral Feature Candidates

The following 10 feature candidates are derived directly from the empirical findings established during EDA:

### Feature 1: `move_time_cv` (Move-Time Coefficient of Variation)
- **Definition**: The ratio of the standard deviation of move time to the mean move time over a player's recent move window:
  $$\text{CV}_k = \frac{\sigma(\Delta t_{k-w:k})}{\mu(\Delta t_{k-w:k})}$$
- **Source Columns**: `move_time`, `ply`
- **Why EDA Suggests It**: Finding 2 demonstrated that human move times have a 2.62× higher variance in paired matches ($s^2 = 51.13$ vs $19.50$). Humans oscillate between instant moves and long calculation tanks, whereas AI search algorithms produce tightly regulated, uniform move intervals.
- **Potential Leakage Risk**: **None**. Uses only past move times within the current game.
- **Expected Interpretation**: High CV indicates human cognitive burstiness; low, stable CV indicates automated engine pacing.

---

### Feature 2: `premove_rate` (Rolling Zero-Second Move Frequency)
- **Definition**: The proportion of moves within a window where recorded elapsed time was $0.0\text{s}$:
  $$\text{PMR}_k = \frac{1}{w} \sum_{i=k-w+1}^{k} \mathbb{I}(\Delta t_i = 0.0)$$
- **Source Columns**: `move_time`
- **Why EDA Suggests It**: Finding 1 revealed that AI accounts make $0.0\text{s}$ moves at a rate of **25.52%** overall (and **54.70%** in the Opening), compared to **14.78%** for humans. Instant dispatch via API allows bots to execute moves without human motor latency.
- **Potential Leakage Risk**: **None**. Calculated strictly from move-by-move clock deltas.
- **Expected Interpretation**: Abnormally elevated pre-move rates in unpressured positions strongly favor AI classification.

---

### Feature 3: `phase_deliberation_ratio` (Middlegame-to-Opening Thinking Ratio)
- **Definition**: The ratio of average thinking time in the Middlegame to average thinking time in the Opening:
  $$\text{PDR} = \frac{\mu(\Delta t_{\text{Middlegame}})}{\max(0.5, \mu(\Delta t_{\text{Opening}}))}$$
- **Source Columns**: `move_time`, `game_phase`
- **Why EDA Suggests It**: Finding 3 showed that humans spend almost double the time in the middlegame (5.29s vs 2.84s, ratio $\approx 1.86$), whereas bots maintain flat pacing across phases (4.69s vs 2.97s, ratio $\approx 1.58$).
- **Potential Leakage Risk**: **None**. Derived from board FEN state and elapsed times.
- **Expected Interpretation**: Humans exhibit a large spike in deliberation upon entering the middlegame; bots exhibit minimal phase acceleration.

---

### Feature 4: `time_pressure_jitter` (Scramble Latency Standard Deviation)
- **Definition**: The standard deviation of move times when remaining clock is under 15 seconds:
  $$\text{TPJ} = \sigma(\Delta t \mid \text{clock\_after\_move} < 15\text{s})$$
- **Source Columns**: `move_time`, `clock_after_move`
- **Why EDA Suggests It**: Finding 5 proved that under $<10\text{s}$ clock pressure, humans exhibit severe latency volatility ($\text{std} = 2.27\text{s}$) due to frantic manual mouse clicking and panic pre-moves (34.68% zero rate). AI engines maintain a calm, uniform $\text{std} = 1.57\text{s}$.
- **Potential Leakage Risk**: **None**.
- **Expected Interpretation**: High variance during scramble indicates human stress; low variance indicates mechanical engine execution.

---

### Feature 5: `relative_move_time` (Budget-Normalized Move Time)
- **Definition**: Move time scaled by the standard time-budget unit of the game ($T_{\text{base}} / 40$):
  $$\text{RMT} = \frac{\Delta t}{\max(1.0, T_{\text{base}} / 40)}$$
- **Source Columns**: `move_time`, `time_control`
- **Why EDA Suggests It**: Finding 4 showed that raw move time cannot be compared directly across Bullet, Blitz, and Classical time controls. Normalizing by the expected per-move budget provides an invariant measure across speed pools.
- **Potential Leakage Risk**: **None**. $T_{\text{base}}$ is fixed and known before the game begins.
- **Expected Interpretation**: Scale-invariant measure of relative cognitive effort per move.

---

### Feature 6: `time_spent_ratio` (Clock Bank Depletion Fraction)
- **Definition**: The fraction of the player's remaining clock consumed on the current move:
  $$\text{TSR}_k = \frac{\Delta t_k}{C_{k-1} + I}$$
- **Source Columns**: `move_time`, `clock_after_move`
- **Why EDA Suggests It**: Captures how deeply a player is willing to deplete their available clock reserves on a single decision. Humans regularly commit 20%–50% of their remaining time on critical moves, whereas bots are bounded by search depth allocators.
- **Potential Leakage Risk**: **None**.
- **Expected Interpretation**: Extreme single-move TSR spikes indicate human hesitation/agony.

---

### Feature 7: `tank_move_indicator` (Deep Pause Presence)
- **Definition**: A binary flag or cumulative count of moves where $\Delta t > 30\text{s}$ (or $\Delta t > 3 \times \mu$):
  $$\text{TANK}_k = \mathbb{I}(\Delta t_k > 30.0)$$
- **Source Columns**: `move_time`
- **Why EDA Suggests It**: Finding 6 proved that **96.34%** of moves with thinking time $> 60\text{s}$ were played by humans. Deep tanks are predominantly human cognitive events.
- **Potential Leakage Risk**: **None**.
- **Expected Interpretation**: Presence of extreme pause outliers strongly increases the likelihood of a human player.

---

### Feature 8: `endgame_clock_buffer` (Normalized Endgame Clock Reserve)
- **Definition**: Remaining clock time at the moment the game transitions from Middlegame to Endgame, normalized by initial base time:
  $$\text{ECB} = \frac{C_{\text{endgame\_start}}}{T_{\text{base}}}$$
- **Source Columns**: `clock_after_move`, `game_phase`, `time_control`
- **Why EDA Suggests It**: Finding 3 highlighted that humans arrive in the endgame with significantly depleted clocks ($78.6\text{s}$ mean), whereas AI maintains superior clock management ($111.9\text{s}$ mean).
- **Potential Leakage Risk**: **None**.
- **Expected Interpretation**: Higher relative clock reserves in the endgame correlate with algorithmic time budgeting.

---

### Feature 9: `move_time_skewness` (Thinking Time Asymmetry)
- **Definition**: The non-parametric quantile skewness of move times over a player's history:
  $$\text{Skew}_{\text{np}} = \frac{(Q_3 - Q_2) - (Q_2 - Q_1)}{Q_3 - Q_1}$$
- **Source Columns**: `move_time`
- **Why EDA Suggests It**: Finding 2 revealed that human move-time distributions have pronounced right-side fat tails (rare deep calculations), whereas bot distributions have more symmetric or bimodal structures (fast book vs. constant depth).
- **Potential Leakage Risk**: **None**.
- **Expected Interpretation**: Heavy positive skewness aligns with human decision patterns.

---

### Feature 10: `consecutive_fast_moves` (Run Length of Near-Instant Replies)
- **Definition**: The current count of consecutive moves played with $\Delta t \le 0.5\text{s}$:
  $$\text{CFM}_k = \begin{cases} \text{CFM}_{k-1} + 1 & \text{if } \Delta t_k \le 0.5\text{s} \\ 0 & \text{otherwise} \end{cases}$$
- **Source Columns**: `move_time`
- **Why EDA Suggests It**: In the Opening and early Middlegame, AI engines frequently chain 5–15 consecutive instant moves from deep opening books and hash tables, a phenomenon rarely sustained by human players without at least momentary pauses.
- **Potential Leakage Risk**: **None**.
- **Expected Interpretation**: Long runs of sub-second moves are strong signatures of automated engine play.

---

## 3. Candidate Summary Matrix

| Candidate Feature | Category | Granularity | Leakage Risk | Primary Empirical Basis |
| :--- | :--- | :--- | :--- | :--- |
| `move_time_cv` | Timing Volatility | Rolling / Game | None | Finding 2 (Human variance 2.62× AI) |
| `premove_rate` | Latency / Pre-moves | Rolling / Game | None | Finding 1 (AI zero-moves 25.5% vs Human 14.8%) |
| `phase_deliberation_ratio` | Strategic Context | Game / Phase | None | Finding 3 (Human middlegame surge) |
| `time_pressure_jitter` | Stress Response | Time Trouble | None | Finding 5 (Human panic jitter vs AI composure) |
| `relative_move_time` | Normalized Pacing | Move Level | None | Finding 4 (Cross-speed normalization) |
| `time_spent_ratio` | Clock Allocation | Move Level | None | Finding 2 (Human deep tanks) |
| `tank_move_indicator` | Cognitive Outlier | Move Level | None | Finding 6 (96.3% of >60s moves are Human) |
| `endgame_clock_buffer` | Time Management | Phase Transition | None | Finding 3 (Bot endgame clock conservation) |
| `move_time_skewness` | Distributional Shape | Rolling / Game | None | Finding 2 (Heavy human right tail) |
| `consecutive_fast_moves` | Machine Automation | Move Level | None | Finding 1 & 3 (Extended engine book runs) |
