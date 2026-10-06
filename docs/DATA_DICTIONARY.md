# Data Dictionary

**Project**: EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Document**: Data Dictionary for Clean Move-Level Dataset and Raw PGN Metadata  
**Primary Dataset**: `data/processed/clean_chess_moves.csv`  
**Intermediate Dataset**: `data/processed/intermediate/game_metadata.csv`  
**Audit Record**: `data/processed/intermediate/cleaning_audit.json`  

---

## 1. Primary Move-Level Dataset (`data/processed/clean_chess_moves.csv`)

This is the primary tabular dataset resulting from the Data Inspection and Preprocessing phase. Each row represents **one move (ply)** played by either White or Black in a standard rated chess game.

### Schema Overview

| Column Name | Data Type | Nullable? | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| `game_id` | String | No (0% null) | `jF3xnsFH` | Unique 8-character Lichess identifier extracted from the `Site` URL header. Identifies the game session. |
| `player_id` | String | No (0% null) | `lokkkrrd` | Lichess username of the player executing this move. |
| `player_color` | String (Categorical) | No (0% null) | `White`, `Black` | Piece color commanded by the player making the move. |
| `move_number` | Integer | No (0% null) | `1`, `2`, `24` | Full-move counter according to standard chess convention ($1, 2, 3, \dots$). Increments after Black's move. |
| `move` | String | No (0% null) | `e4`, `Nf3`, `O-O`, `exd5` | Move notation in Standard Algebraic Notation (SAN). Fully validated as a legal move on the current board state. |
| `move_time` | Float | Yes (0.010% null) | `2.45`, `0.0`, `14.2` | Thinking time spent by the player on this move in seconds. Computed from clock deltas and Fischer increments. NaN if uncomputable. |
| `player_rating` | Integer | No (0% null) | `1742` | Pre-game Glicko-2 rating (displayed as Elo equivalent) of the player making the move. Range: [488, 3012]. |
| `game_phase` | String (Categorical) | No (0% null) | `Opening`, `Middlegame`, `Endgame` | Categorical game phase derived from a deterministic, reproducible material and move count rule. |
| `position` | String (FEN) | No (0% null) | `rnbqkbnr/pppppppp/8/8/4P3/...` | Board position state in Forsyth–Edwards Notation (FEN) **immediately before** the move was played. |
| `label` | String (Categorical) | No (0% null) | `HUMAN`, `AI` | Ground-truth player classification. `AI` if player has verified platform badge `BOT`; `HUMAN` otherwise. |
| `ply` | Integer | No (0% null) | `1`, `2`, `47` | Sequential half-move counter ($1, 2, 3, \dots, N$) within the game. White plays odd plies; Black plays even plies. |
| `clock_after_move`| Float | No (0% null) | `178.5` | Remaining clock time for the active player in seconds immediately after completing the move, from `[%clk]`. |
| `time_control` | String | No (0% null) | `180+2`, `60+0`, `300+3` | Official game time control format: `{base_seconds}+{increment_seconds}`. |
| `speed_category` | String (Categorical) | No (0% null) | `Bullet`, `Blitz`, `Rapid`, `Classical` | Standardized game speed category derived from estimated duration ($T_{\text{base}} + 40 \times I$). |
| `opponent_id` | String | No (0% null) | `maia1` | Lichess username of the opponent player. |
| `opponent_rating` | Integer | No (0% null) | `1519` | Pre-game Glicko-2 rating of the opponent. Range: [488, 3012]. |
| `game_result` | String (Categorical) | No (0% null) | `1-0`, `0-1`, `1/2-1/2` | Final game outcome: `1-0` (White win), `0-1` (Black win), `1/2-1/2` (Draw). |
| `eco` | String | No (0% null) | `B00`, `C50`, `E60` | Encyclopedia of Chess Openings (ECO) classification code identifying the opening branch. |

---

## 2. Detailed Definitions & Derivation Rules

### 2.1 `move_time` (Thinking Time)
- **Mathematical Formula**:
  - For player $p \in \{\text{White}, \text{Black}\}$ on their first move ($k = 1$, corresponding to plies 1 and 2):
    $$\Delta t_{p, 1} = \begin{cases} 0.0 & \text{if } T_{\text{base}} = 0 \text{ and } C_{p, 1} > 0 \\ \max(0.0, T_{\text{base}} - C_{p, 1}) & \text{otherwise} \end{cases}$$
  - For all subsequent moves ($k > 1$):
    $$\Delta t_{p, k} = C_{p, k-1} + I - C_{p, k}$$
    where $C_{p, k}$ is the player's clock immediately after move $k$ and $I$ is the Fischer increment in seconds.
- **Handling of Special Cases**:
  - **Instantaneous Pre-moves**: When $\Delta t = 0.0$, the move was queued in advance or played within the 0-second client window. This is a legitimate human and bot behavioral signal and is preserved as `0.0`.
  - **Clock Additions / Lag Inversion ($\Delta t < 0$)**: In online chess on Lichess, an opponent can click the "+15 seconds" gift button, or server lag compensation can adjust the player's clock upwards. In these 41 cases ($0.010\%$ of moves), elapsed thinking time cannot be computed reliably. Under strict data integrity protocols, `move_time` is set to `NaN` (null) without fabrication or imputation.

### 2.2 `label` (Ground Truth)
- **Platform Verification**:
  - `AI`: Assigned if the player possesses the official `BOT` platform title (`[WhiteTitle "BOT"]` or `[BlackTitle "BOT"]`). Registered through the Lichess Bot API (`/api/bot/account/upgrade`).
  - `HUMAN`: Assigned if the player account has no title or possesses human competitive titles (`GM`, `IM`, `FM`, `CM`, `NM`, `WFM`).
- **Purity Rule**: No labels are inferred from move accuracy, centipawn loss, engine evaluation, or player rating.

### 2.3 `game_phase` (Phase Heuristic)
A deterministic, reproducible piece-count and move heuristic:
- **Opening**: Move number $\le 10$ (first 20 plies) AND both Queens are present on board AND total non-pawn pieces $\ge 10$.
- **Endgame**:
  - Both Queens are off the board AND total non-pawn material $\le 16$ points (where Queen=9, Rook=5, Bishop=3, Knight=3), OR
  - If a Queen remains on the board, total non-pawn material $\le 13$ points (e.g., Queen vs Queen endgame or Queen vs single minor piece).
- **Middlegame**: All positions that satisfy neither Opening nor Endgame criteria.

### 2.4 `position` (Board State)
- Forsyth–Edwards Notation (FEN) representing the board configuration **before** the move is executed.
- Enables downstream feature extractors to query legal move counts, piece attacks, checks, captures, king safety, and material imbalances.

---

## 3. Intermediate Game Metadata (`data/processed/intermediate/game_metadata.csv`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `game_id` | String | Unique 8-character game ID. |
| `event` | String | Event header (e.g., `Rated Blitz game`, `Rated Bullet tournament`). |
| `speed_category` | String | Speed pool: `Bullet`, `Blitz`, `Rapid`, `Classical`. |
| `time_control` | String | Time control specification (e.g., `180+2`). |
| `white_player` | String | White player username. |
| `black_player` | String | Black player username. |
| `white_elo` | Integer | White player rating. |
| `black_elo` | Integer | Black player rating. |
| `white_title` | String | White title (`BOT`, `GM`, `FM`, or empty). |
| `black_title` | String | Black title (`BOT`, `GM`, `FM`, or empty). |
| `game_type` | String | Pairing category: `HUMAN_vs_HUMAN`, `HUMAN_vs_BOT`, `BOT_vs_BOT`. |
| `result` | String | Game outcome: `1-0`, `0-1`, `1/2-1/2`. |
| `eco` | String | ECO opening code. |
| `total_plies` | Integer | Total half-moves in the game. |

---

## 4. Raw PGN Header Tags Inventory

| Tag Name | Source | Description |
| :--- | :--- | :--- |
| `Event` | PGN Header | Game tournament or matchmaking format. |
| `Site` | PGN Header | Permanent URL link on Lichess. |
| `Date` | PGN Header | Date of the match. |
| `Round` | PGN Header | Match round. |
| `White` | PGN Header | White username. |
| `Black` | PGN Header | Black username. |
| `Result` | PGN Header | Match outcome. |
| `UTCDate` | PGN Header | UTC date of match start. |
| `UTCTime` | PGN Header | UTC time of match start. |
| `WhiteElo` | PGN Header | Pre-game rating of White. |
| `BlackElo` | PGN Header | Pre-game rating of Black. |
| `WhiteRatingDiff` | PGN Header | Rating change for White. |
| `BlackRatingDiff` | PGN Header | Rating change for Black. |
| `WhiteTitle` | PGN Header | Platform title badge for White. |
| `BlackTitle` | PGN Header | Platform title badge for Black. |
| `ECO` | PGN Header | ECO opening code. |
| `Opening` | PGN Header | Human-readable opening name. |
| `TimeControl` | PGN Header | Clock settings (`base+inc`). |
| `Termination` | PGN Header | Reason for game termination (`Normal`, `Time forfeit`, `Abandoned`). |
