# Version 2 Exploratory Data Analysis & Behavioral Feature Engineering Report

**Project Title:** EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  
**Stage:** Version 2 — Multi-Game Data Preparation, Exploratory Data Analysis, and Behavioral Feature Engineering  
**Date:** October 2026  
**Status:** Completed and Verified  

---

## 1. Executive Summary

Version 2 advances the project from single-game parsing (Version 0) and isolated engine evaluation (Version 1) into a production-grade **multi-game batch data preparation and behavioral feature engineering pipeline**. 

A batch pipeline was constructed to ingest PGN files organized by provenance (`human_games`, `engine_generated_games`, and `unlabeled`), extract board context and tactical features strictly prior to move execution, execute Stockfish 19 evaluations, and aggregate move-level dynamics into player-game behavioral profiles.

The pipeline successfully generated:
1. `data/processed/move_level_features.csv` (181 mainline moves, 40 features)
2. `data/processed/player_game_features.csv` (8 player-game profiles, 49 behavioral features)
3. `data/processed/dataset_manifest.json` (comprehensive provenance and error tracking)
4. `notebooks/02_eda_and_feature_engineering.ipynb` (executable EDA notebook with 13 analytical sections and 5 embedded charts)

---

## 2. Dataset Overview and Provenance Audit

The multi-game pipeline ingested 4 PGN files across three source categories. In accordance with rigorous scientific provenance standards, an independent provenance audit was conducted for every raw game file rather than relying on unverified PGN metadata headers:

| Source File | Category | Plies | Provenance Status | Authenticity Assessment & Source Reference |
|---|---|---|---|---|
| `human_games/immortal_game.pgn` | `human_games` | 45 | **Verified Historical Human** | **Independently Verified**: Adolf Anderssen vs. Lionel Kieseritzky (London, 21 June 1851). Canonical historical game documented across 170+ years of chess literature (Chessgames ID 1018910). |
| `human_games/opera_game_1858.pgn` | `human_games` | 33 | **Verified Historical Human** | **Independently Verified**: Paul Morphy vs. Duke Karl of Brunswick and Count Isouard (Paris Opera, 2 November 1858). Canonical historical game documented in standard game collections (Chessgames ID 1233404). |
| `engine_generated_games/tcec_engine_game.pgn` | `engine_generated_games` | 51 | **Unverified Demonstration Data** | **Unverified**: PGN headers state "TCEC Season 20 Superfinal", but the move sequence (26-ply Sicilian Najdorf Poisoned Pawn ending in `26. Bxa5`) diverges from the official TCEC Season 20 Superfinal games. Retained solely as an illustrative demonstration game for pipeline testing; **not an authenticated TCEC record**. |
| `unlabeled/casual_game.pgn` | `unlabeled` | 52 | **Synthetic Test Game** | **Programmatically Constructed**: Generated programmatically during project development to test parser and pipeline edge cases (e.g. unrated games). **Not a real-world game**. |
| **Total** | | **181** | **2 Verified / 2 Demonstration** | **8 player-game behavioral profiles** |

### Critical Separation: Verified Research Corpus vs. Demonstration Fixtures
A central finding of this provenance audit is that **only 2 of the 4 games (78 moves) represent authentic, verifiable human play**. 
- The 2 verified historical human games provide research-grade ground truth for 19th-century master play.
- The 2 remaining games (`tcec_engine_game.pgn` and `casual_game.pgn`) serve strictly as **demonstration and testing fixtures** to validate multi-game pipeline mechanics, batch error handling, and schema integrity.
- These demonstration fixtures must **never be merged into empirical research benchmarks or used for training machine learning detectors**.

### Provenance Tracking in Artifacts
Every record in both `move_level_features.csv` and `player_game_features.csv` retains its immutable `source_file`, `source_category`, and `game_id`. The pipeline manifest (`data/processed/dataset_manifest.json`) records file-by-file provenance audit status and disclaimers.

---

## 3. Data Quality, Missingness, and Schema Integrity

### Handling of Missing Values
Missing data was analyzed systematically:
- **Centipawn Loss (`centipawn_loss`)**: Missing in **5.5%** of moves. Crucially, this missingness is domain-correct: in forced checkmate sequences (e.g., plies 41–45 in the Immortal Game), evaluations are sentinel strings (`#+5`, `#+4`, `#+2`, `#+1`, `#+0`). The mathematical difference between a checkmate sentinel and a numeric centipawn score is undefined; converting forced mates to `0` would introduce severe bias.
- **Player Ratings (`player_elo`, `white_elo`, `black_elo`)**: Missing in historical 19th-century games prior to the Elo rating system. Preserved as empty strings (`""`) rather than artificial defaults.
- **Clock Annotations (`clock_seconds`)**: Present only when real `[%clk ...]` tags exist in the PGN. Never fabricated.

### Data Hygiene
- **Duplicate Plies**: `0` duplicate `(game_id, ply_number)` keys.
- **Duplicate Profiles**: `0` duplicate `(game_id, player_color)` keys.

---

## 4. Move-Level Behavioral Distributions

### 4.1 Centipawn Loss Distribution
Centipawn loss exhibits a strong **right-skewed distribution**:
- **Median Loss**: $13.0$ centipawns.
- **Mean Loss**: $37.9$ centipawns (elevated by the long tail of blunders).
- **Standard Deviation**: $65.8$ centipawns.
- Over **60%** of moves fall in the "Best" or "Excellent" categories ($\le 25$ cp loss).

### 4.2 Candidate Gap (`best_vs_second_cp_gap`)
- The difference between Stockfish's 1st and 2nd best candidate moves has a median of $32.0$ centipawns.
- High-gap positions ($> 150$ cp) represent **tactically forced positions** where only one good move exists. In these positions, both human masters and engines play the top move with high frequency.

### 4.3 Position Complexity (`legal_moves_count`)
- Legal move counts range from $2$ to $48$ (mean: $31.8$).
- Board complexity peaks in the late opening and early middlegame before piece exchanges simplify the board.

---

## 5. Game Phase Comparison

Features assessed on board positions before move execution reveal distinct behavioral patterns across phases:

| Game Phase | Moves Count | Mean Centipawn Loss | Median CP Loss | Top-1 Engine Agreement |
|---|---|---|---|---|
| **Opening** | 77 | 25.1 cp | 8.0 cp | 68.8% |
| **Middlegame** | 94 | 46.8 cp | 20.0 cp | 50.0% |
| **Endgame** | 10 | 18.0 cp | 0.0 cp | 70.0% |

### Key Insight:
Middlegame positions exhibit nearly **double the average centipawn loss** of opening moves ($46.8$ vs $25.1$ cp) and lower engine agreement ($50.0\%$ vs $68.8\%$). The opening phase benefits from established book theory and lower tactical volatility.

---

## 6. Outlier Analysis (IQR Rule)

Using the standard statistical $1.5 \times \text{IQR}$ threshold:
- $\text{Q}_{25} = 0.0$ cp, $\text{Q}_{75} = 45.0$ cp, $\text{IQR} = 45.0$ cp.
- **Outlier Upper Bound**: $45.0 + 1.5 \times 45.0 = 112.5$ cp.
- **Identified Outliers**: 17 moves ($9.9\%$ of scored moves) have centipawn loss $> 112.5$ cp, corresponding to tactical blunders and strategic inaccuracies.

---

## 7. Player-Game Behavioral Profiling

The player-game dataset aggregates performance into behavioral metrics:
- **Scored Moves Denominator**: Rate metrics (`top1_match_rate`, `top3_match_rate`, `blunder_rate`) strictly use `scored_moves` as the denominator.
- **Top-1 Match Rates**:
  - Illustrative engine game (`tcec_engine_game.pgn`, demonstration data): White ($73.1\%$), Black ($69.2\%$). Illustrates typical high engine concordance on complex tactical lines.
  - Verified human games (1851 / 1858): Winners exhibited high top-1 rates ($75.0\%$ Anderssen, $70.6\%$ Morphy) during sharp tactical king attacks, while losing defenders exhibited elevated blunder rates ($20.0\%$ Kieseritzky, $18.8\%$ Duke Karl / Count Isouard).
  - Synthetic development game (`casual_game.pgn`, demonstration data): White ($57.7\%$), Black ($53.8\%$), with moderate blunder rates ($7.7\%$ / $3.8\%$).

---

## 8. Feature Engineering Summary

| Feature Category | Features Added | Behavioral Rationale |
|---|---|---|
| **Board Context** | `legal_moves_count`, `is_capture`, `is_check`, `is_promotion` | Quantifies decision complexity and tactical forcing level |
| **Move Quality** | `move_quality` (Best, Excellent, Good, Inaccuracy, Mistake, Blunder, Forced Mate) | Categorizes player accuracy into discrete behavioral tiers |
| **Engine Dynamics** | `played_move_rank_top5`, `best_vs_second_cp_gap`, `is_forced_mate` | Measures alignment with engine candidates and choice difficulty |
| **Player-Game Aggregates** | `mean_centipawn_loss`, `std_centipawn_loss`, `top1_match_rate`, `blunder_rate`, phase metrics | Distills whole-game play into robust statistical profiles |

---

## 9. Scientific Scope and Limitations

1. **Strict Provenance Separation**:
   - Only `human_games/immortal_game.pgn` and `human_games/opera_game_1858.pgn` are independently authenticated historical games.
   - `engine_generated_games/tcec_engine_game.pgn` is unverified against official TCEC archives and is retained solely as an illustrative demonstration file.
   - `unlabeled/casual_game.pgn` is a programmatically constructed synthetic fixture created for testing.
   - Any future research benchmarking or empirical model training must strictly exclude illustrative/synthetic fixtures and ingest an authoritative, independently sourced corpus.
2. **Demonstration Dataset Size**: The current pipeline processes 4 demonstration games (181 plies). Statistical distributions serve to validate the analytical pipeline and feature engineering routines, not to draw population-level conclusions about human vs. AI chess play.
3. **Engine Alignment Does Not Equal Cheating**: High engine agreement is an index of move accuracy under engine evaluation, not proof of illicit assistance. In forced tactical sequences, book openings, and simplified endgames, human grandmasters routinely achieve $>70\%$ top-1 match rates.
4. **Binary Categories vs. Covert Assistance**: Macro comparisons between engine games and human games do not reflect real-world covert cheating, which typically features selective, critical-moment engine consultations rather than uniform engine play.
