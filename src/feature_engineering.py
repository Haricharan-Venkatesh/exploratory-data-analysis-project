"""
Feature Engineering Pipeline for Human-AI Chess Move Detection
==============================================================
Constructs temporally valid, leakage-free behavioral features at:
- Move level
- Rolling-window level (W=10 moves per player)
- Cumulative game context level

Outputs:
- data/processed/chess_behavioral_features.csv
"""

import os
import time
import numpy as np
import pandas as pd

DATA_PATH = "data/processed/clean_chess_moves.csv"
OUTPUT_PATH = "data/processed/chess_behavioral_features.csv"

def extract_features():
    start_time = time.time()
    print("=== Loading Clean Move Dataset ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,d} rows and {df.shape[1]} columns.")
    
    # Fill missing move_time (41 rows due to +15s gifts) with 0.0 for rolling calculations,
    # but create a missingness indicator
    df["move_time_missing"] = df["move_time"].isnull().astype(int)
    # Forward fill or set to 0.0 for uncomputable clock deltas
    clean_move_time = df["move_time"].fillna(0.0)
    df["clean_move_time"] = clean_move_time
    
    # Parse base time and increment safely
    def parse_tc(tc):
        if "+" in tc:
            parts = tc.split("+")
            return int(parts[0]), int(parts[1])
        return 180, 0
        
    tcs = df["time_control"].apply(parse_tc)
    df["base_time"] = [t[0] for t in tcs]
    df["inc_time"] = [t[1] for t in tcs]
    
    print("Extracting move-level and rolling features...")
    # Group by (game_id, player_color) for per-player sequential moves
    grouped = df.groupby(["game_id", "player_color"])
    
    # -------------------------------------------------------------
    # 1. MOVE-LEVEL FEATURES
    # -------------------------------------------------------------
    # Relative move time (budget-normalized)
    df["relative_move_time"] = df["clean_move_time"] / np.maximum(1.0, df["base_time"] / 40.0)
    
    # Time spent ratio: move_time / previous_available_clock
    prev_clk = grouped["clock_after_move"].shift(1)
    is_move_1 = prev_clk.isna()
    init_clk = np.where(df["base_time"] > 0, df["base_time"], 3.0)
    prev_available_clk = np.where(is_move_1, init_clk, prev_clk + df["inc_time"])
    df["time_spent_ratio"] = np.clip(
        df["clean_move_time"] / np.maximum(1.0, prev_available_clk), 0.0, 1.0
    )
    
    # Instantaneous move flags
    df["is_premove"] = (df["clean_move_time"] == 0.0).astype(int)
    df["tank_move_indicator"] = (df["clean_move_time"] > 30.0).astype(int)
    
    # Clock remaining ratio
    df["clock_remaining_ratio"] = np.clip(
        df["clock_after_move"] / np.maximum(1.0, df["base_time"]), 0.0, 5.0
    )
    
    # Rating and progress controls
    df["player_rating_normalized"] = (df["player_rating"] - 1500.0) / 400.0
    df["opponent_rating_difference"] = df["player_rating"] - df["opponent_rating"]
    df["move_number_normalized"] = np.clip(df["move_number"] / 50.0, 0.0, 2.0)
    
    # -------------------------------------------------------------
    # 2. ROLLING-WINDOW FEATURES (W = 10 moves for active player)
    # -------------------------------------------------------------
    roll = grouped["clean_move_time"].rolling(10, min_periods=1)
    roll_mean = roll.mean().reset_index(level=[0, 1], drop=True).sort_index()
    roll_std = roll.std().reset_index(level=[0, 1], drop=True).sort_index().fillna(0.0)
    
    df["move_time_mean_10"] = roll_mean
    df["move_time_std_10"] = roll_std
    df["move_time_cv_10"] = np.where(roll_mean > 0.05, roll_std / roll_mean, 0.0)
    
    # Rolling pre-move rate
    roll_pmr = grouped["is_premove"].rolling(10, min_periods=1).mean().reset_index(level=[0, 1], drop=True).sort_index()
    df["premove_rate_10"] = roll_pmr
    
    # Rolling tank count
    roll_tank = grouped["tank_move_indicator"].rolling(10, min_periods=1).sum().reset_index(level=[0, 1], drop=True).sort_index()
    df["tank_move_count_10"] = roll_tank.astype(int)
    
    # Rolling skewness (min 4 moves)
    roll_skew = grouped["clean_move_time"].rolling(10, min_periods=4).skew().reset_index(level=[0, 1], drop=True).sort_index().fillna(0.0)
    df["move_time_skewness_10"] = np.clip(roll_skew, -5.0, 5.0)
    
    # Consecutive fast moves (<= 0.5s)
    is_fast = df["clean_move_time"] <= 0.5
    df["consecutive_fast_moves"] = grouped["clean_move_time"].transform(
        lambda s: (s <= 0.5).groupby((~(s <= 0.5)).cumsum()).cumsum()
    )
    
    # Time pressure jitter (< 15s remaining)
    is_tp = df["clock_after_move"] < 15.0
    df["in_time_pressure"] = is_tp.astype(int)
    
    tp_df = df[is_tp].copy()
    if len(tp_df) > 0:
        tp_std = tp_df.groupby(["game_id", "player_color"])["clean_move_time"].rolling(5, min_periods=2).std()
        tp_jitter_series = tp_std.reset_index(level=[0, 1], drop=True).sort_index().fillna(0.0)
        df["time_pressure_jitter"] = 0.0
        df.loc[tp_df.index, "time_pressure_jitter"] = tp_jitter_series
    else:
        df["time_pressure_jitter"] = 0.0
        
    # -------------------------------------------------------------
    # 3. CUMULATIVE & CONTEXT FEATURES (No Lookahead)
    # -------------------------------------------------------------
    # Cumulative stats up to move k
    cum_count = grouped.cumcount() + 1
    cum_sum = grouped["clean_move_time"].cumsum()
    df["cumulative_mean_move_time"] = cum_sum / cum_count
    
    # Cumulative variance: E[X^2] - (E[X])^2
    cum_sum_sq = grouped["clean_move_time"].transform(lambda s: (s**2).cumsum())
    cum_var = np.maximum(0.0, (cum_sum_sq / cum_count) - (df["cumulative_mean_move_time"]**2))
    df["cumulative_move_time_std"] = np.sqrt(cum_var)
    
    # Cumulative pre-move rate
    cum_premoves = grouped["is_premove"].cumsum()
    df["cumulative_premove_rate"] = cum_premoves / cum_count
    
    # Phase deliberation ratio: Middlegame cumulative mean / Opening mean
    # Fully vectorized per (game_id, player_color)
    is_open = (df["game_phase"] == "Opening")
    is_mid = (df["game_phase"] == "Middlegame")
    is_end = (df["game_phase"] == "Endgame")
    
    open_cumsum = df["clean_move_time"].where(is_open, 0.0).groupby([df["game_id"], df["player_color"]]).cumsum()
    open_count = is_open.astype(int).groupby([df["game_id"], df["player_color"]]).cumsum()
    open_mean = np.where(open_count > 0, open_cumsum / open_count, 2.0)
    
    # Forward fill the opening mean across the game
    df["_open_mean_raw"] = np.where(is_open, open_mean, np.nan)
    open_mean_ffill = df.groupby(["game_id", "player_color"])["_open_mean_raw"].ffill().fillna(2.0)
    
    mid_cumsum = df["clean_move_time"].where(is_mid, 0.0).groupby([df["game_id"], df["player_color"]]).cumsum()
    mid_count = is_mid.astype(int).groupby([df["game_id"], df["player_color"]]).cumsum()
    mid_mean = np.where(mid_count > 0, mid_cumsum / mid_count, open_mean_ffill)
    
    df["phase_deliberation_ratio"] = np.clip(
        np.where(is_open, 1.0, mid_mean / np.maximum(0.5, open_mean_ffill)),
        0.0, 20.0
    )
    df.drop(columns=["_open_mean_raw"], inplace=True)
    
    # Current Endgame Clock Ratio (Leakage-safe)
    df["is_endgame"] = is_end.astype(int)
    df["current_endgame_clock_ratio"] = np.where(
        is_end,
        np.clip(df["clock_after_move"] / np.maximum(1.0, df["base_time"]), 0.0, 5.0),
        0.0
    )
    
    # One-hot contextual flags
    df["game_phase_opening"] = is_open.astype(int)
    df["game_phase_middlegame"] = is_mid.astype(int)
    df["game_phase_endgame"] = is_end.astype(int)
    
    df["speed_is_bullet"] = (df["speed_category"] == "Bullet").astype(int)
    df["speed_is_blitz"] = (df["speed_category"] == "Blitz").astype(int)
    df["speed_is_rapid"] = (df["speed_category"] == "Rapid").astype(int)
    df["speed_is_classical"] = (df["speed_category"] == "Classical").astype(int)
    
    # Target encoding
    df["target"] = (df["label"] == "AI").astype(int)
    
    # Clean up temporary calculation columns
    df.drop(columns=["clean_move_time", "base_time", "inc_time"], inplace=True)
    
    print("\n--- Feature Extraction Complete ---")
    print(f"Total Columns: {df.shape[1]}")
    
    # Validation checks
    assert df["move_time_cv_10"].isna().sum() == 0, "move_time_cv_10 has NaNs"
    assert not np.isinf(df["move_time_cv_10"]).any(), "move_time_cv_10 has infs"
    assert (df["time_spent_ratio"] >= 0.0).all() and (df["time_spent_ratio"] <= 1.0).all(), "time_spent_ratio out of [0, 1]"
    assert (df["premove_rate_10"] >= 0.0).all() and (df["premove_rate_10"] <= 1.0).all(), "premove_rate_10 out of [0, 1]"
    assert (df["consecutive_fast_moves"] >= 0).all(), "consecutive_fast_moves negative"
    
    print("Saving feature dataset to", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Saved {OUTPUT_PATH} ({file_size_mb:.2f} MB)")
    print(f"Total time elapsed: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    extract_features()
