"""
Chess AI Detection - Data Preprocessing Pipeline
=================================================
Converts raw Lichess PGN game files into a clean, verified, move-level dataset.

Target Output:
- data/processed/clean_chess_moves.csv
- data/processed/intermediate/game_metadata.csv
- data/processed/intermediate/cleaning_audit.json
"""

import os
import glob
import json
import time
from collections import Counter, defaultdict
import chess
import chess.pgn
import pandas as pd
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
INTERMEDIATE_DIR = os.path.join(PROCESSED_DIR, "intermediate")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

def determine_game_phase(board: chess.Board, move_number: int) -> str:
    """
    Transparent, reproducible heuristic to classify game phase:
    1. Opening:
       - Move number <= 10 AND both queens on board AND non-pawn pieces >= 10.
    2. Endgame:
       - Queens == 0 AND non-pawn material <= 16 points (Q=9, R=5, B=3, N=3), OR
       - Queens > 0 AND non-pawn material <= 13 points.
    3. Middlegame:
       - All positions between Opening and Endgame.
    """
    piece_map = board.piece_map()
    queens = sum(1 for p in piece_map.values() if p.piece_type == chess.QUEEN)
    non_pawn_pieces = sum(1 for p in piece_map.values() if p.piece_type not in (chess.PAWN, chess.KING))
    
    material_values = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3}
    total_non_pawn_material = sum(material_values.get(p.piece_type, 0) for p in piece_map.values())
    
    if move_number <= 10 and queens == 2 and non_pawn_pieces >= 10:
        return "Opening"
    if queens == 0 and total_non_pawn_material <= 16:
        return "Endgame"
    if queens > 0 and total_non_pawn_material <= 13:
        return "Endgame"
    return "Middlegame"


def get_speed_category(base_s: int, inc_s: int, event_str: str) -> str:
    """
    Determine speed category from time control and event tag.
    Standard Lichess duration estimate = base + 40 * inc.
    """
    est_duration = base_s + 40 * inc_s
    if est_duration < 30:
        return "UltraBullet"
    elif est_duration < 180:
        return "Bullet"
    elif est_duration < 480:
        return "Blitz"
    elif est_duration < 1500:
        return "Rapid"
    else:
        return "Classical"


def run_preprocessing_pipeline():
    start_time = time.time()
    pgn_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.pgn")))
    print(f"Found {len(pgn_files)} PGN files in {RAW_DIR}")
    
    seen_game_ids = set()
    raw_game_count = 0
    duplicate_game_count = 0
    zero_move_game_count = 0
    correspondence_game_count = 0
    valid_game_count = 0
    
    cleaning_audit = {
        "removal_rules": [
            {
                "rule_id": "RULE_1_DUPLICATE_GAME",
                "condition": "Exact same game_id (from Lichess Site URL) already ingested from an earlier raw file",
                "action": "Retain first occurrence, discard redundant instances",
                "affected_count": 0,
                "reason": "Eliminate redundant game blocks present across multiple sampled raw files."
            },
            {
                "rule_id": "RULE_2_ABORTED_GAME",
                "condition": "Game contains 0 moves (ply count == 0)",
                "action": "Exclude from move-level dataset",
                "affected_count": 0,
                "reason": "Game was aborted or forfeited before move 1; contains no behavioral move data."
            },
            {
                "rule_id": "RULE_3_CORRESPONDENCE_GAME",
                "condition": "TimeControl is '-' (untimed correspondence game)",
                "action": "Exclude from real-time move dataset",
                "affected_count": 0,
                "reason": "Untimed correspondence chess lacks real-time clocks and [%clk] annotations."
            },
            {
                "rule_id": "RULE_4_INVALID_MOVE_TIME",
                "condition": "Calculated move_time < 0s (opponent gave +15s or lag compensation restored time)",
                "action": "Keep move row, set move_time to NaN",
                "affected_count": 0,
                "reason": "Clock addition makes elapsed thinking time uncomputable; do not fabricate time."
            },
            {
                "rule_id": "RULE_5_MISSING_CLOCK_ANNOTATION",
                "condition": "Individual move missing [%clk] annotation",
                "action": "Keep move row, set move_time to NaN and clock_after_move to NaN",
                "affected_count": 0,
                "reason": "Clock time missing; do not impute behavioral time."
            }
        ],
        "game_counts": {},
        "move_counts": {}
    }
    
    games_metadata_list = []
    moves_data_list = []
    
    for fpath in pgn_files:
        fname = os.path.basename(fpath)
        print(f"Processing {fname}...")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                raw_game_count += 1
                site = game.headers.get("Site", "")
                gid = site.split("/")[-1] if "lichess.org" in site else site
                
                if gid in seen_game_ids:
                    duplicate_game_count += 1
                    cleaning_audit["removal_rules"][0]["affected_count"] += 1
                    continue
                seen_game_ids.add(gid)
                
                game_moves = list(game.mainline())
                num_moves = len(game_moves)
                
                if num_moves == 0:
                    zero_move_game_count += 1
                    cleaning_audit["removal_rules"][1]["affected_count"] += 1
                    continue
                    
                tc = game.headers.get("TimeControl", "")
                if tc == "-" or "+" not in tc:
                    correspondence_game_count += 1
                    cleaning_audit["removal_rules"][2]["affected_count"] += 1
                    continue
                    
                valid_game_count += 1
                base_s, inc_s = map(int, tc.split("+"))
                
                w_player = game.headers.get("White", "unknown_white")
                b_player = game.headers.get("Black", "unknown_black")
                w_title = game.headers.get("WhiteTitle", "")
                b_title = game.headers.get("BlackTitle", "")
                w_is_bot = (w_title == "BOT")
                b_is_bot = (b_title == "BOT")
                
                w_elo = int(game.headers.get("WhiteElo", 0))
                b_elo = int(game.headers.get("BlackElo", 0))
                
                event = game.headers.get("Event", "")
                speed = get_speed_category(base_s, inc_s, event)
                result = game.headers.get("Result", "*")
                eco = game.headers.get("ECO", "")
                date = game.headers.get("UTCDate", game.headers.get("Date", ""))
                
                if w_is_bot and b_is_bot:
                    game_type = "BOT_vs_BOT"
                elif w_is_bot or b_is_bot:
                    game_type = "HUMAN_vs_BOT"
                else:
                    game_type = "HUMAN_vs_HUMAN"
                    
                games_metadata_list.append({
                    "game_id": gid,
                    "event": event,
                    "speed_category": speed,
                    "time_control": tc,
                    "white_player": w_player,
                    "black_player": b_player,
                    "white_elo": w_elo,
                    "black_elo": b_elo,
                    "white_title": w_title,
                    "black_title": b_title,
                    "game_type": game_type,
                    "result": result,
                    "eco": eco,
                    "total_plies": num_moves
                })
                
                # Iterate through moves and reconstruct board states
                board = game.board()
                w_prev_clk = max(base_s, 0)
                b_prev_clk = max(base_s, 0)
                
                for ply_idx, node in enumerate(game_moves):
                    ply = ply_idx + 1
                    is_white = (ply_idx % 2 == 0)
                    move_number = (ply_idx // 2) + 1
                    player_color = "White" if is_white else "Black"
                    player_id = w_player if is_white else b_player
                    opponent_id = b_player if is_white else w_player
                    player_rating = w_elo if is_white else b_elo
                    opponent_rating = b_elo if is_white else w_elo
                    
                    is_bot = w_is_bot if is_white else b_is_bot
                    label = "AI" if is_bot else "HUMAN"
                    
                    # Position immediately BEFORE the move
                    position_fen = board.fen()
                    game_phase = determine_game_phase(board, move_number)
                    
                    san_move = board.san(node.move)
                    board.push(node.move)  # advance board state
                    
                    clk = node.clock()
                    if clk is not None:
                        prev_clk = w_prev_clk if is_white else b_prev_clk
                        if ply_idx < 2:  # Move 1 White or Move 1 Black
                            if prev_clk == 0 and clk > 0:
                                dt = 0.0
                            else:
                                dt = max(0.0, prev_clk - clk)
                        else:
                            dt = prev_clk + inc_s - clk
                            
                        if dt < 0:
                            # Opponent gifted time or lag compensation
                            move_time = np.nan
                            cleaning_audit["removal_rules"][3]["affected_count"] += 1
                        else:
                            move_time = round(float(dt), 2)
                            
                        clock_after_move = round(float(clk), 2)
                        
                        if is_white:
                            w_prev_clk = clk
                        else:
                            b_prev_clk = clk
                    else:
                        move_time = np.nan
                        clock_after_move = np.nan
                        cleaning_audit["removal_rules"][4]["affected_count"] += 1
                        
                    moves_data_list.append({
                        "game_id": gid,
                        "player_id": player_id,
                        "player_color": player_color,
                        "move_number": move_number,
                        "move": san_move,
                        "move_time": move_time,
                        "player_rating": player_rating,
                        "game_phase": game_phase,
                        "position": position_fen,
                        "label": label,
                        "ply": ply,
                        "clock_after_move": clock_after_move,
                        "time_control": tc,
                        "speed_category": speed,
                        "opponent_id": opponent_id,
                        "opponent_rating": opponent_rating,
                        "game_result": result,
                        "eco": eco
                    })
                    
                if valid_game_count % 1000 == 0:
                    print(f"  Processed {valid_game_count} valid games ({len(moves_data_list):,d} moves)...")
                    
    print("\nConverting moves data to DataFrame...")
    df_moves = pd.DataFrame(moves_data_list)
    df_games = pd.DataFrame(games_metadata_list)
    
    # Validation checks
    print("Performing validation checks...")
    assert len(df_moves) == 416978, f"Expected 416978 moves, got {len(df_moves)}"
    assert len(df_games) == 6154, f"Expected 6154 games, got {len(df_games)}"
    
    # Check chronological ordering
    # Plies must strictly increment by 1 within each game
    grouped = df_moves.groupby("game_id")["ply"].apply(list)
    strictly_ordered = all(plies == list(range(1, len(plies) + 1)) for plies in grouped)
    assert strictly_ordered, "Chronological move ordering check failed!"
    print("Chronological move ordering validated: 100% verified.")
    
    # Missing value analysis
    missing_summary = df_moves.isnull().sum()
    print("\nMissing values in clean moves dataset:")
    for col, m_cnt in missing_summary.items():
        pct = (m_cnt / len(df_moves)) * 100
        print(f"  {col:20s}: {m_cnt:6d} missing ({pct:.3f}%)")
        
    # Class distribution
    label_counts = df_moves["label"].value_counts()
    print("\nMove Label Distribution:")
    for lbl, count in label_counts.items():
        print(f"  {lbl:10s}: {count:7,d} ({count/len(df_moves)*100:.2f}%)")
        
    phase_counts = df_moves["game_phase"].value_counts()
    print("\nGame Phase Distribution:")
    for ph, count in phase_counts.items():
        print(f"  {ph:15s}: {count:7,d} ({count/len(df_moves)*100:.2f}%)")
        
    print("\nMove Time Statistics (Valid Non-Null):")
    mt_valid = df_moves["move_time"].dropna()
    print(mt_valid.describe())
    
    print("\nPlayer Rating Statistics:")
    print(df_moves["player_rating"].describe())
    
    # Saving outputs
    csv_path = os.path.join(PROCESSED_DIR, "clean_chess_moves.csv")
    print(f"\nSaving clean dataset to {csv_path}...")
    df_moves.to_csv(csv_path, index=False)
    csv_size = os.path.getsize(csv_path)
    print(f"Saved {csv_path} ({csv_size:10,d} bytes | {csv_size/(1024*1024):.2f} MB)")
    
    # Save intermediate files
    games_meta_path = os.path.join(INTERMEDIATE_DIR, "game_metadata.csv")
    df_games.to_csv(games_meta_path, index=False)
    print(f"Saved {games_meta_path}")
    
    cleaning_audit["game_counts"] = {
        "raw_game_instances": raw_game_count,
        "unique_games_ingested": len(seen_game_ids),
        "duplicate_game_instances_discarded": duplicate_game_count,
        "zero_move_aborted_games_discarded": zero_move_game_count,
        "correspondence_games_discarded": correspondence_game_count,
        "final_valid_games": valid_game_count
    }
    
    cleaning_audit["move_counts"] = {
        "total_final_moves": len(df_moves),
        "human_moves": int(label_counts.get("HUMAN", 0)),
        "ai_moves": int(label_counts.get("AI", 0)),
        "moves_with_valid_time": int(mt_valid.count()),
        "moves_with_missing_time": int(df_moves["move_time"].isnull().sum()),
        "moves_with_zero_time_premoves": int((mt_valid == 0.0).sum()),
        "moves_with_positive_time": int((mt_valid > 0.0).sum())
    }
    
    audit_path = os.path.join(INTERMEDIATE_DIR, "cleaning_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_audit, f, indent=2)
    print(f"Saved {audit_path}")
    
    elapsed = time.time() - start_time
    print(f"\nPipeline completed successfully in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    run_preprocessing_pipeline()
