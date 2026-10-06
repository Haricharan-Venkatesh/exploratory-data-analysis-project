"""
Comprehensive Raw Data Inspector for Chess PGN files in data/raw/
Analyzes all files, detects duplicates, parses games, checks clocks,
identifies anomalies, and reports findings.
"""

import os
import glob
import re
from collections import Counter, defaultdict
import chess
import chess.pgn

RAW_DIR = "data/raw"

def inspect_all_raw_files():
    pgn_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.pgn")))
    print(f"Found {len(pgn_files)} PGN files in {RAW_DIR}:")
    
    file_stats = {}
    game_registry = defaultdict(list)  # game_id -> list of (filename, game_idx)
    
    for fpath in pgn_files:
        fname = os.path.basename(fpath)
        fsize = os.path.getsize(fpath)
        g_count = 0
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            while True:
                headers = chess.pgn.read_headers(f)
                if headers is None:
                    break
                g_count += 1
                site = headers.get("Site", f"game_{g_count}")
                gid = site.split("/")[-1] if "lichess.org" in site else site
                game_registry[gid].append((fname, g_count))
        file_stats[fname] = {"size": fsize, "games": g_count}
        print(f"  {fname:35s}: {fsize:10,d} bytes | {g_count:5,d} games")
        
    total_raw_games = sum(s["games"] for s in file_stats.values())
    unique_games_count = len(game_registry)
    duplicate_ids = {k: v for k, v in game_registry.items() if len(v) > 1}
    
    print("\n--- Summary of Game Duplication ---")
    print(f"Total game instances across all raw files: {total_raw_games}")
    print(f"Unique game IDs: {unique_games_count}")
    print(f"Duplicated game IDs: {len(duplicate_ids)}")
    print(f"Total redundant game instances: {total_raw_games - unique_games_count}")
    
    # Detailed scan across unique games
    print("\n--- Detailed Scan across Unique Games ---")
    
    seen_ids = set()
    scanned_games = 0
    games_with_zero_moves = 0
    games_with_missing_tc = 0
    games_with_non_standard_tc = 0
    games_with_missing_elo = 0
    games_without_any_clk = 0
    
    total_moves = 0
    moves_with_clk = 0
    moves_without_clk = 0
    illegal_moves = 0
    
    game_types = Counter() # HUMAN_vs_HUMAN, HUMAN_vs_BOT, BOT_vs_BOT
    tc_distribution = Counter()
    
    move_times = []
    negative_move_times = 0
    zero_move_times = 0
    normal_move_times = 0
    
    human_moves = 0
    bot_moves = 0
    
    for fpath in pgn_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                site = game.headers.get("Site", "")
                gid = site.split("/")[-1] if "lichess.org" in site else site
                if gid in seen_ids:
                    continue
                seen_ids.add(gid)
                scanned_games += 1
                
                # Check headers
                w_title = game.headers.get("WhiteTitle")
                b_title = game.headers.get("BlackTitle")
                w_is_bot = (w_title == "BOT")
                b_is_bot = (b_title == "BOT")
                
                if w_is_bot and b_is_bot:
                    gtype = "BOT_vs_BOT"
                elif w_is_bot or b_is_bot:
                    gtype = "HUMAN_vs_BOT"
                else:
                    gtype = "HUMAN_vs_HUMAN"
                game_types[gtype] += 1
                
                w_elo = game.headers.get("WhiteElo")
                b_elo = game.headers.get("BlackElo")
                if not (w_elo and w_elo.isdigit()) or not (b_elo and b_elo.isdigit()):
                    games_with_missing_elo += 1
                    
                tc = game.headers.get("TimeControl", "")
                tc_distribution[tc] += 1
                
                has_tc = False
                base_s, inc_s = 0, 0
                if "+" in tc:
                    parts = tc.split("+")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        base_s, inc_s = int(parts[0]), int(parts[1])
                        has_tc = True
                    else:
                        games_with_non_standard_tc += 1
                else:
                    if tc == "-":
                        games_with_non_standard_tc += 1
                    else:
                        games_with_missing_tc += 1
                        
                # Scan mainline moves
                board = game.board()
                w_prev_clk = max(base_s, 0)
                b_prev_clk = max(base_s, 0)
                
                game_moves = list(game.mainline())
                if len(game_moves) == 0:
                    games_with_zero_moves += 1
                    continue
                    
                game_has_any_clk = False
                
                for idx, node in enumerate(game_moves):
                    total_moves += 1
                    is_white = (idx % 2 == 0)
                    player_label = "AI" if (w_is_bot if is_white else b_is_bot) else "HUMAN"
                    if player_label == "AI":
                        bot_moves += 1
                    else:
                        human_moves += 1
                        
                    # Verify legality by pushing to board
                    try:
                        board.push(node.move)
                    except Exception as e:
                        illegal_moves += 1
                        
                    clk = node.clock()
                    if clk is not None:
                        moves_with_clk += 1
                        game_has_any_clk = True
                        
                        if has_tc:
                            prev_clk = w_prev_clk if is_white else b_prev_clk
                            if idx < 2:  # Move 1
                                if prev_clk == 0 and clk > 0:
                                    dt = 0.0
                                else:
                                    dt = max(0.0, prev_clk - clk)
                            else:
                                dt = prev_clk + inc_s - clk
                                
                            if dt < 0:
                                negative_move_times += 1
                            elif dt == 0.0:
                                zero_move_times += 1
                            else:
                                normal_move_times += 1
                                
                            move_times.append(dt)
                            
                        if is_white:
                            w_prev_clk = clk
                        else:
                            b_prev_clk = clk
                    else:
                        moves_without_clk += 1
                        
                if not game_has_any_clk:
                    games_without_any_clk += 1
                    
                if scanned_games % 1000 == 0:
                    print(f"  Scanned {scanned_games}/{unique_games_count} unique games...")
                    
    print("\n================ INSPECTION RESULTS ================")
    print(f"Unique games scanned: {scanned_games}")
    print(f"Game Types:")
    for gt, c in game_types.items():
        print(f"  {gt:20s}: {c:5d} games ({c/scanned_games*100:.1f}%)")
    print(f"Zero-move games (aborted): {games_with_zero_moves}")
    print(f"Games without standard TimeControl: {games_with_non_standard_tc}")
    print(f"Games with missing Elo: {games_with_missing_elo}")
    print(f"Games without any clock annotations: {games_without_any_clk}")
    print(f"\nMove Statistics:")
    print(f"Total moves: {total_moves}")
    print(f"Illegal/unparseable moves: {illegal_moves}")
    print(f"Human moves: {human_moves} ({human_moves/total_moves*100:.2f}%)")
    print(f"AI/Bot moves: {bot_moves} ({bot_moves/total_moves*100:.2f}%)")
    print(f"Moves with clock annotation: {moves_with_clk} ({moves_with_clk/total_moves*100:.2f}%)")
    print(f"Moves without clock annotation: {moves_without_clk} ({moves_without_clk/total_moves*100:.2f}%)")
    print(f"\nMove Time Breakdown (for moves with valid TimeControl & Clocks):")
    print(f"Normal move times (>0s): {normal_move_times} ({normal_move_times/len(move_times)*100:.2f}%)")
    print(f"Zero move times (=0s / pre-moves): {zero_move_times} ({zero_move_times/len(move_times)*100:.2f}%)")
    print(f"Negative move times (<0s / clock additions): {negative_move_times} ({negative_move_times/len(move_times)*100:.3f}%)")
    if move_times:
        valid_pos = [t for t in move_times if t >= 0]
        print(f"Valid move times count: {len(valid_pos)}")
        print(f"Move time quantiles: min={min(valid_pos):.1f}s, p25={sorted(valid_pos)[int(len(valid_pos)*0.25)]:.1f}s, median={sorted(valid_pos)[int(len(valid_pos)*0.5)]:.1f}s, p75={sorted(valid_pos)[int(len(valid_pos)*0.75)]:.1f}s, p95={sorted(valid_pos)[int(len(valid_pos)*0.95)]:.1f}s, max={max(valid_pos):.1f}s")

if __name__ == "__main__":
    inspect_all_raw_files()
