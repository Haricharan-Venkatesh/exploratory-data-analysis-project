"""
Verify cleaning rules and exact counts before writing pipeline.
"""
import glob, os, chess.pgn
from collections import Counter, defaultdict
import numpy as np

raw_files = sorted(glob.glob("data/raw/*.pgn"))
seen_gids = set()

stats = {
    "total_raw_games": 0,
    "unique_games": 0,
    "duplicate_game_instances": 0,
    "zero_move_games": 0,
    "correspondence_games": 0,
    "valid_timed_games": 0,
    "total_raw_moves": 0,
    "moves_in_zero_move_games": 0,
    "moves_in_correspondence_games": 0,
    "moves_in_valid_games": 0,
    "human_moves": 0,
    "ai_moves": 0,
    "moves_with_clk": 0,
    "moves_without_clk": 0,
    "moves_with_negative_dt": 0,
    "moves_with_zero_dt": 0,
    "moves_with_positive_dt": 0,
}

for fpath in raw_files:
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            stats["total_raw_games"] += 1
            site = game.headers.get("Site", "")
            gid = site.split("/")[-1] if "lichess.org" in site else site
            
            if gid in seen_gids:
                stats["duplicate_game_instances"] += 1
                continue
            seen_gids.add(gid)
            stats["unique_games"] += 1
            
            moves = list(game.mainline())
            num_moves = len(moves)
            stats["total_raw_moves"] += num_moves
            
            if num_moves == 0:
                stats["zero_move_games"] += 1
                continue
                
            tc = game.headers.get("TimeControl", "")
            if tc == "-":
                stats["correspondence_games"] += 1
                stats["moves_in_correspondence_games"] += num_moves
                continue
                
            stats["valid_timed_games"] += 1
            stats["moves_in_valid_games"] += num_moves
            
            w_bot = (game.headers.get("WhiteTitle") == "BOT")
            b_bot = (game.headers.get("BlackTitle") == "BOT")
            
            base_s, inc_s = map(int, tc.split("+"))
            w_prev = max(base_s, 0)
            b_prev = max(base_s, 0)
            
            for idx, node in enumerate(moves):
                is_white = (idx % 2 == 0)
                is_bot = w_bot if is_white else b_bot
                if is_bot:
                    stats["ai_moves"] += 1
                else:
                    stats["human_moves"] += 1
                    
                clk = node.clock()
                if clk is not None:
                    stats["moves_with_clk"] += 1
                    prev = w_prev if is_white else b_prev
                    if idx < 2:
                        dt = 0.0 if (prev == 0 and clk > 0) else max(0.0, prev - clk)
                    else:
                        dt = prev + inc_s - clk
                        
                    if dt < 0:
                        stats["moves_with_negative_dt"] += 1
                    elif dt == 0.0:
                        stats["moves_with_zero_dt"] += 1
                    else:
                        stats["moves_with_positive_dt"] += 1
                        
                    if is_white:
                        w_prev = clk
                    else:
                        b_prev = clk
                else:
                    stats["moves_without_clk"] += 1

print("--- Exact Preprocessing Audit ---")
for k, v in stats.items():
    print(f"{k:30s}: {v:,d}")
