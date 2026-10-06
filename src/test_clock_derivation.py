"""
Demonstration of Move-Time Derivation from [%clk] Annotations.

This script tests and verifies that move-level thinking time can be accurately
and reliably derived from standard PGN [%clk] annotations in our raw data.
"""

import os
import chess.pgn

def test_clock_derivation():
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "lichess_human_vs_bot_games.pgn")
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        game_count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            moves = list(game.mainline())
            if len(moves) < 8:
                continue
                
            game_count += 1
            w_name = game.headers.get("White")
            w_title = game.headers.get("WhiteTitle", "Human")
            w_elo = game.headers.get("WhiteElo", "?")
            b_name = game.headers.get("Black")
            b_title = game.headers.get("BlackTitle", "Human")
            b_elo = game.headers.get("BlackElo", "?")
            tc = game.headers.get("TimeControl", "")
            
            print(f"=== Game {game_count}: {w_name} ({w_title}, {w_elo}) vs {b_name} ({b_title}, {b_elo}) ===")
            print(f"Time Control: {tc} | Event: {game.headers.get('Event')} | Result: {game.headers.get('Result')}")
            
            try:
                base_s, inc_s = map(int, tc.split("+"))
            except Exception:
                base_s, inc_s = 0, 0
                
            w_prev_clk = base_s
            b_prev_clk = base_s
            
            board = game.board()
            for idx, node in enumerate(moves[:12]):
                is_white = (idx % 2 == 0)
                curr_clk = node.clock()
                player_title = w_title if is_white else b_title
                player_label = "BOT" if player_title == "BOT" else "HUMAN"
                color = "White" if is_white else "Black"
                move_num = (idx // 2) + 1
                
                prev_clk = w_prev_clk if is_white else b_prev_clk
                if curr_clk is not None and prev_clk is not None:
                    # Time spent = previous clock + increment - current clock
                    time_spent = round(prev_clk + inc_s - curr_clk, 2)
                else:
                    time_spent = None
                    
                if is_white:
                    w_prev_clk = curr_clk
                else:
                    b_prev_clk = curr_clk
                    
                san = board.san(node.move)
                board.push(node.move)
                print(f"  Move {move_num:2d}{'.' if is_white else '...'} [{color:5s} | {player_label:5s}]: {san:6s} | clk={str(curr_clk)+'s':8s} | move_time={time_spent}s")
                
            if game_count >= 2:
                break

if __name__ == "__main__":
    test_clock_derivation()
