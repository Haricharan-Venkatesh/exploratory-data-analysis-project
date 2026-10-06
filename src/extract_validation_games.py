"""
Extract and verify real test games from data/raw/ across the 6 required categories:
1. Known HUMAN game
2. Known AI/BOT game
3. HUMAN-vs-BOT game
4. Bullet game
5. Blitz game
6. Rapid/Classical game
"""

import os
import sys
import chess
import chess.pgn
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

def extract_game(filename, filter_fn, max_scan=500):
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        count = 0
        while count < max_scan:
            count += 1
            game = chess.pgn.read_game(f)
            if game is None:
                break
            moves = list(game.mainline())
            has_clk = any(node.clock() is not None for node in moves)
            if len(moves) >= 20 and has_clk and filter_fn(game):
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
                return {
                    "headers": dict(game.headers),
                    "plies": len(moves),
                    "pgn": game.accept(exporter)
                }
    return None

def main():
    print("Finding 6 authentic test games from data/raw/...")
    
    # 1. Known Human Game (no bot titles)
    g1 = extract_game(
        "lichess_human_human_games.pgn",
        lambda g: "BOT" not in g.headers.get("WhiteTitle", "") and "BOT" not in g.headers.get("BlackTitle", "")
    )
    print(f"1. Human Game: {g1['headers'].get('White')} vs {g1['headers'].get('Black')} ({g1['plies']} plies, TC: {g1['headers'].get('TimeControl')})")
    
    # 2. Known AI/Bot Game (bot vs bot)
    g2 = extract_game(
        "lichess_bot_vs_bot_games.pgn",
        lambda g: True
    )
    print(f"2. Bot Game: {g2['headers'].get('White')} vs {g2['headers'].get('Black')} ({g2['plies']} plies, TC: {g2['headers'].get('TimeControl')})")
    
    # 3. Human vs Bot Game
    g3 = extract_game(
        "lichess_human_vs_bot_games.pgn",
        lambda g: True
    )
    print(f"3. Human vs Bot: {g3['headers'].get('White')} vs {g3['headers'].get('Black')} ({g3['plies']} plies, TC: {g3['headers'].get('TimeControl')})")
    
    # 4. Bullet Game (TimeControl starting with 60 or 120, or Event containing Bullet)
    g4 = extract_game(
        "lichess_human_human_games.pgn",
        lambda g: "Bullet" in g.headers.get("Event", "") or g.headers.get("TimeControl", "").startswith("60+") or g.headers.get("TimeControl", "").startswith("120+")
    )
    if not g4:
        g4 = extract_game("lichess_human_games_sample.pgn", lambda g: "Bullet" in g.headers.get("Event", ""))
    print(f"4. Bullet Game: {g4['headers'].get('White')} vs {g4['headers'].get('Black')} ({g4['plies']} plies, TC: {g4['headers'].get('TimeControl')})")
    
    # 5. Blitz Game (TimeControl starting with 180 or 300, or Event containing Blitz)
    g5 = extract_game(
        "lichess_human_human_games.pgn",
        lambda g: "Blitz" in g.headers.get("Event", "") or g.headers.get("TimeControl", "").startswith("180+")
    )
    print(f"5. Blitz Game: {g5['headers'].get('White')} vs {g5['headers'].get('Black')} ({g5['plies']} plies, TC: {g5['headers'].get('TimeControl')})")
    
    # 6. Rapid/Classical Game
    g6 = extract_game(
        "lichess_human_vs_bot_games_sample.pgn",
        lambda g: "Rapid" in g.headers.get("Event", "") or "Classical" in g.headers.get("Event", "") or int(g.headers.get("TimeControl", "0+0").split("+")[0]) >= 600
    )
    if not g6:
        g6 = extract_game(
            "lichess_human_games_sample.pgn",
            lambda g: "Rapid" in g.headers.get("Event", "") or "Classical" in g.headers.get("Event", "") or int(g.headers.get("TimeControl", "0+0").split("+")[0]) >= 600
        )
    print(f"6. Rapid/Classical Game: {g6['headers'].get('White')} vs {g6['headers'].get('Black')} ({g6['plies']} plies, TC: {g6['headers'].get('TimeControl')})")

    out_file = os.path.join(PROJECT_ROOT, "data", "test_validation_games.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "human_game": g1,
            "bot_game": g2,
            "human_vs_bot_game": g3,
            "bullet_game": g4,
            "blitz_game": g5,
            "rapid_classical_game": g6
        }, f, indent=2)
    print(f"Saved 6 real test games to {out_file}")

if __name__ == "__main__":
    main()
