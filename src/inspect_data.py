"""
Inspect the downloaded Lichess PGN files.

This script analyzes all files in data/raw/ and reports:
- File format and size
- Number of games
- Number of moves
- Available PGN header tags
- Clock annotation presence
- Human/bot identity information
- Player ratings distribution
- Sample games

Does NOT modify any raw files.
"""

import os
import re
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ---- PGN Parsing Helpers ---------------------------------------------------------

def parse_pgn_headers(game_block: str) -> dict:
    """Extract all PGN header tags from a game block."""
    headers = {}
    for line in game_block.splitlines():
        if line.startswith("[") and "]" in line:
            match = re.match(r'\[(\w+)\s+"(.*)"\]', line)
            if match:
                headers[match.group(1)] = match.group(2)
    return headers


def count_moves(pgn_block: str) -> int:
    """Count the number of plies (half-moves) in a PGN game."""
    # Find the movetext section (after headers)
    lines = pgn_block.splitlines()
    movetext_lines = [l for l in lines if not l.startswith("[") and l.strip()]
    movetext = " ".join(movetext_lines)
    # Count move numbers like "1." "2." etc
    moves = re.findall(r'\d+\.', movetext)
    if not moves:
        return 0
    last_move = int(moves[-1].rstrip('.'))
    # Rough estimate: count actual move notations
    san_moves = re.findall(r'\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[KQRBN])?\+?\#?|O-O-O|O-O', movetext)
    return len(san_moves)


def count_clock_annotations(pgn_block: str) -> int:
    """Count number of [%clk] annotations in a game."""
    return len(re.findall(r'\[%clk', pgn_block))


def has_eval_annotations(pgn_block: str) -> bool:
    """Check if the game has [%eval] engine annotations."""
    return "[%eval" in pgn_block


def split_games(pgn_text: str) -> list:
    """Split PGN text into individual game blocks."""
    games = []
    parts = re.split(r'\n\n(?=\[Event )', pgn_text)
    for part in parts:
        part = part.strip()
        if part.startswith("[Event "):
            games.append(part)
    return games


# ---- File Analysis ---------------------------------------------------------------

def analyze_file(filepath: str) -> dict:
    """Analyze a single PGN file and return statistics."""
    stats = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "size_mb": os.path.getsize(filepath) / 1024 / 1024,
        "games": [],
        "tags_found": Counter(),
        "ratings_white": [],
        "ratings_black": [],
        "has_clocks": 0,
        "has_evals": 0,
        "bot_players": Counter(),
        "human_players": Counter(),
        "time_controls": Counter(),
        "events": Counter(),
        "terminations": Counter(),
        "openings": Counter(),
        "total_moves": 0,
        "total_clk_annotations": 0,
    }

    print(f"\nAnalyzing: {os.path.basename(filepath)} ({stats['size_mb']:.1f} MB)")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    games = split_games(text)
    stats["game_count"] = len(games)
    print(f"  Found {len(games):,} games")

    for i, game in enumerate(games[:500]):  # Analyze up to 500 games
        headers = parse_pgn_headers(game)
        stats["tags_found"].update(headers.keys())

        # Ratings
        try:
            if "WhiteElo" in headers and headers["WhiteElo"] != "?":
                stats["ratings_white"].append(int(headers["WhiteElo"]))
        except:
            pass
        try:
            if "BlackElo" in headers and headers["BlackElo"] != "?":
                stats["ratings_black"].append(int(headers["BlackElo"]))
        except:
            pass

        # Bot detection
        white_title = headers.get("WhiteTitle", "")
        black_title = headers.get("BlackTitle", "")
        if white_title == "BOT":
            stats["bot_players"][headers.get("White", "?")] += 1
        else:
            stats["human_players"][headers.get("White", "?")] += 1
        if black_title == "BOT":
            stats["bot_players"][headers.get("Black", "?")] += 1
        else:
            stats["human_players"][headers.get("Black", "?")] += 1

        # Clock & eval
        clk_count = count_clock_annotations(game)
        stats["total_clk_annotations"] += clk_count
        if clk_count > 0:
            stats["has_clocks"] += 1
        if has_eval_annotations(game):
            stats["has_evals"] += 1

        # Other metadata
        stats["time_controls"][headers.get("TimeControl", "?")] += 1
        stats["events"][headers.get("Event", "?")] += 1
        stats["terminations"][headers.get("Termination", "?")] += 1
        if "Opening" in headers:
            stats["openings"][headers.get("Opening", "?")] += 1

        # Moves
        n_moves = count_moves(game)
        stats["total_moves"] += n_moves

    return stats


def print_report(stats: dict):
    """Print a human-readable report for a file."""
    n_analyzed = min(stats["game_count"], 500)
    print(f"\n{'='*60}")
    print(f"FILE: {stats['filename']}")
    print(f"{'='*60}")
    print(f"  Size: {stats['size_mb']:.2f} MB")
    print(f"  Total games: {stats['game_count']:,}")
    print(f"  Games analyzed (sample): {n_analyzed:,}")
    print(f"  Estimated total moves: {stats['total_moves']:,} (from sample)")
    print()
    print("  LABELS:")
    print(f"    Bot players (identified by BOT title): {sum(stats['bot_players'].values()):,}")
    print(f"    Human players (no BOT title): {sum(stats['human_players'].values()):,}")
    if stats["bot_players"]:
        top_bots = stats["bot_players"].most_common(5)
        print(f"    Top bot usernames: {top_bots}")
    print()
    print("  CLOCK DATA:")
    print(f"    Games with [%clk] annotations: {stats['has_clocks']:,} / {n_analyzed}")
    print(f"    Total [%clk] annotations: {stats['total_clk_annotations']:,}")
    print(f"    Games with [%eval] annotations: {stats['has_evals']:,} / {n_analyzed}")
    print()
    print("  PLAYER RATINGS:")
    if stats["ratings_white"]:
        print(f"    White Elo: min={min(stats['ratings_white'])}, "
              f"max={max(stats['ratings_white'])}, "
              f"avg={sum(stats['ratings_white'])//len(stats['ratings_white'])}")
    if stats["ratings_black"]:
        print(f"    Black Elo: min={min(stats['ratings_black'])}, "
              f"max={max(stats['ratings_black'])}, "
              f"avg={sum(stats['ratings_black'])//len(stats['ratings_black'])}")
    print()
    print("  AVAILABLE TAGS:")
    for tag, count in sorted(stats["tags_found"].items()):
        print(f"    [{tag}]: {count} games")
    print()
    print("  TOP TIME CONTROLS:")
    for tc, count in stats["time_controls"].most_common(5):
        print(f"    {tc}: {count}")
    print()
    print("  SAMPLE GAMES (first 2):")


def print_sample_games(filepath: str, n: int = 2):
    """Print sample games from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    games = split_games(text)
    for i, game in enumerate(games[:n]):
        print(f"\n  --- Sample Game {i+1} ---")
        for line in game.splitlines()[:25]:
            if line.strip():
                print(f"    {line}")
        if len(game.splitlines()) > 25:
            print(f"    ... (truncated)")


# ---- Main ------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Lichess Dataset Inspection Report")
    print("=" * 65)

    pgn_files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".pgn") and os.path.isfile(os.path.join(DATA_DIR, f))
    ]

    if not pgn_files:
        print(f"No PGN files found in {DATA_DIR}")
        return

    print(f"\nFound {len(pgn_files)} PGN file(s) in data/raw/")
    all_stats = []

    for filename in sorted(pgn_files):
        filepath = os.path.join(DATA_DIR, filename)
        stats = analyze_file(filepath)
        print_report(stats)
        print_sample_games(filepath, n=1)
        all_stats.append(stats)

    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY ACROSS ALL FILES")
    print(f"{'='*65}")
    total_games = sum(s["game_count"] for s in all_stats)
    total_size = sum(s["size_mb"] for s in all_stats)
    print(f"  Total files: {len(all_stats)}")
    print(f"  Total games: {total_games:,}")
    print(f"  Total size: {total_size:.1f} MB")

    all_bot = sum(sum(s["bot_players"].values()) for s in all_stats)
    all_human = sum(sum(s["human_players"].values()) for s in all_stats)
    total_players = all_bot + all_human
    print(f"  Bot player-appearances: {all_bot:,}")
    print(f"  Human player-appearances: {all_human:,}")
    if total_players:
        print(f"  Bot ratio: {all_bot/total_players:.2%}")

    all_with_clk = sum(s["has_clocks"] for s in all_stats)
    all_analyzed = sum(min(s["game_count"], 500) for s in all_stats)
    print(f"  Games with [%clk]: {all_with_clk:,} / {all_analyzed:,} analyzed ({all_with_clk/max(all_analyzed,1):.1%})")


if __name__ == "__main__":
    main()
