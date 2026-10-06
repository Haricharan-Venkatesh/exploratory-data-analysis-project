"""
Download a representative sample from the Lichess Open Database.

CORRECTED Strategy:
- BOT games: Games where AT LEAST ONE player has title BOT (human-vs-bot or bot-vs-bot)
  - The BOT player's moves are the AI/bot moves
  - The human player's moves are the human moves
  - This gives us per-PLAYER labeling, not per-GAME labeling
- HUMAN games: Games where NEITHER player has any title or has a non-BOT title (human-vs-human)

For this project, we want per-player move labeling, so:
- human-vs-bot games give us BOTH human and bot move examples in one game
- human-vs-human games give us pure human move examples

Data model:
  game -> player -> is_bot (True/False), rating, moves with clock times

Source: Lichess Open Database (CC0 license, no auth needed)
"""

import os
import requests
import zstandard as zstd

# ---- Configuration ----------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# URL of the monthly database
DB_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"

# Stream limits
MAX_COMPRESSED_BYTES = 150 * 1024 * 1024  # 150 MB compressed = ~1.5 GB uncompressed

# Game targets
TARGET_BOT_SIDE_GAMES = 3000    # Games where at least one side is a bot
TARGET_HUMAN_GAMES = 3000       # Pure human-vs-human games

# ---- PGN Classification ----------------------------------------------------------

def classify_game(pgn_block: str) -> str:
    """
    Classify a PGN game block.
    Returns: 'human_vs_human', 'human_vs_bot', or 'bot_vs_bot'
    """
    white_is_bot = '[WhiteTitle "BOT"]' in pgn_block
    black_is_bot = '[BlackTitle "BOT"]' in pgn_block

    if white_is_bot and black_is_bot:
        return "bot_vs_bot"
    elif white_is_bot or black_is_bot:
        return "human_vs_bot"
    else:
        return "human_vs_human"


def has_clock_data(pgn_block: str) -> bool:
    """Check if a PGN game has clock annotations."""
    return "[%clk" in pgn_block


def stream_and_filter_games(url: str, max_bytes: int):
    """
    Stream a zst-compressed PGN file and split into categories.
    """
    print(f"Streaming from: {url}")
    print(f"Max compressed bytes: {max_bytes / 1024 / 1024:.0f} MB")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(response.raw)

    human_games = []    # Pure human-vs-human
    hvb_games = []      # Human-vs-bot (gold standard: mixed labels)
    bvb_games = []      # Bot-vs-bot

    games_processed = 0
    bytes_read = 0
    clk_count = 0

    buffer = ""
    chunk_size = 64 * 1024

    print("Processing games...")

    while bytes_read < max_bytes:
        chunk = reader.read(chunk_size)
        if not chunk:
            break
        bytes_read += len(chunk)

        buffer += chunk.decode("utf-8", errors="replace")

        while True:
            game_end = buffer.find("\n\n[Event ")
            if game_end == -1:
                break

            game_block = buffer[:game_end + 2].strip()
            buffer = buffer[game_end + 2:]

            if game_block and game_block.startswith("[Event "):
                games_processed += 1
                label = classify_game(game_block)
                has_clk = has_clock_data(game_block)
                if has_clk:
                    clk_count += 1

                if label == "human_vs_human" and len(human_games) < TARGET_HUMAN_GAMES:
                    human_games.append(game_block)
                elif label == "human_vs_bot" and len(hvb_games) < TARGET_BOT_SIDE_GAMES:
                    hvb_games.append(game_block)
                elif label == "bot_vs_bot":
                    bvb_games.append(game_block)

                if games_processed % 5000 == 0:
                    print(f"  Processed {games_processed:,} | "
                          f"Human-Human: {len(human_games):,} | "
                          f"Human-Bot: {len(hvb_games):,} | "
                          f"Bot-Bot: {len(bvb_games)} | "
                          f"With clocks: {clk_count:,} | "
                          f"Bytes: {bytes_read/1024/1024:.0f} MB")

            # Stop if we have enough
            if len(hvb_games) >= TARGET_BOT_SIDE_GAMES and len(human_games) >= TARGET_HUMAN_GAMES:
                print(f"\nTargets reached at {games_processed:,} games processed.")
                break

        if len(hvb_games) >= TARGET_BOT_SIDE_GAMES and len(human_games) >= TARGET_HUMAN_GAMES:
            break

    reader.close()

    print(f"\nFinal counts:")
    print(f"  Total games processed: {games_processed:,}")
    print(f"  Pure Human-vs-Human: {len(human_games):,}")
    print(f"  Human-vs-Bot: {len(hvb_games):,}")
    print(f"  Bot-vs-Bot: {len(bvb_games)}")
    print(f"  Games with [%clk] annotations: {clk_count:,} / {games_processed:,}")
    print(f"  Bytes streamed: {bytes_read/1024/1024:.1f} MB")

    return human_games, hvb_games, bvb_games


def save_games(games: list, filepath: str, label: str, source_url: str):
    """Save a list of PGN game blocks to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"% Lichess {label} Games Sample\n")
        f.write(f"% Source: {source_url}\n")
        f.write(f"% License: Creative Commons CC0\n")
        f.write(f"% Games: {len(games)}\n\n")
        for game in games:
            f.write(game.strip() + "\n\n")
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  [{label}] {len(games):,} games -> {os.path.basename(filepath)} ({size_mb:.1f} MB)")


def main():
    print("=" * 65)
    print("Lichess Open Database Sampler — Human vs Bot Classification")
    print("=" * 65)
    print()

    human_games, hvb_games, bvb_games = stream_and_filter_games(DB_URL, MAX_COMPRESSED_BYTES)

    print("\nSaving files to data/raw/...")

    human_file = os.path.join(OUTPUT_DIR, "lichess_human_human_games.pgn")
    hvb_file = os.path.join(OUTPUT_DIR, "lichess_human_vs_bot_games.pgn")
    bvb_file = os.path.join(OUTPUT_DIR, "lichess_bot_vs_bot_games.pgn")

    save_games(human_games, human_file, "Human-vs-Human", DB_URL)
    save_games(hvb_games, hvb_file, "Human-vs-Bot", DB_URL)
    if bvb_games:
        save_games(bvb_games, bvb_file, "Bot-vs-Bot", DB_URL)

    # Write manifest
    manifest_path = os.path.join(OUTPUT_DIR, "MANIFEST.txt")
    with open(manifest_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("Lichess Open Database — Human vs Bot Game Sample\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Source: {DB_URL}\n")
        f.write(f"License: Creative Commons CC0 (public domain)\n")
        f.write(f"Strategy: Streamed first {MAX_COMPRESSED_BYTES/1024/1024:.0f} MB compressed\n\n")
        f.write("Files:\n")
        for fp in [human_file, hvb_file, bvb_file]:
            if os.path.exists(fp):
                size = os.path.getsize(fp) / 1024 / 1024
                f.write(f"  {os.path.basename(fp)}: {size:.1f} MB\n")
        f.write("\nLabeling:\n")
        f.write("  Human-vs-Human: Both players have no BOT title -> pure HUMAN behavior\n")
        f.write("  Human-vs-Bot: One player has [WhiteTitle/BlackTitle 'BOT'] -> known labels per player\n")
        f.write("  Bot-vs-Bot: Both players are BOTs -> pure AI behavior\n")
    print(f"\nManifest written: {manifest_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
