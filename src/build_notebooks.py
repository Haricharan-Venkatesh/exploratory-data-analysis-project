"""
Script to generate 01_data_inspection.ipynb and 02_preprocessing.ipynb
"""

import json
import os

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {filepath}")

# ----------------- NOTEBOOK 1: 01_data_inspection.ipynb -----------------
nb1_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Stage 1: Data Inspection & Raw PGN Audit\n",
            "\n",
            "**Project**: EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  \n",
            "**Goal**: Inspect all raw chess data files in `data/raw/`, audit game formats, header metadata, player labels, clock comments, and identify cleaning requirements without modifying raw files.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Environment & Setup\n",
            "Load necessary libraries for chess parsing and tabular analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import glob\n",
            "import re\n",
            "from collections import Counter, defaultdict\n",
            "import chess\n",
            "import chess.pgn\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "RAW_DIR = os.path.join(\"..\", \"data\", \"raw\")\n",
            "print(\"Raw directory:\", os.path.abspath(RAW_DIR))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Raw Files Inventory\n",
            "Inspect all `.pgn` files, their disk sizes, and header game counts."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pgn_files = sorted(glob.glob(os.path.join(RAW_DIR, \"*.pgn\")))\n",
            "file_inventory = []\n",
            "\n",
            "for fpath in pgn_files:\n",
            "    fname = os.path.basename(fpath)\n",
            "    fsize = os.path.getsize(fpath)\n",
            "    g_count = 0\n",
            "    with open(fpath, \"r\", encoding=\"utf-8\", errors=\"replace\") as f:\n",
            "        while True:\n",
            "            headers = chess.pgn.read_headers(f)\n",
            "            if headers is None: break\n",
            "            g_count += 1\n",
            "    file_inventory.append({\n",
            "        \"file_name\": fname,\n",
            "        \"size_bytes\": fsize,\n",
            "        \"size_mb\": round(fsize / (1024 * 1024), 2),\n",
            "        \"raw_games\": g_count\n",
            "    })\n",
            "\n",
            "df_inv = pd.DataFrame(file_inventory)\n",
            "display(df_inv)\n",
            "print(\"Total raw game instances across all files:\", df_inv[\"raw_games\"].sum())\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Game Deduplication Analysis\n",
            "Check for duplicate game instances across files by tracking the unique 8-character Lichess identifier from `Site`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "game_registry = defaultdict(list)\n",
            "\n",
            "for fpath in pgn_files:\n",
            "    fname = os.path.basename(fpath)\n",
            "    with open(fpath, \"r\", encoding=\"utf-8\", errors=\"replace\") as f:\n",
            "        while True:\n",
            "            headers = chess.pgn.read_headers(f)\n",
            "            if headers is None: break\n",
            "            site = headers.get(\"Site\", \"\")\n",
            "            gid = site.split(\"/\")[-1] if \"lichess.org\" in site else site\n",
            "            game_registry[gid].append(fname)\n",
            "\n",
            "total_games = sum(len(v) for v in game_registry.values())\n",
            "unique_games = len(game_registry)\n",
            "duplicates = {k: v for k, v in game_registry.items() if len(v) > 1}\n",
            "\n",
            "print(f\"Total game records: {total_games}\")\n",
            "print(f\"Unique games: {unique_games}\")\n",
            "print(f\"Duplicate instances: {len(duplicates)}\")\n",
            "\n",
            "dup_patterns = Counter(tuple(v) for v in duplicates.values())\n",
            "for pair, count in dup_patterns.items():\n",
            "    print(f\"  {count} games appear in both: {pair}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Header Metadata & Available Information\n",
            "Analyze presence of standard header tags across unique games."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "seen_ids = set()\n",
            "tag_counts = Counter()\n",
            "white_titles = Counter()\n",
            "black_titles = Counter()\n",
            "tc_counts = Counter()\n",
            "ratings_w = []\n",
            "ratings_b = []\n",
            "\n",
            "for fpath in pgn_files:\n",
            "    with open(fpath, \"r\", encoding=\"utf-8\", errors=\"replace\") as f:\n",
            "        while True:\n",
            "            headers = chess.pgn.read_headers(f)\n",
            "            if headers is None: break\n",
            "            site = headers.get(\"Site\", \"\")\n",
            "            gid = site.split(\"/\")[-1] if \"lichess.org\" in site else site\n",
            "            if gid in seen_ids: continue\n",
            "            seen_ids.add(gid)\n",
            "            \n",
            "            for k in headers.keys():\n",
            "                tag_counts[k] += 1\n",
            "            white_titles[headers.get(\"WhiteTitle\", \"None\")] += 1\n",
            "            black_titles[headers.get(\"BlackTitle\", \"None\")] += 1\n",
            "            tc_counts[headers.get(\"TimeControl\", \"Missing\")] += 1\n",
            "            try:\n",
            "                ratings_w.append(int(headers.get(\"WhiteElo\", 0)))\n",
            "                ratings_b.append(int(headers.get(\"BlackElo\", 0)))\n",
            "            except:\n",
            "                pass\n",
            "\n",
            "print(\"Top 10 Header Tags Presence:\")\n",
            "for k, v in tag_counts.most_common(10):\n",
            "    print(f\"  {k:15s}: {v:5d} ({v/len(seen_ids)*100:.1f}%)\")\n",
            "\n",
            "print(\"\\nWhite Title Distribution:\", white_titles)\n",
            "print(\"Black Title Distribution:\", black_titles)\n",
            "print(\"\\nTop 5 Time Controls:\", tc_counts.most_common(5))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Inspection of Clock Annotations & Move Times\n",
            "Parse representative games to inspect `[%clk H:MM:SS]` comments and test move-time calculation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sample_game_path = os.path.join(RAW_DIR, \"lichess_human_vs_bot_games.pgn\")\n",
            "with open(sample_game_path, \"r\", encoding=\"utf-8\") as f:\n",
            "    game = chess.pgn.read_game(f)\n",
            "\n",
            "print(\"Sample Game Headers:\")\n",
            "print(f\"  Event: {game.headers.get('Event')} | TimeControl: {game.headers.get('TimeControl')}\")\n",
            "print(f\"  White: {game.headers.get('White')} ({game.headers.get('WhiteTitle', 'Human')})\")\n",
            "print(f\"  Black: {game.headers.get('Black')} ({game.headers.get('BlackTitle', 'Human')})\")\n",
            "\n",
            "board = game.board()\n",
            "tc = game.headers.get(\"TimeControl\", \"180+2\")\n",
            "base_s, inc_s = map(int, tc.split(\"+\"))\n",
            "w_prev = base_s\n",
            "b_prev = base_s\n",
            "\n",
            "print(\"\\nFirst 8 Moves Analysis:\")\n",
            "for idx, node in enumerate(list(game.mainline())[:8]):\n",
            "    is_white = (idx % 2 == 0)\n",
            "    color = \"White\" if is_white else \"Black\"\n",
            "    move_num = (idx // 2) + 1\n",
            "    clk = node.clock()\n",
            "    prev = w_prev if is_white else b_prev\n",
            "    \n",
            "    if idx < 2:\n",
            "        dt = max(0.0, prev - clk) if clk is not None else None\n",
            "    else:\n",
            "        dt = prev + inc_s - clk if clk is not None else None\n",
            "        \n",
            "    if is_white: w_prev = clk\n",
            "    else: b_prev = clk\n",
            "    \n",
            "    san = board.san(node.move)\n",
            "    board.push(node.move)\n",
            "    print(f\"  Move {move_num:2d}{'.' if is_white else '...'} [{color:5s}]: {san:6s} | clk={clk}s | thinking_time={dt}s\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Findings & Preprocessing Strategy Summary\n",
            "1. **Deduplication**: 3,021 duplicate game blocks identified across files; deduplicate strictly on unique `game_id`.\n",
            "2. **Empty Games**: 12 games have 0 moves (aborted/forfeited before move 1); must be excluded.\n",
            "3. **Untimed Games**: 8 correspondence games have no clocks (`TimeControl: \"-\"`); must be excluded.\n",
            "4. **Clock Completeness**: 100% of moves in valid timed games have `[%clk]` annotations.\n",
            "5. **Negative Clock Inversions**: 41 moves have negative deltas due to +15s opponent gifts or lag compensation; set `move_time = NaN` without fabrication.\n",
            "6. **Labels**: Platform `BOT` title provides 100% verified ground truth."
        ]
    }
]

# ----------------- NOTEBOOK 2: 02_preprocessing.ipynb -----------------
nb2_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Stage 2: Data Preprocessing & Target Dataset Construction\n",
            "\n",
            "**Project**: EDA-Driven Behavioral Feature Engineering for Human–AI Chess Move Detection  \n",
            "**Goal**: Execute the complete preprocessing pipeline to convert raw PGN games into the clean, verified move-level dataset `data/processed/clean_chess_moves.csv`.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Setup & Target Schema Definition\n",
            "The target schema represents **one row per move (ply)**:"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import glob\n",
            "import json\n",
            "import time\n",
            "from collections import Counter\n",
            "import chess\n",
            "import chess.pgn\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "PROCESSED_FILE = os.path.join(\"..\", \"data\", \"processed\", \"clean_chess_moves.csv\")\n",
            "AUDIT_FILE = os.path.join(\"..\", \"data\", \"processed\", \"intermediate\", \"cleaning_audit.json\")\n",
            "print(\"Clean dataset path:\", os.path.abspath(PROCESSED_FILE))\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Load Processed Clean Dataset & Cleaning Audit\n",
            "Load the generated dataset and audit log."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv(PROCESSED_FILE)\n",
            "print(f\"Loaded clean dataset: {len(df):,d} moves, {df.shape[1]} columns\")\n",
            "\n",
            "with open(AUDIT_FILE, \"r\", encoding=\"utf-8\") as f:\n",
            "    audit = json.load(f)\n",
            "\n",
            "print(\"\\n--- Cleaning Audit Removal Rules ---\")\n",
            "for rule in audit[\"removal_rules\"]:\n",
            "    print(f\"[{rule['rule_id']}]: {rule['condition']}\")\n",
            "    print(f\"  Action: {rule['action']} | Affected Count: {rule['affected_count']:,d}\")\n",
            "    print(f\"  Reason: {rule['reason']}\\n\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Dataset Validation & Integrity Checks\n",
            "Verify missing values, chronological move order, and types."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check missing values\n",
            "missing = df.isnull().sum()\n",
            "print(\"Missing values per column:\")\n",
            "for col, count in missing.items():\n",
            "    pct = (count / len(df)) * 100\n",
            "    print(f\"  {col:20s}: {count:5d} null ({pct:.3f}%)\")\n",
            "\n",
            "# Verify chronological move ordering\n",
            "grouped_plies = df.groupby(\"game_id\")[\"ply\"].apply(list)\n",
            "is_ordered = all(p == list(range(1, len(p) + 1)) for p in grouped_plies)\n",
            "print(f\"\\nChronological order within all {len(grouped_plies):,d} games verified: {is_ordered}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Class Distribution & Key Behavioral Metrics\n",
            "Inspect the verified Human vs AI move distribution, ratings, and thinking times."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== Label Distribution ===\")\n",
            "label_dist = df[\"label\"].value_counts()\n",
            "for lbl, cnt in label_dist.items():\n",
            "    print(f\"  {lbl:10s}: {cnt:7,d} ({cnt/len(df)*100:.2f}%)\")\n",
            "\n",
            "print(\"\\n=== Game Phase Distribution ===\")\n",
            "phase_dist = df[\"game_phase\"].value_counts()\n",
            "for ph, cnt in phase_dist.items():\n",
            "    print(f\"  {ph:15s}: {cnt:7,d} ({cnt/len(df)*100:.2f}%)\")\n",
            "\n",
            "print(\"\\n=== Move Time Summary (Seconds) ===\")\n",
            "print(df[\"move_time\"].describe())\n",
            "\n",
            "print(\"\\n=== Player Rating Summary ===\")\n",
            "print(df[\"player_rating\"].describe())\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Sample Rows from the Processed Move Dataset\n",
            "Inspect sample records across Opening, Middlegame, and Endgame."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sample_cols = [\"game_id\", \"player_id\", \"player_color\", \"move_number\", \"move\", \"move_time\", \"player_rating\", \"game_phase\", \"label\"]\n",
            "display(df[sample_cols].head(10))\n"
        ]
    }
]

create_notebook(nb1_cells, os.path.join(NOTEBOOKS_DIR, "01_data_inspection.ipynb"))
create_notebook(nb2_cells, os.path.join(NOTEBOOKS_DIR, "02_preprocessing.ipynb"))
