"""
Inference Feature Extraction Module
===================================
Canonical, reusable feature-engineering module for Human-AI chess move detection.
Strictly replicates the feature formulas from the training pipeline with zero lookahead.

Features:
1. consecutive_fast_moves
2. premove_rate_10
3. cumulative_premove_rate
4. current_endgame_clock_ratio
5. phase_deliberation_ratio
6. move_time_cv_10
7. time_pressure_jitter
8. relative_move_time
9. time_spent_ratio
10. tank_move_count_10
"""

import io
import re
import math
from typing import Dict, List, Any, Optional, Tuple
import chess
import chess.pgn
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "consecutive_fast_moves",
    "premove_rate_10",
    "cumulative_premove_rate",
    "current_endgame_clock_ratio",
    "phase_deliberation_ratio",
    "move_time_cv_10",
    "time_pressure_jitter",
    "relative_move_time",
    "time_spent_ratio",
    "tank_move_count_10"
]

def determine_game_phase(board: chess.Board, move_number: int) -> str:
    """
    Classify game phase using the exact heuristic from training:
    - Opening: move_number <= 10 and 2 queens and non_pawn_pieces >= 10.
    - Endgame: queens == 0 and non-pawn material <= 16, or queens > 0 and non-pawn material <= 13.
    - Middlegame: all other positions.
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


def parse_time_control(tc_str: str) -> Tuple[int, int]:
    """Parse Lichess TimeControl header, e.g. '180+0' or '60+1' -> (180, 0)."""
    if not tc_str or tc_str == "-" or "+" not in tc_str:
        return 180, 0  # Standard fallback: 3+0 Blitz
    try:
        parts = tc_str.split("+")
        return max(1, int(parts[0])), max(0, int(parts[1]))
    except Exception:
        return 180, 0


def get_speed_category(base_s: int, inc_s: int, event_str: str = "") -> str:
    """Classify game into speed category (Bullet, Blitz, Rapid, Classical)."""
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


class PlayerMoveTracker:
    """
    Maintains sequential, leakage-free behavioral state for a single player.
    All computations are streaming/online up to current move k.
    """
    def __init__(self, color: str, base_time: int, inc_time: int):
        self.color = color
        self.base_time = max(1.0, float(base_time))
        self.inc_time = max(0.0, float(inc_time))
        
        # Move history lists for this player
        self.move_times: List[float] = []
        self.clock_after_moves: List[float] = []
        self.phases: List[str] = []
        self.tp_move_times: List[float] = []  # move times when clock < 15s
        
        # Running statistics
        self.consecutive_fast: int = 0
        self.opening_move_times: List[float] = []
        self.middlegame_move_times: List[float] = []

    def record_move(
        self,
        move_time: float,
        clock_after_move: float,
        phase: str
    ) -> Dict[str, float]:
        """
        Record a move and return the exact 10 behavioral features.
        Zero future leakage: only uses previous moves plus current move.
        """
        clean_move_time = max(0.0, float(move_time))
        clk_after = max(0.0, float(clock_after_move))
        
        # Previous available clock calculation
        if len(self.clock_after_moves) == 0:
            prev_available_clk = self.base_time
        else:
            prev_available_clk = self.clock_after_moves[-1] + self.inc_time
            
        # Update fast moves streak (fast move <= 0.5s)
        if clean_move_time <= 0.5:
            self.consecutive_fast += 1
        else:
            self.consecutive_fast = 0
            
        # Append to history
        self.move_times.append(clean_move_time)
        self.clock_after_moves.append(clk_after)
        self.phases.append(phase)
        
        if phase == "Opening":
            self.opening_move_times.append(clean_move_time)
        elif phase == "Middlegame":
            self.middlegame_move_times.append(clean_move_time)
            
        k = len(self.move_times)
        
        # 1. consecutive_fast_moves
        feat_consecutive_fast = float(self.consecutive_fast)
        
        # 2. premove_rate_10 (last min(10, k) moves)
        w10 = self.move_times[-10:]
        feat_premove_rate_10 = float(sum(1 for t in w10 if t == 0.0) / len(w10))
        
        # 3. cumulative_premove_rate
        cum_premoves = sum(1 for t in self.move_times if t == 0.0)
        feat_cum_premove_rate = float(cum_premoves / k)
        
        # 4. current_endgame_clock_ratio
        if phase == "Endgame":
            feat_endgame_clock = float(np.clip(clk_after / self.base_time, 0.0, 5.0))
        else:
            feat_endgame_clock = 0.0
            
        # 5. phase_deliberation_ratio
        open_mean = float(np.mean(self.opening_move_times)) if self.opening_move_times else 2.0
        open_mean_safe = max(0.5, open_mean)
        if phase == "Opening":
            feat_phase_ratio = 1.0
        else:
            mid_mean = float(np.mean(self.middlegame_move_times)) if self.middlegame_move_times else open_mean
            feat_phase_ratio = float(np.clip(mid_mean / open_mean_safe, 0.0, 20.0))
            
        # 6. move_time_cv_10
        if len(w10) == 1:
            feat_cv_10 = 0.0
        else:
            m = float(np.mean(w10))
            s = float(np.std(w10, ddof=0))
            feat_cv_10 = float(s / m) if m > 0.05 else 0.0
            
        # 7. time_pressure_jitter
        is_tp = clk_after < 15.0
        if is_tp:
            self.tp_move_times.append(clean_move_time)
            w_tp = self.tp_move_times[-5:]
            if len(w_tp) >= 2:
                feat_tp_jitter = float(np.std(w_tp, ddof=0))
            else:
                feat_tp_jitter = 0.0
        else:
            feat_tp_jitter = 0.0
            
        # 8. relative_move_time (normalized by base_time / 40.0)
        feat_rel_time = float(clean_move_time / max(1.0, self.base_time / 40.0))
        
        # 9. time_spent_ratio
        feat_time_spent_ratio = float(np.clip(clean_move_time / max(1.0, prev_available_clk), 0.0, 1.0))
        
        # 10. tank_move_count_10 (moves > 30s in last 10)
        feat_tank_count_10 = float(sum(1 for t in w10 if t > 30.0))
        
        return {
            "consecutive_fast_moves": feat_consecutive_fast,
            "premove_rate_10": feat_premove_rate_10,
            "cumulative_premove_rate": feat_cum_premove_rate,
            "current_endgame_clock_ratio": feat_endgame_clock,
            "phase_deliberation_ratio": feat_phase_ratio,
            "move_time_cv_10": feat_cv_10,
            "time_pressure_jitter": feat_tp_jitter,
            "relative_move_time": feat_rel_time,
            "time_spent_ratio": feat_time_spent_ratio,
            "tank_move_count_10": feat_tank_count_10
        }


def parse_pgn_game(pgn_content: str) -> Dict[str, Any]:
    """
    Parses a PGN string into game metadata and per-move tracking data.
    Gracefully handles missing clock comments or malformed PGNs.
    """
    pgn_io = io.StringIO(pgn_content.strip())
    game = chess.pgn.read_game(pgn_io)
    
    if game is None:
        raise ValueError("Invalid or empty PGN text provided.")
        
    headers = dict(game.headers)
    white_player = headers.get("White", "White Player")
    black_player = headers.get("Black", "Black Player")
    white_elo = headers.get("WhiteElo", "1500")
    black_elo = headers.get("BlackElo", "1500")
    tc_str = headers.get("TimeControl", "180+0")
    base_s, inc_s = parse_time_control(tc_str)
    speed_cat = get_speed_category(base_s, inc_s, headers.get("Event", ""))
    result = headers.get("Result", "*")
    eco = headers.get("ECO", "")
    opening_name = headers.get("Opening", "")
    
    # Trackers for both players
    white_tracker = PlayerMoveTracker("White", base_s, inc_s)
    black_tracker = PlayerMoveTracker("Black", base_s, inc_s)
    
    board = game.board()
    w_prev_clk = float(base_s)
    b_prev_clk = float(base_s)
    
    moves_data = []
    has_clock_data = False
    clock_missing_count = 0
    
    # Iterate through move nodes
    game_moves = list(game.mainline())
    if len(game_moves) == 0:
        return {
            "metadata": {
                "white": white_player,
                "black": black_player,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "time_control": tc_str,
                "base_time": base_s,
                "increment": inc_s,
                "speed_category": speed_cat,
                "result": result,
                "eco": eco,
                "opening_name": opening_name,
                "total_plies": 0
            },
            "has_clock_data": False,
            "warning": "PGN contains no moves (game aborted or empty).",
            "moves": []
        }
        
    # Pre-scan for clock comments
    has_clock_data = any(node.clock() is not None for node in game_moves)
    clock_missing_count = 0
    
    for ply_idx, node in enumerate(game_moves):
        ply = ply_idx + 1
        is_white = (ply_idx % 2 == 0)
        move_number = (ply_idx // 2) + 1
        color_str = "White" if is_white else "Black"
        active_player = white_player if is_white else black_player
        
        fen_before = board.fen()
        phase = determine_game_phase(board, move_number)
        
        san_move = board.san(node.move)
        uci_move = node.move.uci()
        
        move_time = None
        clock_after_move = None
        features_10 = None
        
        if has_clock_data:
            clk = node.clock()
            if clk is not None:
                prev_clk = w_prev_clk if is_white else b_prev_clk
                if ply_idx < 2:
                    if prev_clk == 0 and clk > 0:
                        dt = 0.0
                    else:
                        dt = max(0.0, prev_clk - float(clk))
                else:
                    dt = prev_clk + inc_s - float(clk)
                    
                move_time = max(0.0, round(float(dt), 2))
                clock_after_move = max(0.0, round(float(clk), 2))
                
                if is_white:
                    w_prev_clk = clock_after_move
                else:
                    b_prev_clk = clock_after_move
            else:
                clock_missing_count += 1
                # Rare missing clock in otherwise timed game: interpolate with previous
                prev_clk = w_prev_clk if is_white else b_prev_clk
                move_time = 2.0
                clock_after_move = max(0.0, prev_clk - move_time)
                if is_white:
                    w_prev_clk = clock_after_move
                else:
                    b_prev_clk = clock_after_move
                    
            tracker = white_tracker if is_white else black_tracker
            features_10 = tracker.record_move(move_time, clock_after_move, phase)
            
        # Push move to advance board
        board.push(node.move)
        fen_after = board.fen()
        
        moves_data.append({
            "ply": ply,
            "move_number": move_number,
            "player": color_str,
            "player_name": active_player,
            "san": san_move,
            "uci": uci_move,
            "from_square": chess.square_name(node.move.from_square),
            "to_square": chess.square_name(node.move.to_square),
            "fen_before": fen_before,
            "fen_after": fen_after,
            "game_phase": phase,
            "move_time": move_time,
            "clock_after_move": clock_after_move,
            "is_premove": (move_time == 0.0) if move_time is not None else False,
            "features_10": features_10
        })
        
    warning = None
    if not has_clock_data:
        warning = "Move clock data unavailable. Timing-based behavioral prediction may be unreliable."
    elif clock_missing_count > 0:
        warning = f"{clock_missing_count} moves were missing clock comments; interpolated with preceding clock."
    elif len(moves_data) < 20:
        warning = f"Short game ({len(moves_data)} plies observed). Rolling window metrics carry higher initial variance."
        
    return {
        "metadata": {
            "white": white_player,
            "black": black_player,
            "white_elo": white_elo,
            "black_elo": black_elo,
            "time_control": tc_str,
            "base_time": base_s,
            "increment": inc_s,
            "speed_category": speed_cat,
            "result": result,
            "eco": eco,
            "opening_name": opening_name,
            "total_plies": len(moves_data)
        },
        "has_clock_data": has_clock_data,
        "warning": warning,
        "moves": moves_data
    }
