import chess

def determine_game_phase(board: chess.Board, move_number: int) -> str:
    """
    Transparent, reproducible rule to classify chess game phase:
    
    1. Opening:
       - Full-move number <= 10 (first 20 plies) AND both queens are on board AND
         total non-pawn pieces >= 10 (indicating development phase).
    
    2. Endgame:
       - Both queens are off the board AND total non-pawn material <= 16 points
         (where Q=9, R=5, B=3, N=3), OR
       - If any queen is on the board, total non-pawn material <= 13 points
         (e.g., Queen vs Queen with no other major pieces, or Queen vs Rook).
         
    3. Middlegame:
       - Any position that is neither Opening nor Endgame.
    """
    # Count pieces
    piece_map = board.piece_map()
    queens = sum(1 for p in piece_map.values() if p.piece_type == chess.QUEEN)
    non_pawn_pieces = sum(1 for p in piece_map.values() if p.piece_type not in (chess.PAWN, chess.KING))
    
    # Material evaluation (standard values)
    material_values = {chess.QUEEN: 9, chess.ROOK: 5, chess.BISHOP: 3, chess.KNIGHT: 3}
    total_non_pawn_material = sum(material_values.get(p.piece_type, 0) for p in piece_map.values())
    
    if move_number <= 10 and queens == 2 and non_pawn_pieces >= 10:
        return "Opening"
    
    if queens == 0 and total_non_pawn_material <= 16:
        return "Endgame"
    if queens > 0 and total_non_pawn_material <= 13:
        return "Endgame"
        
    return "Middlegame"

# Test with starting board
b = chess.Board()
print("Starting board phase:", determine_game_phase(b, 1))
