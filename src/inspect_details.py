import chess.pgn

with open('data/raw/lichess_human_vs_bot_games.pgn', 'r', encoding='utf-8') as f:
    for g_i in range(3):
        game = chess.pgn.read_game(f)
        if not game: break
        tc = game.headers.get('TimeControl', '')
        site = game.headers.get('Site', '')
        print(f"=== Game {g_i+1}: {site} | TC: {tc} ===")
        board = game.board()
        for idx, node in enumerate(list(game.mainline())[:10]):
            color = 'White' if idx % 2 == 0 else 'Black'
            ply = idx + 1
            move_num = (idx // 2) + 1
            clk = node.clock()
            san = board.san(node.move)
            board.push(node.move)
            print(f"  Ply {ply:2d} (Move {move_num:2d} {color:5s}): {san:6s} | clk: {clk}")
