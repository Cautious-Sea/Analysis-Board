"""
Test for the foul rule in the chess variant game.
"""
import chess
from chess_variant.game_logic import ChessVariantGame

def main():
    """Test the foul rule."""
    print("Testing the foul rule...")
    
    # Create a game
    game = ChessVariantGame()
    
    # Set up a position for testing the foul rule
    game.setup_empty_board()
    
    # Place pieces for a foul scenario
    # We need enough pieces to make 5 moves without captures
    game.place_piece(chess.E1, chess.QUEEN, chess.WHITE)  # Queen (9 points)
    game.place_piece(chess.E7, chess.PAWN, chess.BLACK)   # Pawn (1 point)
    
    # Add some more pieces to move around
    game.place_piece(chess.A1, chess.ROOK, chess.WHITE)
    game.place_piece(chess.H1, chess.ROOK, chess.WHITE)
    game.place_piece(chess.A8, chess.ROOK, chess.BLACK)
    game.place_piece(chess.H8, chess.ROOK, chess.BLACK)
    
    # Set white to move
    game.set_side_to_move(chess.WHITE)
    
    # Make 5 moves for each player
    print("Making 5 moves for each player...")
    
    # Move 1
    game.make_move(chess.Move(chess.A1, chess.A2))
    print("White move 1: a1a2")
    game.make_move(chess.Move(chess.A8, chess.A7))
    print("Black move 1: a8a7")
    
    # Move 2
    game.make_move(chess.Move(chess.A2, chess.A3))
    print("White move 2: a2a3")
    game.make_move(chess.Move(chess.A7, chess.A6))
    print("Black move 2: a7a6")
    
    # Move 3
    game.make_move(chess.Move(chess.A3, chess.A4))
    print("White move 3: a3a4")
    game.make_move(chess.Move(chess.A6, chess.A5))
    print("Black move 3: a6a5")
    
    # Move 4
    game.make_move(chess.Move(chess.H1, chess.H2))
    print("White move 4: h1h2")
    game.make_move(chess.Move(chess.H8, chess.H7))
    print("Black move 4: h8h7")
    
    # Move 5
    game.make_move(chess.Move(chess.H2, chess.H3))
    print("White move 5: h2h3")
    game.make_move(chess.Move(chess.H7, chess.H6))
    print("Black move 5: h7h6")
    
    # Print the current move counts
    white_moves = game.move_counts[chess.WHITE]
    black_moves = game.move_counts[chess.BLACK]
    print(f"Before 6th move: White moves = {white_moves}, Black moves = {black_moves}")
    
    # Now make the 6th move for white - a foul (capturing a pawn with the queen)
    print("Making White's 6th move - a foul (capturing a pawn with the queen)...")
    game.make_move(chess.Move(chess.E1, chess.E7))
    
    # Print the move that was made and the foul status
    print(f"White's 6th move: e1e7 (Queen captures Pawn)")
    print(f"Foul committed: {game.foul_committed[chess.WHITE]}")
    
    # Check if black got an extra move
    status = game.get_game_status()
    print(f"Black extra moves: {status['black_moves']['extra']}")
    
    # Let's manually check the conditions for the foul
    last_move = game.move_history[-1]
    _, is_capture, captured_piece_type, moving_piece_type = last_move
    print(f"Last move details: capture={is_capture}, captured={captured_piece_type}, moving={moving_piece_type}")
    
    # The foul rule should give black an extra move
    if status["black_moves"]["extra"] == 1:
        print("PASS: Black received 1 extra move as expected")
    else:
        print(f"FAIL: Black should have 1 extra move, got {status['black_moves']['extra']}")
    
    # Print final game status
    print("\nFinal game status:")
    for key, value in status.items():
        if key != "board":  # Skip printing the board
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
