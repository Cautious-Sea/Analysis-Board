"""
Simple test for the chess variant game logic.
"""
import chess
from chess_variant.game_logic import ChessVariantGame

def main():
    """Run a simple test."""
    print("Testing the chess variant game logic...")
    
    # Create a game
    game = ChessVariantGame()
    
    # Set up a position for testing the foul rule
    game.setup_empty_board()
    
    # Place pieces for a foul scenario
    game.place_piece(chess.E2, chess.QUEEN, chess.WHITE)  # Queen (9 points)
    game.place_piece(chess.E4, chess.PAWN, chess.BLACK)   # Pawn (1 point)
    
    # Set white to move
    game.set_side_to_move(chess.WHITE)
    
    # Print initial state
    print("Initial state:")
    print(f"Board: {game.board}")
    print(f"White moves: {game.move_counts[chess.WHITE]}")
    print(f"Black moves: {game.move_counts[chess.BLACK]}")
    
    # Make a move
    move = chess.Move(chess.E2, chess.E4)
    result = game.make_move(move)
    
    # Print result
    print(f"Move result: {result}")
    print(f"Board after move: {game.board}")
    print(f"White moves: {game.move_counts[chess.WHITE]}")
    print(f"Black moves: {game.move_counts[chess.BLACK]}")
    print(f"White score: {game.capture_scores[chess.WHITE]}")
    print(f"Black score: {game.capture_scores[chess.BLACK]}")
    
    # Check if it was a capture
    print(f"Last move: {game.last_move}")
    print(f"Move history: {game.move_history}")
    
    # Print game status
    status = game.get_game_status()
    print("Game status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
