"""
Test script for the custom chess variant game logic.
"""
import chess
from chess_variant.game_logic import ChessVariantGame

def test_move_limit():
    """Test that the move limit is enforced correctly."""
    game = ChessVariantGame()
    game.setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # Make 6 moves for white
    white_moves = 0
    while white_moves < 6:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        # Find a move for white
        move = legal_moves[0]
        game.make_move(move)

        # Count white moves
        if game.board.turn == chess.BLACK:
            white_moves += 1
            print(f"White move {white_moves}: {move.uci()}")

        # Make a move for black if it's black's turn
        if game.board.turn == chess.BLACK:
            legal_moves = game.get_legal_moves()
            if legal_moves:
                move = legal_moves[0]
                game.make_move(move)
                print(f"Black move {game.move_counts[chess.BLACK]}: {move.uci()}")

    # Try to make one more move for white
    if game.board.turn == chess.WHITE:
        legal_moves = game.get_legal_moves()
        if legal_moves:
            move = legal_moves[0]
            result = game.make_move(move)
            print(f"Extra white move allowed: {result}")
            assert not result, "White should not be allowed more than 6 moves"

    print("Move limit test passed!")

def test_scoring_system():
    """Test that the scoring system works correctly."""
    game = ChessVariantGame()

    # Set up a position with some pieces
    game.setup_empty_board()

    # Place some pieces
    game.place_piece(chess.E4, chess.PAWN, chess.WHITE)
    game.place_piece(chess.D5, chess.PAWN, chess.BLACK)
    game.place_piece(chess.A1, chess.ROOK, chess.WHITE)
    game.place_piece(chess.H1, chess.ROOK, chess.WHITE)  # Add another white rook for black to capture
    game.place_piece(chess.H8, chess.ROOK, chess.BLACK)

    # Set white to move
    game.set_side_to_move(chess.WHITE)

    # Make a capture
    capture_move = chess.Move(chess.E4, chess.D5)
    game.make_move(capture_move)

    # Check the score
    status = game.get_game_status()
    assert status["scores"]["white"] == 1, f"White score should be 1, got {status['scores']['white']}"

    # Make a move for black to capture the white rook
    game.make_move(chess.Move(chess.H8, chess.H1))

    # Check the score
    status = game.get_game_status()
    assert status["scores"]["black"] == 5, f"Black score should be 5, got {status['scores']['black']}"

    print("Scoring system test passed!")

def test_foul_rule():
    """Test that the foul rule is enforced correctly."""
    game = ChessVariantGame()

    # Set up a position for testing the foul rule
    game.setup_empty_board()

    # Place pieces for a foul scenario
    game.place_piece(chess.E2, chess.QUEEN, chess.WHITE)  # Queen (9 points)
    game.place_piece(chess.E4, chess.PAWN, chess.BLACK)   # Pawn (1 point)

    # Set white to move
    game.set_side_to_move(chess.WHITE)

    # Make 5 moves for white and black
    for i in range(5):
        # Move white queen back and forth
        if i % 2 == 0:
            game.make_move(chess.Move(chess.E2, chess.D2))
        else:
            game.make_move(chess.Move(chess.D2, chess.E2))

        # Move black pawn back and forth
        if i % 2 == 0:
            game.make_move(chess.Move(chess.E4, chess.E5))
        else:
            game.make_move(chess.Move(chess.E5, chess.E4))

    # Print the current move counts
    white_moves = game.move_counts[chess.WHITE]
    black_moves = game.move_counts[chess.BLACK]
    print(f"Before 6th move: White moves = {white_moves}, Black moves = {black_moves}")

    # Now make the 6th move for white - a foul (capturing a pawn with the queen)
    game.make_move(chess.Move(chess.E2, chess.E4))

    # Print the move that was made and the foul status
    print(f"White's 6th move: e2e4 (Queen captures Pawn)")
    print(f"Foul committed: {game.foul_committed[chess.WHITE]}")

    # Check if black got an extra move
    status = game.get_game_status()
    print(f"Black extra moves: {status['black_moves']['extra']}")

    # Let's manually check the conditions for the foul
    last_move = game.move_history[-1]
    move, is_capture, captured_piece_type, moving_piece_type = last_move
    print(f"Last move details: capture={is_capture}, captured={captured_piece_type}, moving={moving_piece_type}")

    # The foul rule should give black an extra move
    assert status["black_moves"]["extra"] == 1, f"Black should have 1 extra move, got {status['black_moves']['extra']}"

    print("Foul rule test passed!")

def run_all_tests():
    """Run all tests."""
    print("Running tests for the custom chess variant game logic...")
    test_move_limit()
    test_scoring_system()
    test_foul_rule()
    print("All tests passed!")

if __name__ == "__main__":
    run_all_tests()
