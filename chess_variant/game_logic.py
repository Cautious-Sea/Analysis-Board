"""
Game logic for the custom chess variant with 6-move limit and special rules.
"""
import chess
from typing import Dict, List, Optional, Tuple

class ChessVariantGame:
    """
    Implementation of the custom chess variant with the following rules:
    - Each player has exactly 6 moves
    - Game starts from a configurable position
    - Scoring based on captured pieces if no checkmate
    - Foul rule on the 6th move if capturing with a higher-value piece
    """
    # Piece values for scoring
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King cannot be captured
    }

    def __init__(self):
        """Initialize the game with default settings."""
        self.board = chess.Board(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.reset_game()

    def reset_game(self):
        """Reset the game state without changing the board position."""
        self.move_counts = {chess.WHITE: 0, chess.BLACK: 0}
        self.extra_moves = {chess.WHITE: 0, chess.BLACK: 0}
        self.capture_scores = {chess.WHITE: 0, chess.BLACK: 0}
        self.move_history = []
        self.game_over = False
        self.winner = None
        self.draw = False
        self.last_move = None
        self.foul_committed = {chess.WHITE: False, chess.BLACK: False}

    def setup_position(self, fen: str):
        """
        Set up a custom position using FEN notation.

        Args:
            fen: FEN string representing the position
        """
        try:
            self.board = chess.Board(fen=fen)
            self.reset_game()
            return True
        except ValueError:
            return False

    def setup_empty_board(self):
        """Set up an empty board."""
        self.board = chess.Board(fen="8/8/8/8/8/8/8/8 w - - 0 1")
        self.reset_game()

    def place_piece(self, square: int, piece_type: int, color: bool) -> bool:
        """
        Place a piece on the board during setup.

        Args:
            square: Chess square (0-63)
            piece_type: Type of piece (PAWN, KNIGHT, etc.)
            color: Color of piece (WHITE/BLACK)

        Returns:
            Success of the operation
        """
        if not 0 <= square < 64:
            return False

        # Remove any existing piece
        self.board.remove_piece_at(square)

        # Place the new piece if not None
        if piece_type is not None:
            self.board.set_piece_at(square, chess.Piece(piece_type, color))

        return True

    def remove_piece(self, square: int) -> bool:
        """
        Remove a piece from the board during setup.

        Args:
            square: Chess square (0-63)

        Returns:
            Success of the operation
        """
        if not 0 <= square < 64:
            return False

        # Remove the piece
        self.board.remove_piece_at(square)

        return True

    def set_side_to_move(self, color: bool):
        """
        Set which side moves first.

        Args:
            color: chess.WHITE or chess.BLACK
        """
        # Create a new FEN with the same position but different side to move
        parts = self.board.fen().split()
        parts[1] = "w" if color == chess.WHITE else "b"
        new_fen = " ".join(parts)
        self.board = chess.Board(fen=new_fen)

    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move on the board, applying the custom rules.

        Args:
            move: Chess move to make

        Returns:
            Whether the move was legal and executed
        """
        if self.game_over:
            return False

        # Check if the move is legal
        if move not in self.board.legal_moves:
            return False

        current_player = self.board.turn
        move_number = self.move_counts[current_player] + 1

        # Check if player has used all their moves
        if move_number > 6 + self.extra_moves[current_player]:
            return False

        # Store information about the move for foul detection
        is_capture = self.board.is_capture(move)
        captured_piece_type = None
        captured_piece_value = 0
        moving_piece_type = self.board.piece_type_at(move.from_square)
        moving_piece_value = self.PIECE_VALUES[moving_piece_type]

        # Get captured piece info before the move
        if is_capture:
            # Handle en passant capture
            if self.board.is_en_passant(move):
                captured_piece_type = chess.PAWN
                captured_piece_value = self.PIECE_VALUES[chess.PAWN]
            else:
                captured_piece_type = self.board.piece_type_at(move.to_square)
                captured_piece_value = self.PIECE_VALUES[captured_piece_type]

        # Make the move
        self.last_move = move
        self.board.push(move)
        self.move_counts[current_player] += 1
        self.move_history.append((move, is_capture, captured_piece_type, moving_piece_type))

        # Update score if it was a capture
        if is_capture:
            self.capture_scores[current_player] += captured_piece_value

            # Check for foul on 6th move (unfavorable trade)
            if move_number == 6 and moving_piece_value > captured_piece_value:
                self.foul_committed[current_player] = True
                opponent = not current_player
                self.extra_moves[opponent] += 1

        # Check for game end conditions
        self._check_game_end()

        return True

    def _check_game_end(self):
        """Check if the game has ended and determine the winner."""
        # Check for checkmate
        if self.board.is_checkmate():
            self.game_over = True
            self.winner = not self.board.turn  # The side that just moved won
            return

        # Check for stalemate or insufficient material
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            self.game_over = True
            self.draw = True
            return

        # Check if both players have used all their moves (including extra moves)
        white_moves_left = 6 + self.extra_moves[chess.WHITE] - self.move_counts[chess.WHITE]
        black_moves_left = 6 + self.extra_moves[chess.BLACK] - self.move_counts[chess.BLACK]

        if white_moves_left <= 0 and black_moves_left <= 0:
            self.game_over = True

            # Determine winner by capture score
            if self.capture_scores[chess.WHITE] > self.capture_scores[chess.BLACK]:
                self.winner = chess.WHITE
            elif self.capture_scores[chess.BLACK] > self.capture_scores[chess.WHITE]:
                self.winner = chess.BLACK
            else:
                self.draw = True

    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves in the current position."""
        if self.game_over:
            return []

        current_player = self.board.turn
        move_number = self.move_counts[current_player] + 1

        # Check if player has used all their moves
        if move_number > 6 + self.extra_moves[current_player]:
            return []

        return list(self.board.legal_moves)

    def get_game_status(self) -> Dict:
        """
        Get the current game status.

        Returns:
            Dictionary with game status information
        """
        current_player = self.board.turn
        white_moves_made = self.move_counts[chess.WHITE]
        black_moves_made = self.move_counts[chess.BLACK]
        white_moves_left = 6 + self.extra_moves[chess.WHITE] - white_moves_made
        black_moves_left = 6 + self.extra_moves[chess.BLACK] - black_moves_made

        return {
            "board": self.board,
            "current_player": "White" if current_player == chess.WHITE else "Black",
            "white_moves": {
                "made": white_moves_made,
                "left": white_moves_left,
                "extra": self.extra_moves[chess.WHITE],
                "foul": self.foul_committed[chess.WHITE]
            },
            "black_moves": {
                "made": black_moves_made,
                "left": black_moves_left,
                "extra": self.extra_moves[chess.BLACK],
                "foul": self.foul_committed[chess.BLACK]
            },
            "scores": {
                "white": self.capture_scores[chess.WHITE],
                "black": self.capture_scores[chess.BLACK]
            },
            "game_over": self.game_over,
            "winner": "White" if self.winner == chess.WHITE else "Black" if self.winner == chess.BLACK else None,
            "draw": self.draw,
            "last_move": self.last_move
        }

    def get_piece_at(self, square: int) -> Optional[Tuple[int, bool]]:
        """
        Get the piece at a specific square.

        Args:
            square: Chess square (0-63)

        Returns:
            Tuple of (piece_type, color) or None if empty
        """
        piece = self.board.piece_at(square)
        if piece:
            return (piece.piece_type, piece.color)
        return None
