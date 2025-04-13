"""
GUI implementation for the custom chess variant using Tkinter.
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import chess
import os
import sys
import time
from PIL import Image, ImageTk
from typing import Dict, List, Optional, Tuple, Callable
import threading

from chess_variant.game_logic import ChessVariantGame
from chess_variant.engine import StockfishEngine

class ChessVariantGUI:
    """
    Tkinter-based GUI for the custom chess variant.
    """
    # Colors for the chess board - using higher contrast colors
    LIGHT_SQUARE_COLOR = "#EEEED2"  # Light cream color
    DARK_SQUARE_COLOR = "#769656"  # Medium green color
    SELECTED_SQUARE_COLOR = "#AACCFF"  # Light blue for selected squares
    LAST_MOVE_COLOR = "#BBCC44"  # Yellowish green for last move
    HIGHLIGHT_COLOR = "#FF6666"  # Red for highlights

    # Board dimensions
    SQUARE_SIZE = 64
    BOARD_SIZE = 8 * SQUARE_SIZE

    # Piece symbols for display
    PIECE_SYMBOLS = {
        chess.PAWN: "P",
        chess.KNIGHT: "N",
        chess.BISHOP: "B",
        chess.ROOK: "R",
        chess.QUEEN: "Q",
        chess.KING: "K"
    }

    def __init__(self, root):
        """
        Initialize the GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Custom Chess Variant")
        self.root.resizable(False, False)

        # Initialize game logic
        self.game = ChessVariantGame()

        # Initialize Stockfish engine
        self.engine = StockfishEngine(depth=18)

        # GUI state variables
        self.selected_square = None
        self.legal_moves_for_selected = []  # Store legal moves for the selected piece
        self.setup_mode = False
        self.board_flipped = False  # Track board orientation (False = white at bottom, True = black at bottom)

        # Debug flag - set to False to disable debug output
        self.debug = False

        # Method to print debug messages only when debug is enabled
        self.debug_print = lambda *args, **kwargs: print(*args, **kwargs) if self.debug else None

        # More GUI state variables
        self.setup_piece = None
        self.setup_color = chess.WHITE
        self.best_move_arrow = None

        # Drag and drop variables
        self.drag_start_square = None  # For active drag and drop moves
        self.drag_target_square = None  # For the target square of a drag move
        self.potential_drag_square = None  # For potential drag (during delay)
        self.potential_drag_x = None
        self.potential_drag_y = None
        self.drag_activation_id = None  # For the delayed drag activation
        self.drag_piece_type = None  # For setup mode drag and drop
        self.drag_piece_color = None  # For setup mode drag and drop

        # Analysis state variables
        self.analysis_running = False
        self.analysis_thread = None
        self.analysis_stop_event = threading.Event()

        # Load piece images
        self.piece_images = {}
        self._load_piece_images()

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Create the board canvas
        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.BOARD_SIZE + 40,  # Extra space for coordinates
            height=self.BOARD_SIZE + 40,
            bg="#DDDDDD"
        )
        self.canvas.grid(row=0, column=0, rowspan=2)

        # Add canvas bindings for mouse events
        self.canvas.bind("<ButtonPress-1>", self._canvas_mouse_down)
        self.canvas.bind("<B1-Motion>", self._canvas_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_mouse_up)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

        # Create control panel
        self.control_panel = ttk.Frame(self.main_frame, padding=10)
        self.control_panel.grid(row=0, column=1, sticky="n")

        # Game info panel
        self.info_panel = ttk.Frame(self.main_frame, padding=10)
        self.info_panel.grid(row=1, column=1, sticky="n")

        # Create controls
        self._create_controls()

        # Create info display
        self._create_info_display()

        # Draw the initial board
        self._draw_board()
        self._update_info_display()

    def debug_print(self, message):
        """Print debug messages if debug flag is enabled."""
        if self.debug:
            print(message)

    def _load_piece_images(self):
        """Load chess piece images."""
        # Check for piece images in the assets directory
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        print(f"Looking for chess piece images in: {os.path.abspath(assets_dir)}")

        # If no images found, use text representation
        self.use_text_pieces = True
        self.piece_images = {}

        # List all files in the assets directory
        if os.path.exists(assets_dir):
            files = os.listdir(assets_dir)
            print(f"Files found in assets directory: {files}")
        else:
            print(f"Assets directory not found: {assets_dir}")
            return

        # Try to load images if they exist
        try:
            loaded_count = 0
            for color in [chess.WHITE, chess.BLACK]:
                color_name = "white" if color == chess.WHITE else "black"
                for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                    # Map piece types to filenames
                    piece_map = {
                        chess.PAWN: "pawn",
                        chess.KNIGHT: "knight",
                        chess.BISHOP: "bishop",
                        chess.ROOK: "rook",
                        chess.QUEEN: "queen",
                        chess.KING: "king"
                    }
                    piece_name = piece_map[piece_type]
                    file_path = os.path.join(assets_dir, f"{color_name}_{piece_name}.png")

                    if os.path.exists(file_path):
                        print(f"Loading image: {file_path}")
                        try:
                            # Open the image
                            img = Image.open(file_path)
                            print(f"Image opened: {img.format}, {img.size}, {img.mode}")

                            # Convert to RGBA if not already to preserve transparency
                            if img.mode != 'RGBA':
                                img = img.convert('RGBA')
                                print(f"Converted image to RGBA mode")

                            # Simply resize the image, preserving transparency
                            img = img.resize((self.SQUARE_SIZE, self.SQUARE_SIZE))
                            print(f"Resized image to {self.SQUARE_SIZE}x{self.SQUARE_SIZE}")

                            # Convert to PhotoImage and store
                            photo = ImageTk.PhotoImage(img)
                            self.piece_images[(piece_type, color)] = photo
                            loaded_count += 1
                            print(f"Successfully loaded image for {color_name}_{piece_name}.png")
                        except Exception as e:
                            print(f"Error processing image {file_path}: {e}")
                    else:
                        print(f"Image file not found: {file_path}")

            # Only use images if we loaded all 12 pieces
            if loaded_count == 12:
                self.use_text_pieces = False
                print("All chess piece images loaded successfully. Using graphical pieces.")
            else:
                print(f"Only loaded {loaded_count}/12 piece images. Falling back to text representation.")
        except Exception as e:
            print(f"Error in image loading process: {e}")
            self.use_text_pieces = True

    def _create_controls(self):
        """Create control buttons and widgets."""
        # Game control buttons
        ttk.Label(self.control_panel, text="Game Controls", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Button(self.control_panel, text="New Game", command=self._new_game).grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        ttk.Button(self.control_panel, text="Setup Position", command=self._toggle_setup_mode).grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # Setup controls (initially hidden)
        self.setup_frame = ttk.LabelFrame(self.control_panel, text="Setup Controls")
        self.setup_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        self.setup_frame.grid_remove()  # Hide initially

        # We'll create the piece palette canvas directly on the main canvas when entering setup mode
        self.piece_palette_canvas = None
        self.piece_palette_visible = False

        # Variables for drag and drop
        self.dragged_piece = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_piece_type = None
        self.drag_piece_color = None

        # Side to move selection
        ttk.Label(self.setup_frame, text="Side to Move:").grid(row=1, column=0, sticky="w")

        self.side_to_move_var = tk.StringVar(value="White")
        ttk.Radiobutton(self.setup_frame, text="White", variable=self.side_to_move_var, value="White").grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(self.setup_frame, text="Black", variable=self.side_to_move_var, value="Black").grid(row=2, column=1, sticky="w")

        # Setup action buttons
        ttk.Button(self.setup_frame, text="Clear Board", command=self._clear_board).grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(self.setup_frame, text="Standard Position", command=self._standard_position).grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(self.setup_frame, text="Flip Board", command=self._flip_board).grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(self.setup_frame, text="Done Setup", command=self._done_setup).grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")

        # Engine controls
        ttk.Separator(self.control_panel, orient="horizontal").grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Label(self.control_panel, text="Engine Controls", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=5)

        # Analysis buttons
        self.start_analysis_button = ttk.Button(self.control_panel, text="Start Analysis", command=self._start_analysis)
        self.start_analysis_button.grid(row=6, column=0, pady=5, sticky="ew")

        self.stop_analysis_button = ttk.Button(self.control_panel, text="Stop Analysis", command=self._stop_analysis, state="disabled")
        self.stop_analysis_button.grid(row=6, column=1, pady=5, sticky="ew")

        # Analysis info
        self.analysis_frame = ttk.LabelFrame(self.control_panel, text="Analysis Info")
        self.analysis_frame.grid(row=7, column=0, columnspan=2, pady=5, sticky="ew")

        ttk.Label(self.analysis_frame, text="Best Move:").grid(row=0, column=0, sticky="w")
        self.best_move_label = ttk.Label(self.analysis_frame, text="-")
        self.best_move_label.grid(row=0, column=1, sticky="w")

        ttk.Label(self.analysis_frame, text="Evaluation:").grid(row=1, column=0, sticky="w")
        self.eval_label = ttk.Label(self.analysis_frame, text="-")
        self.eval_label.grid(row=1, column=1, sticky="w")

        # Quit button
        ttk.Separator(self.control_panel, orient="horizontal").grid(row=8, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Button(self.control_panel, text="Quit", command=self._quit).grid(row=9, column=0, columnspan=2, pady=5, sticky="ew")

    def _create_info_display(self):
        """Create the game information display."""
        ttk.Label(self.info_panel, text="Game Information", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        # Current player
        ttk.Label(self.info_panel, text="Current Player:").grid(row=1, column=0, sticky="w")
        self.current_player_label = ttk.Label(self.info_panel, text="White")
        self.current_player_label.grid(row=1, column=1, sticky="w")

        # Move counts
        ttk.Label(self.info_panel, text="White Moves:").grid(row=2, column=0, sticky="w")
        self.white_moves_label = ttk.Label(self.info_panel, text="0/6")
        self.white_moves_label.grid(row=2, column=1, sticky="w")

        ttk.Label(self.info_panel, text="Black Moves:").grid(row=3, column=0, sticky="w")
        self.black_moves_label = ttk.Label(self.info_panel, text="0/6")
        self.black_moves_label.grid(row=3, column=1, sticky="w")

        # Extra moves
        ttk.Label(self.info_panel, text="White Extra Moves:").grid(row=4, column=0, sticky="w")
        self.white_extra_label = ttk.Label(self.info_panel, text="0")
        self.white_extra_label.grid(row=4, column=1, sticky="w")

        ttk.Label(self.info_panel, text="Black Extra Moves:").grid(row=5, column=0, sticky="w")
        self.black_extra_label = ttk.Label(self.info_panel, text="0")
        self.black_extra_label.grid(row=5, column=1, sticky="w")

        # Scores
        ttk.Label(self.info_panel, text="White Score:").grid(row=6, column=0, sticky="w")
        self.white_score_label = ttk.Label(self.info_panel, text="0")
        self.white_score_label.grid(row=6, column=1, sticky="w")

        ttk.Label(self.info_panel, text="Black Score:").grid(row=7, column=0, sticky="w")
        self.black_score_label = ttk.Label(self.info_panel, text="0")
        self.black_score_label.grid(row=7, column=1, sticky="w")

        # Game status
        ttk.Label(self.info_panel, text="Game Status:").grid(row=8, column=0, sticky="w")
        self.game_status_label = ttk.Label(self.info_panel, text="In Progress")
        self.game_status_label.grid(row=8, column=1, sticky="w")

        # Last move
        ttk.Label(self.info_panel, text="Last Move:").grid(row=9, column=0, sticky="w")
        self.last_move_label = ttk.Label(self.info_panel, text="-")
        self.last_move_label.grid(row=9, column=1, sticky="w")

    def _draw_board(self):
        """Draw the chess board with pieces."""
        # Store the current dragged piece ID if it exists
        dragged_piece_id = self.dragged_piece

        # Use update_idletasks to process any pending UI events before redrawing
        # This helps prevent UI lag by ensuring previous operations are completed
        self.root.update_idletasks()

        # Delete board elements but preserve the palette and dragged piece
        self.canvas.delete("board")
        self.canvas.delete("highlight")
        self.canvas.delete("coordinates")

        # Only delete pieces that are not being dragged
        if dragged_piece_id:
            # Get all pieces
            all_pieces = self.canvas.find_withtag("pieces")
            # Delete each piece that is not the dragged piece
            for piece_id in all_pieces:
                if piece_id != dragged_piece_id:
                    self.canvas.delete(piece_id)
        else:
            # No dragged piece, delete all pieces
            self.canvas.delete("pieces")

        # Draw the board squares
        for row in range(8):
            for col in range(8):
                # Adjust coordinates based on board orientation
                if self.board_flipped:
                    # Black at bottom - flip both row and column
                    x1 = (7 - col) * self.SQUARE_SIZE + 20  # Offset for coordinates
                    y1 = row * self.SQUARE_SIZE + 20
                else:
                    # White at bottom - normal orientation
                    x1 = col * self.SQUARE_SIZE + 20  # Offset for coordinates
                    y1 = (7 - row) * self.SQUARE_SIZE + 20  # Flip rows for correct orientation

                x2 = x1 + self.SQUARE_SIZE
                y2 = y1 + self.SQUARE_SIZE

                # Determine square color - in chess, a1 (bottom-left when white is at bottom) is dark
                # and h1 (bottom-right when white is at bottom) is light
                # When the board is flipped, we need to adjust the formula to maintain the pattern
                if self.board_flipped:
                    # When flipped, we need to use the original row/col values to maintain the pattern
                    original_row = 7 - row
                    original_col = 7 - col
                    is_light_square = (original_row + original_col) % 2 == 1
                else:
                    # Normal orientation
                    is_light_square = (row + col) % 2 == 1

                square_color = self.LIGHT_SQUARE_COLOR if is_light_square else self.DARK_SQUARE_COLOR

                # Highlight selected square and legal moves
                square = chess.square(col, row)
                if square == self.selected_square:
                    square_color = self.SELECTED_SQUARE_COLOR

                # Highlight squares for legal moves
                for move in self.legal_moves_for_selected:
                    if move.to_square == square:
                        # Use a different color for legal move targets
                        square_color = "#aaffaa"  # Light green

                # Highlight last move
                if self.game.last_move:
                    if square == self.game.last_move.from_square or square == self.game.last_move.to_square:
                        square_color = self.LAST_MOVE_COLOR

                # Draw the square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=square_color, outline="", tags="board")

                # Add a dot indicator for legal moves
                for move in self.legal_moves_for_selected:
                    if move.to_square == square and square != self.selected_square:
                        # Draw a dot in the center of the square to indicate a legal move
                        center_x = x1 + self.SQUARE_SIZE // 2
                        center_y = y1 + self.SQUARE_SIZE // 2
                        radius = self.SQUARE_SIZE // 8

                        # Use a contrasting color for the dot
                        dot_color = "#333333" if (row + col) % 2 == 0 else "#DDDDDD"

                        self.canvas.create_oval(
                            center_x - radius, center_y - radius,
                            center_x + radius, center_y + radius,
                            fill=dot_color, outline="", tags="highlight"
                        )

                # Draw the piece
                piece = self.game.board.piece_at(square)
                if piece:
                    self._draw_piece(piece.piece_type, piece.color, x1, y1)

        # Draw coordinates
        for i in range(8):
            if self.board_flipped:
                # File labels (a-h) - reversed for black at bottom
                x = (7 - i) * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2
                y = 8 * self.SQUARE_SIZE + 30
                self.canvas.create_text(x, y, text=chr(97 + i), font=("Arial", 12), tags="coordinates")

                # Rank labels (1-8) - reversed for black at bottom
                x = 10
                y = i * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2
                self.canvas.create_text(x, y, text=str(i + 1), font=("Arial", 12), tags="coordinates")
            else:
                # File labels (a-h) - normal for white at bottom
                x = i * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2
                y = 8 * self.SQUARE_SIZE + 30
                self.canvas.create_text(x, y, text=chr(97 + i), font=("Arial", 12), tags="coordinates")

                # Rank labels (1-8) - normal for white at bottom
                x = 10
                y = (7 - i) * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2
                self.canvas.create_text(x, y, text=str(i + 1), font=("Arial", 12), tags="coordinates")

        # Draw best move arrow if available
        if self.best_move_arrow:
            from_square, to_square = self.best_move_arrow
            self._draw_arrow(from_square, to_square)

    def _draw_piece(self, piece_type, color, x, y):
        """
        Draw a chess piece on the board.

        Args:
            piece_type: Type of chess piece
            color: Color of the piece
            x, y: Top-left coordinates of the square
        """
        # Use cached values for better performance
        color_name = "white" if color == chess.WHITE else "black"
        symbol = self.PIECE_SYMBOLS[piece_type]

        # Map piece types to names for error reporting
        piece_map = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king"
        }
        piece_name = piece_map[piece_type]

        # Get the square color to determine if we need to add an outline
        square_row = (y - 20) // self.SQUARE_SIZE
        square_col = (x - 20) // self.SQUARE_SIZE

        # Determine if the square is dark using the same formula as in _draw_board
        # This ensures consistency with the board coloring
        if self.board_flipped:
            original_row = 7 - square_row
            original_col = 7 - square_col
            is_light_square = (original_row + original_col) % 2 == 1
        else:
            is_light_square = (square_row + square_col) % 2 == 1

        is_dark_square = not is_light_square

        # Check if we should use images and if the image exists
        if not self.use_text_pieces:
            key = (piece_type, color)
            if key in self.piece_images:
                # Draw using image
                try:
                    # No special handling needed for black pieces on dark squares
                    # The original images should have proper transparency

                    # Draw the piece image
                    self.canvas.create_image(
                        x + self.SQUARE_SIZE // 2,
                        y + self.SQUARE_SIZE // 2,
                        image=self.piece_images[key],
                        tags="pieces"
                    )
                    return  # Successfully drew the image
                except Exception as e:
                    print(f"Error drawing image for {color_name}_{piece_name}: {e}")
                    # Fall back to text representation
            else:
                print(f"Image not found for {color_name}_{piece_name}")

        # Draw using text as fallback
        # Use the chess symbol (P, N, B, R, Q, K)
        if color == chess.BLACK:
            symbol = symbol.lower()

        # For black pieces on dark squares, add a white background
        if color == chess.BLACK and is_dark_square:
            # Draw a white circle as background
            self.canvas.create_oval(
                x + 10,
                y + 10,
                x + self.SQUARE_SIZE - 10,
                y + self.SQUARE_SIZE - 10,
                fill="white",
                outline=""
            )

            # Draw the text in black
            self.canvas.create_text(
                x + self.SQUARE_SIZE // 2,
                y + self.SQUARE_SIZE // 2,
                text=symbol,
                font=("Arial", 24, "bold"),
                fill="black",
                tags="pieces"
            )
        else:
            # Draw the text normally
            self.canvas.create_text(
                x + self.SQUARE_SIZE // 2,
                y + self.SQUARE_SIZE // 2,
                text=symbol,
                font=("Arial", 24, "bold"),
                fill="white" if color == chess.WHITE else "black",
                tags="pieces"
            )

    def _draw_arrow(self, from_square, to_square):
        """
        Draw an arrow from one square to another.

        Args:
            from_square: Source square
            to_square: Destination square
        """
        # Calculate center coordinates of the squares
        from_col, from_row = chess.square_file(from_square), chess.square_rank(from_square)
        to_col, to_row = chess.square_file(to_square), chess.square_rank(to_square)

        from_x = from_col * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2
        from_y = (7 - from_row) * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2
        to_x = to_col * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2
        to_y = (7 - to_row) * self.SQUARE_SIZE + 20 + self.SQUARE_SIZE // 2

        # Draw the arrow
        self.canvas.create_line(
            from_x, from_y, to_x, to_y,
            fill="#FF0000",
            width=3,
            arrow=tk.LAST,
            arrowshape=(16, 20, 6),
            tags="highlight"
        )

    def _update_info_display(self):
        """Update the game information display."""
        status = self.game.get_game_status()

        # Update current player
        self.current_player_label.config(text=status["current_player"])

        # Update move counts
        white_moves = f"{status['white_moves']['made']}/{6 + status['white_moves']['extra']}"
        black_moves = f"{status['black_moves']['made']}/{6 + status['black_moves']['extra']}"
        self.white_moves_label.config(text=white_moves)
        self.black_moves_label.config(text=black_moves)

        # Update extra moves
        self.white_extra_label.config(text=str(status["white_moves"]["extra"]))
        self.black_extra_label.config(text=str(status["black_moves"]["extra"]))

        # Update scores
        self.white_score_label.config(text=str(status["scores"]["white"]))
        self.black_score_label.config(text=str(status["scores"]["black"]))

        # Update game status
        if status["game_over"]:
            if status["draw"]:
                self.game_status_label.config(text="Draw")
            else:
                self.game_status_label.config(text=f"{status['winner']} wins")
        else:
            self.game_status_label.config(text="In Progress")

        # Update last move
        if status["last_move"]:
            move_uci = status["last_move"].uci()
            self.last_move_label.config(text=move_uci)
        else:
            self.last_move_label.config(text="-")

    def _on_square_clicked(self, event):
        """
        Handle mouse click on the board.

        Args:
            event: Mouse event
        """
        # Convert click coordinates to board square
        board_x = event.x - 20  # Offset for coordinates
        board_y = event.y - 20

        if self.board_flipped:
            # Black at bottom - flip both row and column
            col = 7 - (board_x // self.SQUARE_SIZE)
            row = board_y // self.SQUARE_SIZE
        else:
            # White at bottom - normal orientation
            col = board_x // self.SQUARE_SIZE
            row = 7 - (board_y // self.SQUARE_SIZE)  # Flip for chess coordinates

        # Check if click is within board bounds
        if not (0 <= col < 8 and 0 <= row < 8):
            return

        square = chess.square(col, row)
        print(f"Clicked on square: {chess.square_name(square)}")

        # In setup mode, we only handle drag and drop in this method
        if self.setup_mode:
            return

        # In play mode, handle move selection
        if self.selected_square is None:
            # First click - select a piece
            piece = self.game.board.piece_at(square)
            if piece and piece.color == self.game.board.turn:
                self.selected_square = square
                print(f"Selected piece at {chess.square_name(square)}")

                # Get all legal moves for this piece
                legal_moves = self.game.get_legal_moves()
                print(f"All legal moves: {[m.uci() for m in legal_moves]}")

                self.legal_moves_for_selected = [move for move in legal_moves
                                               if move.from_square == square]
                print(f"Legal moves for selected piece: {[m.uci() for m in self.legal_moves_for_selected]}")
                print(f"Legal move destinations: {[chess.square_name(m.to_square) for m in self.legal_moves_for_selected]}")

                self._draw_board()
        else:
            # Second click - try to make a move
            if square == self.selected_square:
                # Clicked the same square, deselect
                self.selected_square = None
                self.legal_moves_for_selected = []  # Clear legal moves
                self._draw_board()
            else:
                # Check if the clicked square contains a piece of the same color
                piece = self.game.board.piece_at(square)
                if piece and piece.color == self.game.board.turn:
                    # Clicked on another piece of the same color, select it instead
                    self.selected_square = square
                    self.legal_moves_for_selected = [move for move in self.game.get_legal_moves()
                                                  if move.from_square == square]
                    self._draw_board()
                    return

                # Check if this is a legal move for the selected piece
                is_legal = False
                selected_move = None

                # Debug output
                self.debug_print(f"Click-based move: from {chess.square_name(self.selected_square)} to {chess.square_name(square)}")
                self.debug_print(f"Selected square: {self.selected_square}, Target square: {square}")
                self.debug_print(f"Board turn: {self.game.board.turn}")

                # Get fresh legal moves
                all_legal_moves = self.game.get_legal_moves()
                self.legal_moves_for_selected = [move for move in all_legal_moves
                                              if move.from_square == self.selected_square]

                print(f"Checking if move from {chess.square_name(self.selected_square)} to {chess.square_name(square)} is legal")
                print(f"Legal moves for selected piece: {[m.uci() for m in self.legal_moves_for_selected]}")
                print(f"Legal move destinations: {[chess.square_name(m.to_square) for m in self.legal_moves_for_selected]}")

                for move in self.legal_moves_for_selected:
                    if move.to_square == square:
                        is_legal = True
                        selected_move = move  # Store the actual legal move object
                        print(f"Found legal move: {move.uci()}")
                        break

                if not is_legal:
                    print(f"Move to {chess.square_name(square)} is not legal for piece at {chess.square_name(self.selected_square)}")
                    print(f"Legal moves are: {[chess.square_name(m.to_square) for m in self.legal_moves_for_selected]}")
                    messagebox.showerror("Invalid Move", "That move is not allowed.")
                    return

                # Use the pre-existing move object from legal_moves_for_selected
                move = selected_move
                print(f"Attempting move from {chess.square_name(self.selected_square)} to {chess.square_name(square)} with move {move.uci()}")

                # Check if this is a promotion move
                if move.promotion is not None:
                    # This is already a promotion move, no need to create a new one
                    print(f"This is a promotion move to {chess.piece_symbol(move.promotion)}")
                elif self.game.board.piece_type_at(self.selected_square) == chess.PAWN:
                    # Check if this should be a promotion move
                    if (self.game.board.turn == chess.WHITE and chess.square_rank(square) == 7) or \
                       (self.game.board.turn == chess.BLACK and chess.square_rank(square) == 0):
                        # Promotion needed
                        promotion = self._get_promotion_choice()
                        if promotion:
                            move = chess.Move(self.selected_square, square, promotion)
                            print(f"Created promotion move to {chess.piece_symbol(promotion)}")

                # Try to make the move
                print(f"Attempting to make move: {move.uci()}")
                result = self.game.make_move(move)
                print(f"Move result: {result}")

                if result:
                    # Clear best move arrow after a move is made
                    self.best_move_arrow = None

                    # Update the game information display
                    self._update_info_display()

                    # Check for game over
                    status = self.game.get_game_status()
                    if status["game_over"]:
                        self._show_game_result(status)

                    print(f"Move successful!")

                    # If analysis was running, continue it on the new position
                    was_analyzing = self.analysis_running
                    if was_analyzing:
                        self._stop_analysis()
                        # Restart analysis after a short delay to allow the UI to update
                        self.root.after(100, self._start_analysis)

                    # Move successful
                    self.selected_square = None
                    self.legal_moves_for_selected = []  # Clear legal moves
                    self._draw_board()
                else:
                    # Invalid move
                    self.selected_square = None
                    self._draw_board()
                    messagebox.showerror("Invalid Move", "That move is not allowed.")

    def _on_double_click(self, event):
        """
        Handle double-click on the board. In setup mode, this removes a piece.

        Args:
            event: Mouse event
        """
        # Only handle double-clicks in setup mode
        if not self.setup_mode:
            return

        # Convert click coordinates to board square
        board_x = event.x - 20  # Offset for coordinates
        board_y = event.y - 20

        if self.board_flipped:
            # Black at bottom - flip both row and column
            col = 7 - (board_x // self.SQUARE_SIZE)
            row = board_y // self.SQUARE_SIZE
        else:
            # White at bottom - normal orientation
            col = board_x // self.SQUARE_SIZE
            row = 7 - (board_y // self.SQUARE_SIZE)  # Flip for chess coordinates

        # Check if click is within board bounds
        if not (0 <= col < 8 and 0 <= row < 8):
            return

        square = chess.square(col, row)

        # Check if there's a piece at this square
        if self.game.board.piece_at(square):
            # Remove the piece
            self.game.place_piece(square, None, chess.WHITE)  # None removes the piece

            # Use after method to delay redraw slightly, allowing UI to remain responsive
            self.root.after(1, self._draw_board)

            # Play a sound or provide visual feedback - only if debug is enabled
            if self.debug:
                print(f"Piece removed from {chess.square_name(square)}")

    def _get_promotion_choice(self) -> Optional[int]:
        """
        Show a dialog to choose promotion piece.

        Returns:
            Chess piece type for promotion or None if cancelled
        """
        options = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }

        choice = simpledialog.askstring(
            "Promotion",
            "Choose promotion piece (Q, R, B, N):",
            initialvalue="Q"
        )

        if not choice:
            return None

        choice = choice.upper()
        if choice == "Q":
            return chess.QUEEN
        elif choice == "R":
            return chess.ROOK
        elif choice == "B":
            return chess.BISHOP
        elif choice == "N":
            return chess.KNIGHT
        else:
            return chess.QUEEN  # Default to queen

    # Removed _on_piece_selected method as we now use drag and drop

    def _new_game(self):
        """Start a new game."""
        # Stop analysis if running
        if self.analysis_running:
            self._stop_analysis()

        self.game.reset_game()
        self.selected_square = None
        self.best_move_arrow = None
        self._draw_board()
        self._update_info_display()

    def _toggle_setup_mode(self):
        """Toggle setup mode on/off."""
        # If turning on setup mode and analysis is running, stop it
        if not self.setup_mode and self.analysis_running:
            self._stop_analysis()

        # Make sure the engine is stopped when entering setup mode
        if not self.setup_mode and self.engine.engine:
            try:
                self.engine.stop()
                # Wait a moment to ensure the engine is fully stopped
                time.sleep(0.5)
            except Exception as e:
                print(f"Error stopping engine: {e}")

        self.setup_mode = not self.setup_mode

        if self.setup_mode:
            self.setup_frame.grid()
            # Create or show the piece palette
            if not self.piece_palette_visible:
                self._create_piece_palette()

            # Adjust the canvas size to accommodate the palette
            self.canvas.config(height=self.BOARD_SIZE + 60 + 140)  # Add space for palette
        else:
            self.setup_frame.grid_remove()
            # Hide the piece palette by clearing it
            self.canvas.delete("palette_eraser")
            self.canvas.delete("palette_piece_*")
            self.piece_palette_visible = False

            # Reset the canvas size
            self.canvas.config(height=self.BOARD_SIZE + 40)

        self.selected_square = None
        self.best_move_arrow = None
        self._draw_board()

    def _create_piece_palette(self):
        """Create the draggable piece palette at the bottom of the board."""
        # Calculate dimensions for the palette
        palette_width = self.BOARD_SIZE + 40  # Same width as the board including coordinates
        palette_height = 140  # Two rows of pieces with more vertical space

        # Create a frame at the bottom of the canvas for the palette
        palette_x = 0
        palette_y = self.BOARD_SIZE + 60  # Position below the board with more space

        # Create a rectangle on the main canvas to serve as the palette background
        self.canvas.create_rectangle(
            palette_x, palette_y,
            palette_x + palette_width, palette_y + palette_height,
            fill="#EEEEEE", outline="black"
        )

        # Piece types to display
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        # Calculate spacing for better separation - use fixed spacing
        piece_spacing = 80  # Fixed spacing between pieces
        start_x = 60  # Start position for the first piece

        # Draw the pieces on the palette
        for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
            y = palette_y + (color_idx * 80) + 30  # Much more vertical space between rows

            for i, piece_type in enumerate(piece_types):
                x = start_x + i * piece_spacing  # Fixed spacing

                # Draw the piece
                self._draw_piece_on_palette(piece_type, color, x, y)

            # Add an "eraser" (empty square) at the end
            x = start_x + len(piece_types) * piece_spacing
            # Create a unique tag for the eraser
            eraser_tag = f"palette_eraser_{color}"
            # Just use an X for the eraser without a box
            self.canvas.create_text(
                x, y, text="X", font=("Arial", 24, "bold"), tags=["palette_eraser", eraser_tag]
            )
            # Bind the eraser to clear pieces
            self.canvas.tag_bind(eraser_tag, "<ButtonPress-1>",
                               lambda event, c=color:
                               self._palette_piece_clicked(event, None, c))

        # Store the palette dimensions for hit testing
        self.palette_bounds = (palette_x, palette_y, palette_x + palette_width, palette_y + palette_height)
        self.piece_palette_visible = True

    def _draw_piece_on_palette(self, piece_type, color, x, y):
        """Draw a piece on the palette area of the main canvas."""
        # Create a unique tag for this piece
        tag_id = f"palette_piece_{piece_type}_{color}"

        # Draw the piece without a background square
        if not self.use_text_pieces and (piece_type, color) in self.piece_images:
            # Draw using image
            self.canvas.create_image(
                x, y, image=self.piece_images[(piece_type, color)], tags=tag_id
            )
        else:
            # Draw using text
            symbol = self.PIECE_SYMBOLS[piece_type]
            if color == chess.BLACK:
                symbol = symbol.lower()

            self.canvas.create_text(
                x, y, text=symbol, font=("Arial", 20, "bold"),
                fill="white" if color == chess.WHITE else "black", tags=tag_id
            )

        # Store the piece info in the tag dictionary for lookup during drag and drop
        self.canvas.tag_bind(tag_id, "<ButtonPress-1>",
                            lambda event, pt=piece_type, c=color:
                            self._palette_piece_clicked(event, pt, c))

    def _palette_piece_clicked(self, event, piece_type, color):
        """Handle click on a piece in the palette."""
        # Start the drag operation
        self.drag_piece_type = piece_type
        self.drag_piece_color = color
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        # Create a visual representation of the dragged piece
        if self.dragged_piece:
            self.canvas.delete(self.dragged_piece)

        if piece_type is not None:
            # Create a piece image for dragging
            if not self.use_text_pieces and (piece_type, color) in self.piece_images:
                self.dragged_piece = self.canvas.create_image(
                    event.x, event.y, image=self.piece_images[(piece_type, color)],
                    tags="dragged_piece"
                )
            else:
                # Use text representation
                symbol = self.PIECE_SYMBOLS[piece_type]
                if color == chess.BLACK:
                    symbol = symbol.lower()

                self.dragged_piece = self.canvas.create_text(
                    event.x, event.y, text=symbol, font=("Arial", 24, "bold"),
                    fill="white" if color == chess.WHITE else "black",
                    tags="dragged_piece"
                )
        else:
            # Create an eraser icon for dragging
            self.dragged_piece = self.canvas.create_rectangle(
                event.x - 15, event.y - 15, event.x + 15, event.y + 15,
                fill="white", outline="black", tags="dragged_piece"
            )
            self.canvas.create_text(
                event.x, event.y, text="X", font=("Arial", 12, "bold"),
                tags="dragged_piece"
            )

    def _canvas_mouse_down(self, event):
        """Handle mouse down on the canvas."""
        x, y = event.x, event.y
        print(f"Canvas mouse down at ({x}, {y})")

        # Check if we're in setup mode
        if self.setup_mode:
            # In setup mode, check if we're clicking on a piece on the board
            board_x = x - 20  # Offset for coordinates
            board_y = y - 20

            # Check if we're clicking on the palette area
            if self.palette_bounds and (
                self.palette_bounds[0] <= x <= self.palette_bounds[2] and
                self.palette_bounds[1] <= y <= self.palette_bounds[3]
            ):
                # Clicking in the palette area - this is handled by the palette piece click bindings
                return

            # Check if we're clicking on the board
            if self.board_flipped:
                # Black at bottom - flip both row and column
                col = 7 - (board_x // self.SQUARE_SIZE)
                row = board_y // self.SQUARE_SIZE
            else:
                # White at bottom - normal orientation
                col = board_x // self.SQUARE_SIZE
                row = 7 - (board_y // self.SQUARE_SIZE)  # Flip for chess coordinates

            if 0 <= col < 8 and 0 <= row < 8:
                square = chess.square(col, row)
                piece = self.game.board.piece_at(square)

                if piece:
                    # Store the piece information for dragging
                    self.drag_piece_type = piece.piece_type
                    self.drag_piece_color = piece.color
                    self.drag_start_square = square
                    self.drag_start_x = x
                    self.drag_start_y = y

                    # Create a visual representation of the dragged piece
                    if self.dragged_piece:
                        self.canvas.delete(self.dragged_piece)

                    # Create a piece image for dragging
                    if not self.use_text_pieces and (piece.piece_type, piece.color) in self.piece_images:
                        self.dragged_piece = self.canvas.create_image(
                            x, y, image=self.piece_images[(piece.piece_type, piece.color)],
                            tags="dragged_piece"
                        )
                    else:
                        # Use text representation
                        symbol = self.PIECE_SYMBOLS[piece.piece_type]
                        if piece.color == chess.BLACK:
                            symbol = symbol.lower()

                        self.dragged_piece = self.canvas.create_text(
                            x, y, text=symbol, font=("Arial", 24, "bold"),
                            fill="white" if piece.color == chess.WHITE else "black",
                            tags="dragged_piece"
                        )

                    # Remove the piece from the board (will be placed again on mouse up)
                    self.game.remove_piece(square)
                    self._draw_board()
        else:
            # In play mode, check if we're clicking on a piece that can be moved
            board_x = x - 20  # Offset for coordinates
            board_y = y - 20

            col = board_x // self.SQUARE_SIZE
            row = 7 - (board_y // self.SQUARE_SIZE)  # Flip for chess coordinates

            if 0 <= col < 8 and 0 <= row < 8:
                square = chess.square(col, row)
                piece = self.game.board.piece_at(square)

                # Check if this is a valid piece to move (belongs to current player)
                if piece and piece.color == self.game.board.turn:
                    print(f"Valid piece at {chess.square_name(square)}: {piece.symbol()}")
                    # Store the piece information for potential dragging
                    self.potential_drag_square = square
                    self.potential_drag_x = x
                    self.potential_drag_y = y

                    # Set UI busy flag to pause analysis during move
                    if hasattr(self, 'ui_busy'):
                        self.ui_busy = True

                    # Schedule a delayed activation of drag mode
                    # This helps distinguish between a click and a drag
                    self.drag_activation_id = self.root.after(150, self._activate_drag_mode)

                    # We'll create the visual representation when drag mode is activated

                    # Highlight the source square
                    self.selected_square = square

                    # Get all legal moves for this piece
                    all_legal_moves = self.game.get_legal_moves()
                    self.legal_moves_for_selected = [move for move in all_legal_moves
                                                  if move.from_square == square]
                    print(f"Legal moves for selected piece: {[chess.square_name(m.to_square) for m in self.legal_moves_for_selected]}")

                    self._draw_board()

                    # Raise the dragged piece to the top to ensure it's visible
                    self.canvas.tag_raise("dragged_piece")
                    return

            # If we didn't start dragging, use the original click handler
            self._on_square_clicked(event)
            return

        # In setup mode
        # Check if the click is in the palette area
        px1, py1, px2, py2 = self.palette_bounds
        if px1 <= x <= px2 and py1 <= y <= py2:
            # Click is in the palette area - handled by tag bindings
            pass
        else:
            # Click is on the board - handle as before
            self._on_square_clicked(event)

    def _activate_drag_mode(self):
        """Activate drag mode after a short delay."""
        # Only activate if we have a potential drag square
        if hasattr(self, 'potential_drag_square') and self.potential_drag_square is not None:
            # Transfer from potential to actual drag
            self.drag_start_square = self.potential_drag_square
            self.drag_start_x = self.potential_drag_x
            self.drag_start_y = self.potential_drag_y

            # Always delete any existing dragged piece to ensure we create a fresh one
            if self.dragged_piece:
                self.canvas.delete(self.dragged_piece)
                self.dragged_piece = None

            # Create the dragged piece visual
            square = self.drag_start_square
            piece = self.game.board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                color = piece.color

                if not self.use_text_pieces and (piece_type, color) in self.piece_images:
                    self.dragged_piece = self.canvas.create_image(
                        self.drag_start_x, self.drag_start_y,
                        image=self.piece_images[(piece_type, color)],
                        tags="dragged_piece"
                    )
                else:
                    # Use text representation
                    symbol = self.PIECE_SYMBOLS[piece_type]
                    if color == chess.BLACK:
                        symbol = symbol.lower()

                    self.dragged_piece = self.canvas.create_text(
                        self.drag_start_x, self.drag_start_y,
                        text=symbol, font=("Arial", 24, "bold"),
                        fill="white" if color == chess.WHITE else "black",
                        tags="dragged_piece"
                    )

                # Ensure the dragged piece is on top
                self.canvas.tag_raise("dragged_piece")

                print(f"Drag mode activated for piece at {chess.square_name(self.drag_start_square)}")
            else:
                print(f"Error: No piece found at {chess.square_name(self.drag_start_square)}")

    def _canvas_mouse_drag(self, event):
        """Handle mouse drag on the canvas."""
        # If we're in setup mode, we don't need to delay drag activation
        if self.setup_mode and not self.dragged_piece and hasattr(self, 'drag_piece_type') and self.drag_piece_type is not None:
            # Create a visual representation of the dragged piece if it doesn't exist
            if not self.dragged_piece:
                if not self.use_text_pieces and (self.drag_piece_type, self.drag_piece_color) in self.piece_images:
                    self.dragged_piece = self.canvas.create_image(
                        event.x, event.y, image=self.piece_images[(self.drag_piece_type, self.drag_piece_color)],
                        tags="dragged_piece"
                    )
                else:
                    # Use text representation
                    symbol = self.PIECE_SYMBOLS[self.drag_piece_type]
                    if self.drag_piece_color == chess.BLACK:
                        symbol = symbol.lower()

                    self.dragged_piece = self.canvas.create_text(
                        event.x, event.y, text=symbol, font=("Arial", 24, "bold"),
                        fill="white" if self.drag_piece_color == chess.WHITE else "black",
                        tags="dragged_piece"
                    )
                self.drag_start_x = event.x
                self.drag_start_y = event.y
        # If we're in the delay period before drag activation, check if we've moved enough to start dragging
        elif not self.dragged_piece and hasattr(self, 'potential_drag_square') and self.potential_drag_square is not None:
            # Calculate distance moved
            dx = event.x - self.potential_drag_x
            dy = event.y - self.potential_drag_y
            distance = (dx**2 + dy**2)**0.5

            # If we've moved more than a few pixels, activate drag mode immediately
            if distance > 5:
                # Cancel the scheduled activation
                if hasattr(self, 'drag_activation_id') and self.drag_activation_id:
                    self.root.after_cancel(self.drag_activation_id)
                    self.drag_activation_id = None

                # Activate drag mode now
                print(f"Activating drag mode early due to movement")
                self._activate_drag_mode()

        # Handle the actual dragging
        if self.dragged_piece:
            # Move the dragged piece
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.canvas.move(self.dragged_piece, dx, dy)

            # Ensure the dragged piece is on top and visible
            self.canvas.tag_raise("dragged_piece")

            # Update the drag start position
            self.drag_start_x = event.x
            self.drag_start_y = event.y

    def _canvas_mouse_up(self, event):
        """Handle mouse up on the canvas."""
        # Get the current position
        x, y = event.x, event.y

        # Cancel any pending drag activation
        if hasattr(self, 'drag_activation_id') and self.drag_activation_id:
            self.root.after_cancel(self.drag_activation_id)
            self.drag_activation_id = None

        # Check if we have a dragged piece
        has_dragged_piece = self.dragged_piece is not None
        print(f"Mouse up with dragged piece: {has_dragged_piece}")

        # In setup mode, handle placing the piece
        if self.setup_mode and has_dragged_piece:
            # Get the board coordinates
            board_x = x - 20
            board_y = y - 20

            # Check if the coordinates are within the board
            if 0 <= board_x < 8 * self.SQUARE_SIZE and 0 <= board_y < 8 * self.SQUARE_SIZE:
                # Convert to board square
                if self.board_flipped:
                    # Black at bottom - flip both row and column
                    col = 7 - (board_x // self.SQUARE_SIZE)
                    row = board_y // self.SQUARE_SIZE
                else:
                    # White at bottom - normal orientation
                    col = board_x // self.SQUARE_SIZE
                    row = 7 - (board_y // self.SQUARE_SIZE)  # Flip for chess coordinates
                target_square = chess.square(col, row)

                # Place the piece on the board
                if hasattr(self, 'drag_piece_type') and self.drag_piece_type is not None:
                    self.game.place_piece(target_square, self.drag_piece_type, self.drag_piece_color)
                    print(f"Placed {self.drag_piece_type} at {chess.square_name(target_square)}")
                    self._draw_board()

        # Clean up any dragged piece
        if self.dragged_piece:
            self.canvas.delete("dragged_piece")
            self.dragged_piece = None

        # If we have a potential drag square, we need to handle it
        if hasattr(self, 'potential_drag_square') and self.potential_drag_square is not None:
            # Get the board coordinates
            board_x = x - 20
            board_y = y - 20

            # Check if the coordinates are within the board
            if 0 <= board_x < 8 * self.SQUARE_SIZE and 0 <= board_y < 8 * self.SQUARE_SIZE:
                # Convert to board square
                if self.board_flipped:
                    # Black at bottom - flip both row and column
                    col = 7 - (board_x // self.SQUARE_SIZE)
                    row = board_y // self.SQUARE_SIZE
                else:
                    # White at bottom - normal orientation
                    col = board_x // self.SQUARE_SIZE
                    row = 7 - (board_y // self.SQUARE_SIZE)  # Flip for chess coordinates
                target_square = chess.square(col, row)

                # If we're at the same square and there was no drag, treat it as a click
                if target_square == self.potential_drag_square and not has_dragged_piece:
                    # It was a click, not a drag - pass to the square clicked handler
                    square = self.potential_drag_square
                    print(f"Click detected on {chess.square_name(square)}")

                    # Get all legal moves for this piece
                    self.selected_square = square
                    self.legal_moves_for_selected = [move for move in self.game.get_legal_moves()
                                                  if move.from_square == square]
                    print(f"Legal moves for selected piece: {[chess.square_name(m.to_square) for m in self.legal_moves_for_selected]}")

                    # Clear the potential drag state
                    self.potential_drag_square = None
                    self.potential_drag_x = None
                    self.potential_drag_y = None

                    # Update the board to show legal moves
                    self._draw_board()
                    return
                # If we dragged to a different square, try to make a move
                elif target_square != self.potential_drag_square:
                    print(f"Drag detected from {chess.square_name(self.potential_drag_square)} to {chess.square_name(target_square)}")

                    # Set the drag_start_square for the move logic
                    self.drag_start_square = self.potential_drag_square

                    # Also set the selected_square to ensure legal moves are calculated
                    self.selected_square = self.drag_start_square

                    # Store the target square for the move
                    self.drag_target_square = target_square

                    # Check if this is a legal move
                    is_legal = False

                    # Get all legal moves for the selected piece
                    all_legal_moves = self.game.get_legal_moves()
                    self.legal_moves_for_selected = [move for move in all_legal_moves
                                                  if move.from_square == self.drag_start_square]

                    print(f"Checking if move from {chess.square_name(self.drag_start_square)} to {chess.square_name(target_square)} is legal")
                    print(f"Legal moves: {[m.uci() for m in self.legal_moves_for_selected]}")
                    print(f"Legal move destinations: {[chess.square_name(m.to_square) for m in self.legal_moves_for_selected]}")

                    # Initialize move variable
                    move = None

                    # Check if the target square is a legal destination
                    for legal_move in self.legal_moves_for_selected:
                        if legal_move.to_square == target_square:
                            is_legal = True
                            move = legal_move  # Use the pre-existing move object which might include promotion
                            print(f"Found legal move: {move.uci()}")

                            # Try to make the move
                            print(f"Attempting to make move: {move.uci()}")
                            result = self.game.make_move(move)
                            print(f"Move result: {result}")

                            if result:
                                # Clear best move arrow after a move is made
                                self.best_move_arrow = None

                                # Update the game information display
                                self._update_info_display()

                                # Check for game over
                                status = self.game.get_game_status()
                                if status["game_over"]:
                                    self._show_game_result(status)

                                # If analysis was running, continue it on the new position
                                was_analyzing = self.analysis_running
                                if was_analyzing:
                                    self._stop_analysis()
                                    # Restart analysis after a short delay to allow the UI to update
                                    self.root.after(100, self._start_analysis)

                                print(f"Move successful!")

                                # Clear the drag state
                                self.drag_start_square = None
                                self.selected_square = None
                                self.legal_moves_for_selected = []  # Clear legal moves
                                self._draw_board()
                                return
                            else:
                                # Invalid move
                                messagebox.showerror("Invalid Move", "That move is not allowed.")
                                # Reset drag state but keep the piece selected
                                self.drag_start_square = None
                                self._draw_board()
                                return

                    # If not found in legal moves, create a new move object
                    if not is_legal:
                        # Check for promotion
                        if self.game.board.piece_type_at(self.drag_start_square) == chess.PAWN:
                            if (self.game.board.turn == chess.WHITE and chess.square_rank(target_square) == 7) or \
                               (self.game.board.turn == chess.BLACK and chess.square_rank(target_square) == 0):
                                # Promotion needed
                                promotion = self._get_promotion_choice()
                                if promotion:
                                    move = chess.Move(self.drag_start_square, target_square, promotion)
                                    # Check if this promotion move is legal
                                    is_legal = move in self.game.board.legal_moves
                        else:
                            # Regular move
                            move = chess.Move(self.drag_start_square, target_square)
                            # Double-check if it's legal
                            is_legal = move in self.game.board.legal_moves

                    if not is_legal:
                        print(f"Move from {chess.square_name(self.drag_start_square)} to {chess.square_name(target_square)} is not legal")

                        # Check if the piece is still on the board
                        piece = self.game.board.piece_at(self.drag_start_square)
                        if piece and piece.color == self.game.board.turn:
                            # Piece is still on the board, show error
                            messagebox.showerror("Invalid Move", "That move is not allowed.")
                    else:
                        # Try to make the move
                        if self.debug:
                            print(f"Attempting to make move: {move.uci()}")

                        # Make the move without redrawing the board immediately
                        result = self.game.make_move(move)

                        if self.debug:
                            print(f"Move result: {result}")

                        if result:
                            # Clear best move arrow after a move is made
                            self.best_move_arrow = None

                            # Update the game information display - use after to prevent UI lag
                            self.root.after(1, self._update_info_display)

                            # Check for game over
                            status = self.game.get_game_status()
                            if status["game_over"]:
                                # Use after to show game result after the board is updated
                                self.root.after(10, lambda: self._show_game_result(status))

                            # If analysis was running, continue it on the new position
                            was_analyzing = self.analysis_running
                            if was_analyzing:
                                self._stop_analysis()
                                # Restart analysis after a delay to allow the UI to update
                                self.root.after(100, self._start_analysis)

                    # Clear the potential drag state
                    self.potential_drag_square = None
                    self.potential_drag_x = None
                    self.potential_drag_y = None
                    self.selected_square = None
                    self.legal_moves_for_selected = []  # Clear legal moves
                    self._draw_board()

        # Handle actual dragging
        if self.dragged_piece:
            # Remove the dragged piece visual
            self.canvas.delete("dragged_piece")
            self.dragged_piece = None

            # Check if the drop is on the chess board
            board_x = x - 20  # Offset for coordinates
            board_y = y - 20

            col = board_x // self.SQUARE_SIZE
            row = 7 - (board_y // self.SQUARE_SIZE)  # Flip for chess coordinates

            if 0 <= col < 8 and 0 <= row < 8:
                # Valid drop on the board
                target_square = chess.square(col, row)

                # Handle differently based on mode
                if self.setup_mode:
                    # In setup mode, place the piece on the board
                    if hasattr(self, 'drag_piece_type') and self.drag_piece_type is not None:
                        # If we're dragging from the palette or from another square
                        self.game.place_piece(target_square, self.drag_piece_type, self.drag_piece_color)
                        # Reduce debug output to improve performance
                        if self.debug:
                            print(f"Placed {self.drag_piece_type} at {chess.square_name(target_square)}")
                    elif self.drag_start_square is not None:
                        # If we're dragging from another square on the board
                        self.game.place_piece(target_square, self.drag_piece_type, self.drag_piece_color)
                        # Reduce debug output to improve performance
                        if self.debug:
                            print(f"Moved piece from {chess.square_name(self.drag_start_square)} to {chess.square_name(target_square)}")

                    # Use after method to delay redraw slightly, allowing UI to remain responsive
                    self.root.after(1, self._draw_board)
                elif self.drag_start_square is not None:
                    # In play mode
                    # Check if the target square contains a piece of the same color
                    piece = self.game.board.piece_at(target_square)
                    if piece and piece.color == self.game.board.turn and target_square != self.drag_start_square:
                        # Dragged to another piece of the same color, select it instead
                        self.drag_start_square = None
                        self.selected_square = target_square
                        self.legal_moves_for_selected = [move for move in self.game.get_legal_moves()
                                                      if move.from_square == target_square]
                        self._draw_board()
                        return

                    # Check if this is a legal move
                    is_legal = False
                    legal_moves = [move for move in self.game.get_legal_moves()
                                 if move.from_square == self.drag_start_square]

                    print(f"Checking if move from {chess.square_name(self.drag_start_square)} to {chess.square_name(target_square)} is legal")
                    print(f"Legal moves: {[chess.square_name(m.to_square) for m in legal_moves]}")

                    # Initialize move variable
                    move = None

                    # Check if the target square is a legal destination
                    for legal_move in legal_moves:
                        if legal_move.to_square == target_square:
                            is_legal = True
                            move = legal_move  # Use the pre-existing move object which might include promotion
                            print(f"Found legal move: {move.uci()}")
                            break

                    # If not found in legal moves, create a new move object
                    if not is_legal:
                        # Check for promotion
                        if self.game.board.piece_type_at(self.drag_start_square) == chess.PAWN:
                            if (self.game.board.turn == chess.WHITE and chess.square_rank(target_square) == 7) or \
                               (self.game.board.turn == chess.BLACK and chess.square_rank(target_square) == 0):
                                # Promotion needed
                                promotion = self._get_promotion_choice()
                                if promotion:
                                    move = chess.Move(self.drag_start_square, target_square, promotion)
                                    # Check if this promotion move is legal
                                    is_legal = move in self.game.board.legal_moves
                        else:
                            # Regular move
                            move = chess.Move(self.drag_start_square, target_square)
                            # Double-check if it's legal
                            is_legal = move in self.game.board.legal_moves

                    if not is_legal:
                        print(f"Move from {chess.square_name(self.drag_start_square)} to {chess.square_name(target_square)} is not legal")

                        # Check if the piece is still on the board
                        piece = self.game.board.piece_at(self.drag_start_square)
                        if piece and piece.color == self.game.board.turn:
                            # Piece is still on the board, show error
                            messagebox.showerror("Invalid Move", "That move is not allowed.")
                        # Reset drag state but keep the piece selected
                        self.drag_start_square = None
                        self._draw_board()
                        return

                    # Try to make the move
                    if self.debug:
                        print(f"Attempting to make move: {move.uci()}")

                    # Make the move without redrawing the board immediately
                    result = self.game.make_move(move)

                    if self.debug:
                        print(f"Move result: {result}")

                    if result:
                        # Clear best move arrow after a move is made
                        self.best_move_arrow = None

                        # Update the game information display - use after to prevent UI lag
                        self.root.after(1, self._update_info_display)

                        # Check for game over
                        status = self.game.get_game_status()
                        if status["game_over"]:
                            # Use after to show game result after the board is updated
                            self.root.after(10, lambda: self._show_game_result(status))

                        # If analysis was running, continue it on the new position
                        was_analyzing = self.analysis_running
                        if was_analyzing:
                            self._stop_analysis()
                            # Restart analysis after a delay to allow the UI to update
                            self.root.after(100, self._start_analysis)

                        if self.debug:
                            print(f"Move successful!")
                    else:
                        # Invalid move
                        messagebox.showerror("Invalid Move", "That move is not allowed.")

                    # Reset drag state
                    self.drag_start_square = None
                    self.selected_square = None
                    self.legal_moves_for_selected = []  # Clear legal moves

                    # Reset UI busy flag to allow analysis to continue
                    if hasattr(self, 'ui_busy'):
                        self.ui_busy = False

                    self._draw_board()

    def _clear_board(self):
        """Clear the board in setup mode."""
        self.game.setup_empty_board()
        # Use after method to delay redraw slightly, allowing UI to remain responsive
        self.root.after(1, self._draw_board)

    def _standard_position(self):
        """Set up the standard starting position."""
        self.game.setup_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        # Use after method to delay redraw slightly, allowing UI to remain responsive
        self.root.after(1, self._draw_board)

    def _flip_board(self):
        """Flip the board orientation in setup mode."""
        self.board_flipped = not self.board_flipped
        if self.debug:
            print(f"Board flipped: {self.board_flipped}")

        # Use after method to delay redraw slightly, allowing UI to remain responsive
        self.root.after(1, self._draw_board)

    def _done_setup(self):
        """Finish setup mode and start the game."""
        # Stop analysis if running
        if self.analysis_running:
            self._stop_analysis()

        # Make sure the engine is stopped and restarted to avoid issues
        if self.engine.engine:
            try:
                self.engine.stop()
                # Wait a moment to ensure the engine is fully stopped
                time.sleep(0.5)
            except Exception as e:
                print(f"Error stopping engine: {e}")

        # Set the side to move
        side_to_move = chess.WHITE if self.side_to_move_var.get() == "White" else chess.BLACK
        self.game.set_side_to_move(side_to_move)

        # Exit setup mode
        self.setup_mode = False
        self.setup_frame.grid_remove()
        self.selected_square = None
        self.best_move_arrow = None
        self._draw_board()
        self._update_info_display()

    def _start_analysis(self):
        """Start continuous analysis of the current position."""
        if self.analysis_running:
            return

        # Don't allow analysis in setup mode
        if self.setup_mode:
            messagebox.showinfo("Setup Mode", "Analysis is not available in setup mode. Please finish setup first.")
            return

        if self.game.game_over:
            messagebox.showinfo("Game Over", "The game is already over.")
            return

        # Start the engine if not already running
        if not self.engine.engine:
            try:
                self.engine.start()
            except Exception as e:
                messagebox.showerror("Engine Error", f"Failed to start Stockfish: {e}")
                return

        # Reset the stop event
        self.analysis_stop_event.clear()

        # Update UI
        self.start_analysis_button.config(state="disabled")
        self.stop_analysis_button.config(state="normal")
        self.root.update()

        # Set analysis running flag
        self.analysis_running = True

        # Add a flag to indicate when the UI is busy with a move
        self.ui_busy = False

        # Add an engine crash counter to track and handle crashes
        self.engine_crash_count = 0
        self.max_engine_restarts = 3  # Maximum number of times to restart the engine

        # Run continuous engine analysis in a separate thread
        def continuous_analysis():
            try:
                # Configure the engine for continuous analysis
                if not self.engine.engine:
                    try:
                        self.engine.start()
                    except Exception as e:
                        print(f"Failed to start engine: {e}")
                        # Signal the main thread that analysis has stopped
                        self.root.after(0, self._reset_analysis_ui)
                        return

                # Set up the engine for continuous analysis
                try:
                    # Configure the engine for intensive analysis
                    self.engine.depth = 20  # Use a deeper search for better analysis
                    self.engine.time_limit = None  # No time limit for continuous analysis
                    self.engine.nodes = 500000  # Higher node limit for more thorough analysis

                    # Start a non-blocking analysis that will continuously update
                    board = self.game.board

                    # Use multipv analysis to get multiple best moves
                    top_moves = self.engine.get_top_moves(board, num_moves=1)

                    if top_moves and top_moves[0][0]:  # If we have a valid move
                        best_move, score, info = top_moves[0]

                        # Format the evaluation
                        if score is not None:
                            if score > 9000:  # Mate score
                                eval_str = f"Mate in {10000 - abs(score)} moves"
                            elif score < -9000:  # Getting mated
                                eval_str = f"Mate in {10000 - abs(score)} moves"
                            else:
                                eval_str = f"{score/100.0:.2f}"
                        else:
                            eval_str = "?"

                        # Update the GUI from the main thread
                        self.root.after(0, lambda m=best_move, e=eval_str: self._update_analysis_display(m, e))
                except Exception as e:
                    print(f"Analysis error: {e}")
                    # If the engine crashed, try to restart it
                    if "engine process died" in str(e) or "not running" in str(e):
                        self.engine_crash_count += 1
                        if self.engine_crash_count <= self.max_engine_restarts:
                            print(f"Attempting to restart engine (attempt {self.engine_crash_count}/{self.max_engine_restarts})")
                            try:
                                # Stop the engine if it's still running
                                self.engine.stop()
                                # Start a new engine instance
                                self.engine.start()
                                print("Engine restarted successfully")
                            except Exception as restart_error:
                                print(f"Failed to restart engine: {restart_error}")
                                # Signal the main thread that analysis has stopped
                                self.root.after(0, self._reset_analysis_ui)
                                return
                        else:
                            print(f"Maximum engine restart attempts reached ({self.max_engine_restarts})")
                            # Signal the main thread that analysis has stopped
                            self.root.after(0, self._reset_analysis_ui)
                            return

                # Set up continuous analysis with UI updates every 2 seconds
                last_update_time = 0
                update_interval = 2.0  # Update UI every 2 seconds for more responsive updates
                last_analysis_time = 0
                analysis_interval = 0.2  # Run analysis every 0.2 seconds for more frequent updates

                # Configure the engine for continuous analysis
                self.engine.depth = 24  # Use a deeper search for better analysis

                # Store the current best move and evaluation
                current_best_move = None
                current_eval_str = "?"

                # Main analysis loop
                while not self.analysis_stop_event.is_set() and not self.game.game_over:
                    try:
                        current_time = time.time()

                        # Run analysis at regular intervals to get the best move
                        if current_time - last_analysis_time >= analysis_interval:
                            try:
                                # Check if engine is still running
                                if not self.engine.engine:
                                    raise Exception("Engine not running")

                                # Get the best move with a more intensive analysis
                                self.engine.depth = 22  # Increased depth for better analysis
                                self.engine.time_limit = 0.15  # Short time but more frequent updates

                                # Initialize variables
                                best_move = None
                                info = {}

                                # Always run the analysis, but only update UI when not busy
                                # This maximizes CPU usage for analysis
                                best_move, info = self.engine.get_best_move(self.game.board)

                                # Reset crash counter on successful analysis
                                self.engine_crash_count = 0
                            except Exception as e:
                                print(f"Analysis iteration error: {e}")
                                # If the engine crashed, try to restart it
                                if "engine process died" in str(e) or "not running" in str(e):
                                    self.engine_crash_count += 1
                                    if self.engine_crash_count <= self.max_engine_restarts:
                                        print(f"Attempting to restart engine (attempt {self.engine_crash_count}/{self.max_engine_restarts})")
                                        try:
                                            # Stop the engine if it's still running
                                            self.engine.stop()
                                            # Start a new engine instance
                                            self.engine.start()
                                            print("Engine restarted successfully")
                                            # Skip this iteration to allow engine to initialize
                                            time.sleep(0.5)
                                            continue
                                        except Exception as restart_error:
                                            print(f"Failed to restart engine: {restart_error}")
                                            # Signal the main thread that analysis has stopped
                                            self.root.after(0, self._reset_analysis_ui)
                                            return
                                    else:
                                        print(f"Maximum engine restart attempts reached ({self.max_engine_restarts})")
                                        # Signal the main thread that analysis has stopped
                                        self.root.after(0, self._reset_analysis_ui)
                                        return

                            # Extract evaluation
                            score = info.get("score", None)
                            if score:
                                score_value = score.white().score(mate_score=10000)
                                if score.is_mate():
                                    eval_str = f"Mate in {abs(score.white().mate())} moves"
                                else:
                                    eval_str = f"{score_value/100.0:.2f}"
                            else:
                                eval_str = "?"

                            # Update the current best move and evaluation
                            if best_move:
                                current_best_move = best_move
                                current_eval_str = eval_str

                            # Update the last analysis time
                            last_analysis_time = current_time

                        # Check if it's time to update the UI
                        if current_time - last_update_time >= update_interval and current_best_move is not None:
                            # Update the GUI from the main thread
                            self.root.after(0, lambda m=current_best_move, e=current_eval_str:
                                            self._update_analysis_display(m, e))

                            # Update the last update time
                            last_update_time = current_time

                            # Print a message to confirm the update interval
                            print(f"Analysis updated at {time.strftime('%H:%M:%S')} - Best move: {current_best_move}, Eval: {current_eval_str}")
                    except Exception as e:
                        # Only print the error if it's not a common threading error
                        if "main thread is not in main loop" not in str(e):
                            print(f"Analysis loop error: {e}")

                        # If the engine crashed, try to restart it
                        if "engine process died" in str(e) or "not running" in str(e):
                            self.engine_crash_count += 1
                            if self.engine_crash_count <= self.max_engine_restarts:
                                print(f"Attempting to restart engine (attempt {self.engine_crash_count}/{self.max_engine_restarts})")
                                try:
                                    # Stop the engine if it's still running
                                    self.engine.stop()
                                    # Start a new engine instance
                                    self.engine.start()
                                    print("Engine restarted successfully")
                                except Exception as restart_error:
                                    print(f"Failed to restart engine: {restart_error}")
                                    # Signal the main thread that analysis has stopped
                                    self.root.after(0, self._reset_analysis_ui)
                            else:
                                print(f"Maximum engine restart attempts reached ({self.max_engine_restarts})")
                                # Signal the main thread that analysis has stopped
                                self.root.after(0, self._reset_analysis_ui)

                        # Use a shorter sleep time to recover faster
                        time.sleep(0.1)

                    # Very short sleep to maintain thread responsiveness while maximizing CPU usage
                    time.sleep(0.01)  # Minimal sleep for maximum CPU utilization
            finally:
                # Reset UI when analysis stops
                self.root.after(0, self._reset_analysis_ui)

        # Start the analysis thread
        self.analysis_thread = threading.Thread(target=continuous_analysis)
        self.analysis_thread.daemon = True  # Thread will exit when main program exits
        self.analysis_thread.start()

    def _stop_analysis(self):
        """Stop the continuous analysis."""
        if not self.analysis_running:
            return

        # Signal the analysis thread to stop
        self.analysis_stop_event.set()

        # No need to explicitly stop the engine as we're using get_best_move
        # which completes each analysis call

        # Wait for the thread to finish (with timeout)
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)

        # Reset UI
        self._reset_analysis_ui()

    def _reset_analysis_ui(self):
        """Reset the UI after analysis stops."""
        self.analysis_running = False
        self.start_analysis_button.config(state="normal")
        self.stop_analysis_button.config(state="disabled")

        # Reset UI busy flag
        if hasattr(self, 'ui_busy'):
            self.ui_busy = False

        # Make sure to reset the cursor to default
        self.root.config(cursor="")
        self.canvas.config(cursor="")

        # Reset the best move display
        self.best_move_label.config(text="-")
        self.eval_label.config(text="-")

    def _update_analysis_display(self, move, evaluation):
        """
        Update the analysis display with the best move and evaluation.

        Args:
            move: Chess move to display
            evaluation: Evaluation string
        """
        if not move:
            self.best_move_label.config(text="No legal moves")
            self.eval_label.config(text="-")
            self.best_move_arrow = None
            self._draw_board()
            return

        # Only update if this is a different move or we don't have an arrow yet
        update_board = False

        # Only skip updates if UI is actively making a move
        # This allows analysis to continue running at full speed
        # but only pauses the UI updates during actual move execution
        if hasattr(self, 'ui_busy') and self.ui_busy and self.dragged_piece is not None:
            return

        if self.best_move_arrow is None or self.best_move_arrow != (move.from_square, move.to_square):
            # Update the best move arrow
            self.best_move_arrow = (move.from_square, move.to_square)
            update_board = True

        # Update the labels
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        self.best_move_label.config(text=f"{from_square}{to_square}")
        self.eval_label.config(text=evaluation)

        # Only redraw the board if the arrow changed
        if update_board:
            self._draw_board()

    def _show_game_result(self, status):
        """
        Show the game result with detailed information.

        Args:
            status: Game status dictionary
        """
        # Get the board state
        board = status["board"]
        white_score = status["scores"]["white"]
        black_score = status["scores"]["black"]

        # Determine the reason for game end
        reason = ""
        if board.is_checkmate():
            reason = "Checkmate!"
        elif board.is_stalemate():
            reason = "Stalemate!"
        elif board.is_insufficient_material():
            reason = "Insufficient material to continue!"
        else:
            # Must be due to move limit
            reason = "Both players have used all their moves."

        # Create the result message
        if status["draw"]:
            result_msg = f"The game is a draw! ({reason})\n\n"
            result_msg += f"Final score:\nWhite: {white_score}\nBlack: {black_score}"
            messagebox.showinfo("Game Over", result_msg)
        else:
            result_msg = f"{status['winner']} wins! ({reason})\n\n"
            result_msg += f"Final score:\nWhite: {white_score}\nBlack: {black_score}"
            messagebox.showinfo("Game Over", result_msg)

    def _quit(self):
        """Quit the application."""
        # Stop analysis if running
        if self.analysis_running:
            self._stop_analysis()

        # Stop the engine
        if self.engine:
            self.engine.stop()

        self.root.quit()

    def run(self):
        """Run the main application loop."""
        self.root.mainloop()
