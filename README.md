# Custom Chess Variant with Stockfish 16

A Python GUI application for playing a unique chess variant with custom rules, powered by the Stockfish 16 chess engine.

## Features

- **Custom Chess Variant Rules**:
  - Each player is limited to exactly 6 moves per game
  - Game starts from a configurable position (not the standard opening)
  - Scoring based on captured pieces if no checkmate occurs
  - Foul rule on the 6th move if capturing with a higher-value piece
  - Extra moves awarded as penalties for fouls

- **GUI Features**:
  - Interactive chess board with piece movement via mouse clicks
  - Position setup mode for creating custom starting positions
  - Move count and score tracking
  - Best move suggestions from Stockfish 16
  - Game status display

## Requirements

- Python 3.6+
- Stockfish 16 chess engine
- Required Python packages:
  - python-chess
  - stockfish
  - Pillow (for image handling)
  - tkinter (usually included with Python)

## Installation

1. Clone or download this repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Install Stockfish 16:
   - Download from [Stockfish website](https://stockfishchess.org/download/)
   - The application will offer to download Stockfish if it's not found

## Usage

Run the application:
```
python main.py
```

### Game Controls

- **New Game**: Start a new game with the current board position
- **Setup Position**: Enter setup mode to create a custom starting position
- **Show Best Move**: Ask Stockfish to suggest the best move

### Setup Mode

- Select a piece type from the dropdown menu
- Click on the board to place the selected piece
- Choose which side moves first
- Use "Clear Board" to remove all pieces
- Use "Standard Position" to set up the default chess position
- Click "Done Setup" when finished

### Playing the Game

- Click on a piece to select it, then click on a destination square to move
- The game enforces all custom variant rules automatically
- The information panel shows move counts, scores, and game status

## Rules of the Variant

1. **Move Limit**: Each player is restricted to exactly 6 moves per game, alternating turns.
2. **Starting Position**: The game begins from a balanced position, typically a middlegame or endgame setup.
3. **Scoring System**:
   - If no checkmate occurs by the end of the move limit, the winner is determined by the total point value of pieces captured.
   - Piece values: Queen = 9, Rook = 5, Bishop = 3, Knight = 3, Pawn = 1, King = 0
4. **Winning Conditions**:
   - A player wins by either delivering checkmate within their 6 moves or achieving a higher cumulative capture score.
   - If capture scores are equal and no checkmate occurs, the game is a draw.
5. **Foul Rule on the Final Move**:
   - On a player's 6th move, if they capture a piece and the capturing piece has a higher point value than the captured piece, it's a foul.
   - Penalty: The opponent is granted one extra move after completing their own 6 moves.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Stockfish](https://stockfishchess.org/) chess engine
- [python-chess](https://python-chess.readthedocs.io/) library
