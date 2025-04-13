"""
Script to download chess piece images for the GUI.
"""
import os
import urllib.request
import zipfile
import shutil
import time
import random

def download_chess_pieces():
    """Download and extract chess piece images."""
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join("chess_variant", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # URL for chess piece images (using standard SVG pieces from Wikimedia)
    url = "https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces"

    # Piece names and colors
    pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    colors = ["white", "black"]

    # Base URL for the SVG files
    base_url = "https://upload.wikimedia.org/wikipedia/commons/thumb"

    # Mapping of piece names to their Wikimedia filenames
    piece_files = {
        "white_pawn": "/4/45/Chess_plt45.svg",
        "white_knight": "/7/70/Chess_nlt45.svg",
        "white_bishop": "/b/b1/Chess_blt45.svg",
        "white_rook": "/7/72/Chess_rlt45.svg",
        "white_queen": "/1/15/Chess_qlt45.svg",
        "white_king": "/4/42/Chess_klt45.svg",
        "black_pawn": "/c/c7/Chess_pdt45.svg",
        "black_knight": "/e/ef/Chess_ndt45.svg",
        "black_bishop": "/9/98/Chess_bdt45.svg",
        "black_rook": "/f/ff/Chess_rdt45.svg",
        "black_queen": "/4/47/Chess_qdt45.svg",
        "black_king": "/f/f0/Chess_kdt45.svg"
    }

    print("Downloading chess piece images...")

    # Download each piece
    for color in colors:
        for piece in pieces:
            piece_name = f"{color}_{piece}"
            if piece_name in piece_files:
                file_path = piece_files[piece_name]
                # Construct the full URL with size (128px)
                full_url = f"{base_url}{file_path}/128px-{os.path.basename(file_path)}.png"
                output_file = os.path.join(assets_dir, f"{piece_name}.png")

                try:
                    print(f"Downloading {piece_name}...")
                    # Add a proper user agent and delay between requests
                    headers = {
                        'User-Agent': 'ChessVariantApp/1.0 (https://github.com/username/chess-variant; username@example.com)'
                    }
                    req = urllib.request.Request(full_url, headers=headers)
                    with urllib.request.urlopen(req) as response, open(output_file, 'wb') as out_file:
                        out_file.write(response.read())

                    # Add a random delay between requests to avoid rate limiting
                    time.sleep(random.uniform(1.0, 2.0))
                except Exception as e:
                    print(f"Error downloading {piece_name}: {e}")

    print("Download complete!")

if __name__ == "__main__":
    download_chess_pieces()
