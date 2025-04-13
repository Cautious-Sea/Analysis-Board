#!/usr/bin/env python3
"""
Chess FEN Detector

This script detects a chess board in an image and outputs the FEN notation.
It uses a simple approach to detect pieces based on color and shape.

Usage:
    python chess_fen_detector.py <image_path> [--visualize]
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

class ChessFENDetector:
    """Class to detect a chess board in an image and output FEN notation."""

    def __init__(self, model_path=None):
        """Initialize the detector."""
        # Define piece detection thresholds (used as fallback if no model is provided)
        self.EMPTY_THRESHOLD = 0.05  # Threshold for empty square detection
        self.WHITE_THRESHOLD = 0.25  # Threshold for white piece detection
        self.BLACK_THRESHOLD = 0.15  # Threshold for black piece detection

        # Load the model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Will use simple detection instead")

    def preprocess_image(self, img_path):
        """Load and preprocess the image."""
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")

        # Convert to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize if the image is too large
        max_dim = 1200
        h, w = img_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

        return img_rgb

    def detect_board(self, img):
        """Detect the chess board in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the largest contour (likely the chess board)
        if not contours:
            raise ValueError("No contours found in the image")

        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # If the polygon has 4 vertices, it's likely the chess board
        if len(approx) == 4:
            # Sort the points in order: top-left, top-right, bottom-right, bottom-left
            pts = np.array([p[0] for p in approx])
            rect = self.order_points(pts)

            # Apply perspective transform to get a top-down view
            warped = self.four_point_transform(img, rect)

            return warped
        else:
            # If we can't find a quadrilateral, try to use the bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Check if the bounding rectangle is approximately square
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                # Extract the region
                board = img[y:y+h, x:x+w]
                return board
            else:
                # Try to find the board using Hough lines
                return self.detect_board_hough(img)

    def detect_board_hough(self, img):
        """Detect the chess board using Hough lines."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Apply Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        if lines is None:
            raise ValueError("No lines detected in the image")

        # Filter horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            rho, theta = line[0]
            if 0.1 < theta < 0.5:  # Vertical lines
                v_lines.append((rho, theta))
            elif 1.0 < theta < 1.5:  # Horizontal lines
                h_lines.append((rho, theta))

        # Sort lines by rho
        h_lines.sort()
        v_lines.sort()

        # Find the most likely board boundaries
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            # Get the outermost lines
            top_line = h_lines[0]
            bottom_line = h_lines[-1]
            left_line = v_lines[0]
            right_line = v_lines[-1]

            # Calculate intersection points
            tl = self.intersection(top_line, left_line)
            tr = self.intersection(top_line, right_line)
            br = self.intersection(bottom_line, right_line)
            bl = self.intersection(bottom_line, left_line)

            # Apply perspective transform
            pts = np.array([tl, tr, br, bl], dtype=np.float32)
            warped = self.four_point_transform(img, pts)

            return warped

        # If all else fails, return the original image
        return img

    def order_points(self, pts):
        """Order points in: top-left, top-right, bottom-right, bottom-left."""
        # Initialize a list of coordinates
        rect = np.zeros((4, 2), dtype=np.float32)

        # The top-left point has the smallest sum
        # The bottom-right point has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # The top-right point has the smallest difference
        # The bottom-left point has the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image, pts):
        """Apply perspective transform to get a top-down view."""
        # Obtain a consistent order of the points
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Construct the set of destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def intersection(self, line1, line2):
        """Calculate the intersection point of two lines in Hesse normal form."""
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])

        try:
            x, y = np.linalg.solve(A, b)
            return [x, y]
        except np.linalg.LinAlgError:
            # Lines are parallel
            return [0, 0]

    def extract_squares(self, board_img):
        """Extract the 64 squares from the chess board."""
        # Resize to a standard size
        board_img = cv2.resize(board_img, (800, 800))

        # Calculate square size
        h, w = board_img.shape[:2]
        square_size = h // 8

        # Extract each square
        squares = []
        for row in range(8):
            for col in range(8):
                y1 = row * square_size
                y2 = (row + 1) * square_size
                x1 = col * square_size
                x2 = (col + 1) * square_size

                square = board_img[y1:y2, x1:x2]
                squares.append(square)

        return squares

    def detect_piece(self, square_img):
        """Detect if a square contains a piece and its type."""
        # If we have a trained model, use it
        if self.model is not None:
            try:
                # Resize the image to match the model's input shape
                input_shape = self.model.input_shape[1:3]  # (height, width)
                resized_img = cv2.resize(square_img, (input_shape[1], input_shape[0]))

                # Normalize the image
                img_array = np.expand_dims(resized_img, axis=0) / 255.0

                # Make prediction
                predictions = self.model.predict(img_array)
                class_idx = np.argmax(predictions[0])

                # Map class index to piece type
                piece_classes = [
                    'empty',  # 0
                    'P', 'N', 'B', 'R', 'Q', 'K',  # White pieces (1-6)
                    'p', 'n', 'b', 'r', 'q', 'k'   # Black pieces (7-12)
                ]

                # Return the predicted piece type
                return piece_classes[class_idx]
            except Exception as e:
                print(f"Error using model for prediction: {e}")
                print("Falling back to simple detection")

        # Fallback to simple detection if no model or model fails
        # Convert to grayscale
        gray = cv2.cvtColor(square_img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # If no contours found, the square is likely empty
        if not contours:
            return "empty"

        # Get the largest contour (likely the piece)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)

        # Calculate the square area
        square_area = square_img.shape[0] * square_img.shape[1]

        # Calculate the ratio of contour area to square area
        area_ratio = contour_area / square_area

        # If the ratio is too small, the square is likely empty
        if area_ratio < self.EMPTY_THRESHOLD:
            return "empty"

        # Create a mask for the contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)

        # Calculate the average color of the piece
        mean_color = cv2.mean(square_img, mask=mask)[:3]  # RGB

        # Calculate brightness (simple average of RGB)
        brightness = sum(mean_color) / 3

        # Determine if it's a white or black piece based on brightness
        if brightness > 128:
            return "P"  # Default to white pawn
        else:
            return "p"  # Default to black pawn

    def generate_fen(self, piece_map):
        """Generate FEN notation from the piece map."""
        fen = ""

        for row in range(8):
            empty_count = 0

            for col in range(8):
                piece = piece_map[row * 8 + col]

                if piece == "empty":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)
                        empty_count = 0
                    fen += piece

            if empty_count > 0:
                fen += str(empty_count)

            if row < 7:
                fen += "/"

        # Add the rest of the FEN (assuming white to move, no castling, etc.)
        fen += " w - - 0 1"

        return fen

    def detect_position(self, img_path, visualize=False):
        """Detect the chess position from an image."""
        # Preprocess the image
        img = self.preprocess_image(img_path)

        # Detect the board
        try:
            board_img = self.detect_board(img)
        except Exception as e:
            print(f"Error detecting board: {e}")
            board_img = img  # Use the original image if detection fails

        # Extract the squares
        squares = self.extract_squares(board_img)

        # Classify each square
        piece_map = []
        for square in squares:
            piece = self.detect_piece(square)
            piece_map.append(piece)

        # Generate FEN notation
        fen = self.generate_fen(piece_map)

        # Visualize if requested
        if visualize:
            self.visualize_detection(img, board_img, squares, piece_map, fen)

        return fen, board_img, squares, piece_map

    def visualize_detection(self, original_img, board_img, squares, piece_map, fen):
        """Visualize the detection process."""
        plt.figure(figsize=(15, 10))

        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis('off')

        # Detected board
        plt.subplot(2, 2, 2)
        plt.imshow(board_img)
        plt.title("Detected Board")
        plt.axis('off')

        # Squares grid with detected pieces
        plt.subplot(2, 2, 3)
        grid = np.zeros((8*50, 8*50, 3), dtype=np.uint8)

        for i, (square, piece) in enumerate(zip(squares, piece_map)):
            row, col = i // 8, i % 8
            y, x = row * 50, col * 50

            # Resize square to fit in the grid
            resized_square = cv2.resize(square, (50, 50))
            grid[y:y+50, x:x+50] = resized_square

            # Add piece label
            if piece != "empty":
                plt.text(x+25, y+25, piece, color='red', fontsize=12,
                         ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7))

        plt.imshow(grid)
        plt.title("Detected Pieces")
        plt.axis('off')

        # FEN notation
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, f"FEN: {fen}",
                 ha='center', va='center', fontsize=12, wrap=True)
        plt.title("FEN Notation")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def main():
    """Main function to parse arguments and run the detector."""
    parser = argparse.ArgumentParser(description="Chess FEN Detector")
    parser.add_argument("image_path", help="Path to the chess board image")
    parser.add_argument("--model", help="Path to the trained model (optional)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the detection process")

    args = parser.parse_args()

    # Create the detector with the model if provided
    detector = ChessFENDetector(model_path=args.model)

    try:
        fen, board_img, squares, piece_map = detector.detect_position(args.image_path, args.visualize)

        # Print the FEN notation
        print(f"Detected position (FEN): {fen}")

        # Save the detected board image
        output_path = "detected_board.png"
        cv2.imwrite(output_path, cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR))
        print(f"Detected board saved to {output_path}")

        # Print instructions for using the FEN notation
        print("\nTo use this position in the chess variant application:")
        print("1. Open the chess variant application")
        print("2. Click on 'Setup Position'")
        print("3. Click on 'Clear Board'")
        print("4. Set up the pieces according to the FEN notation")
        print("5. Click on 'Done Setup'")

        # Print a link to view the position on lichess.org
        fen_for_url = fen.split(' ')[0]  # Get just the piece placement part
        lichess_url = f"https://lichess.org/editor/{fen_for_url}"
        print(f"\nView this position on lichess.org: {lichess_url}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
