#!/usr/bin/env python3
"""
Collect Training Data for Chess Piece Recognition

This script helps you collect training data for chess piece recognition.
It extracts squares from chess board images and lets you label them.

Usage:
    python collect_training_data.py --image_path PATH_TO_IMAGE --output_dir PATH_TO_DATASET
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Define piece classes
PIECE_CLASSES = [
    'empty',  # 0
    'P', 'N', 'B', 'R', 'Q', 'K',  # White pieces (1-6)
    'p', 'n', 'b', 'r', 'q', 'k'   # Black pieces (7-12)
]

class ChessBoardProcessor:
    """Class to process chess board images and extract squares."""
    
    def __init__(self):
        """Initialize the processor."""
        pass
    
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
                squares.append((square, (row, col)))
        
        return squares
    
    def process_image(self, img_path):
        """Process an image and extract chess squares."""
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
        
        return img, board_img, squares

class SquareLabelingApp:
    """GUI application for labeling chess squares."""
    
    def __init__(self, root, squares, output_dir):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Chess Square Labeling Tool")
        self.root.geometry("1200x800")
        
        self.squares = squares
        self.output_dir = output_dir
        self.current_index = 0
        
        # Set up the main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Set up the image frame
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Set up the control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Set up the canvas for the square image
        self.canvas = tk.Canvas(self.image_frame, width=400, height=400, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Set up the piece selection frame
        self.piece_frame = ttk.LabelFrame(self.control_frame, text="Select Piece")
        self.piece_frame.pack(fill=tk.X, pady=10)
        
        # Set up the piece selection radio buttons
        self.selected_piece = tk.StringVar(value="empty")
        
        for piece_class in PIECE_CLASSES:
            ttk.Radiobutton(
                self.piece_frame, text=piece_class, value=piece_class, 
                variable=self.selected_piece
            ).pack(anchor=tk.W, padx=5, pady=2)
        
        # Set up the buttons
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            self.button_frame, text="Save and Next", 
            command=self.save_and_next
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            self.button_frame, text="Skip", 
            command=self.next_square
        ).pack(fill=tk.X, pady=2)
        
        # Set up the progress label
        self.progress_label = ttk.Label(
            self.control_frame, 
            text=f"Square 1 of {len(self.squares)}"
        )
        self.progress_label.pack(pady=10)
        
        # Set up the coordinate label
        self.coord_label = ttk.Label(
            self.control_frame, 
            text="Coordinates: "
        )
        self.coord_label.pack(pady=5)
        
        # Display the first square
        self.display_current_square()
    
    def display_current_square(self):
        """Display the current square."""
        if self.current_index < len(self.squares):
            square, (row, col) = self.squares[self.current_index]
            
            # Convert to PIL Image
            img = Image.fromarray(square)
            
            # Resize to fit the canvas
            img = img.resize((400, 400), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(img)
            
            # Display on canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Update progress label
            self.progress_label.config(
                text=f"Square {self.current_index + 1} of {len(self.squares)}"
            )
            
            # Update coordinate label
            square_name = f"{chr(97 + col)}{8 - row}"
            self.coord_label.config(
                text=f"Coordinates: {square_name} (row {row}, col {col})"
            )
    
    def save_and_next(self):
        """Save the current square and move to the next one."""
        if self.current_index < len(self.squares):
            square, (row, col) = self.squares[self.current_index]
            piece_class = self.selected_piece.get()
            
            # Create the output directory if it doesn't exist
            piece_dir = os.path.join(self.output_dir, piece_class)
            os.makedirs(piece_dir, exist_ok=True)
            
            # Save the square image
            square_name = f"{chr(97 + col)}{8 - row}"
            filename = f"square_{self.current_index}_{square_name}.png"
            filepath = os.path.join(piece_dir, filename)
            
            # Convert to BGR for OpenCV
            square_bgr = cv2.cvtColor(square, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, square_bgr)
            
            print(f"Saved {filepath}")
            
            # Move to the next square
            self.next_square()
    
    def next_square(self):
        """Move to the next square."""
        self.current_index += 1
        
        if self.current_index < len(self.squares):
            self.display_current_square()
        else:
            # All squares processed
            self.canvas.delete("all")
            self.canvas.create_text(
                200, 200, 
                text="All squares processed!", 
                font=("Arial", 20)
            )
            self.progress_label.config(text="Complete!")
            self.coord_label.config(text="")

def main():
    """Main function to parse arguments and run the application."""
    parser = argparse.ArgumentParser(description="Collect Training Data for Chess Piece Recognition")
    parser.add_argument("--image_path", help="Path to the chess board image")
    parser.add_argument("--output_dir", default="chess_dataset", help="Directory to save labeled squares")
    
    args = parser.parse_args()
    
    if not args.image_path:
        print("Error: Please provide --image_path")
        parser.print_help()
        return 1
    
    if not os.path.isfile(args.image_path):
        print(f"Error: {args.image_path} is not a file")
        return 1
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the image
    processor = ChessBoardProcessor()
    try:
        _, _, squares = processor.process_image(args.image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1
    
    # Create the root window
    root = tk.Tk()
    
    # Create the application
    app = SquareLabelingApp(root, squares, args.output_dir)
    
    # Start the main loop
    root.mainloop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
