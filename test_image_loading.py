"""
Test script for loading chess piece images.
"""
import os
import tkinter as tk
from PIL import Image, ImageTk

def main():
    """Test image loading."""
    print("Testing image loading...")
    
    # Create a simple Tkinter window
    root = tk.Tk()
    root.title("Image Loading Test")
    
    # Check for piece images in the assets directory
    assets_dir = os.path.join("chess_variant", "assets")
    print(f"Looking for images in: {os.path.abspath(assets_dir)}")
    
    # List all files in the assets directory
    if os.path.exists(assets_dir):
        files = os.listdir(assets_dir)
        print(f"Files found: {files}")
    else:
        print(f"Assets directory not found: {assets_dir}")
        return
    
    # Try to load one image
    test_file = "white_king.png"
    file_path = os.path.join(assets_dir, test_file)
    
    if os.path.exists(file_path):
        print(f"Test file exists: {file_path}")
        try:
            img = Image.open(file_path)
            print(f"Image opened: {img.format}, {img.size}, {img.mode}")
            
            # Try to resize the image
            try:
                img = img.resize((64, 64))
                print("Image resized successfully")
            except Exception as e:
                print(f"Error resizing image: {e}")
            
            # Try to convert to PhotoImage
            try:
                photo = ImageTk.PhotoImage(img)
                print("Converted to PhotoImage successfully")
                
                # Display the image
                label = tk.Label(root, image=photo)
                label.image = photo  # Keep a reference
                label.pack(padx=10, pady=10)
                
                print("Image displayed successfully")
            except Exception as e:
                print(f"Error converting to PhotoImage: {e}")
        except Exception as e:
            print(f"Error opening image: {e}")
    else:
        print(f"Test file not found: {file_path}")
    
    # Run the Tkinter main loop for a short time
    root.after(3000, root.destroy)  # Close after 3 seconds
    root.mainloop()
    
    print("Test completed")

if __name__ == "__main__":
    main()
