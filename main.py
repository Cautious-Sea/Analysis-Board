"""
Main entry point for the custom chess variant application.
"""
import tkinter as tk
import os
import sys
import platform
import subprocess
import threading
import time
from tkinter import messagebox

from chess_variant.gui import ChessVariantGUI

def check_stockfish():
    """
    Check if Stockfish is installed and accessible.
    
    Returns:
        bool: True if Stockfish is found, False otherwise
    """
    try:
        # Try to find Stockfish in common locations
        stockfish_path = None
        
        if platform.system() == "Windows":
            paths = [
                "stockfish/stockfish-windows-x86-64.exe",
                "stockfish/stockfish-windows-2022-x86-64-avx2.exe",
                "stockfish/stockfish.exe",
                "C:/Program Files/Stockfish/stockfish.exe",
                "C:/Program Files (x86)/Stockfish/stockfish.exe",
            ]
        elif platform.system() == "Darwin":  # macOS
            paths = [
                "stockfish/stockfish-macos-x86-64",
                "stockfish/stockfish-macos-arm64",
                "stockfish/stockfish",
                "/usr/local/bin/stockfish",
            ]
        else:  # Linux and others
            paths = [
                "stockfish/stockfish-ubuntu-x86-64",
                "stockfish/stockfish",
                "/usr/local/bin/stockfish",
                "/usr/bin/stockfish",
            ]
        
        # Check if any of the paths exist
        for path in paths:
            if os.path.isfile(path):
                stockfish_path = path
                break
        
        # If not found in specific paths, check if it's in PATH
        if not stockfish_path:
            try:
                # Try to run stockfish with a simple command
                if platform.system() == "Windows":
                    subprocess.run(["where", "stockfish"], check=True, capture_output=True)
                else:
                    subprocess.run(["which", "stockfish"], check=True, capture_output=True)
                stockfish_path = "stockfish"  # It's in PATH
            except subprocess.CalledProcessError:
                # Not in PATH
                return False
        
        # Try to run Stockfish with a simple command to verify it works
        process = subprocess.Popen(
            stockfish_path, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send "quit" command
        process.stdin.write("quit\n")
        process.stdin.flush()
        
        # Wait for process to terminate
        process.wait(timeout=2)
        
        return process.returncode == 0
    except Exception as e:
        print(f"Error checking Stockfish: {e}")
        return False

def download_stockfish():
    """
    Download Stockfish for the current platform.
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Create stockfish directory if it doesn't exist
    os.makedirs("stockfish", exist_ok=True)
    
    # Determine URL based on platform
    url = None
    output_file = None
    
    if platform.system() == "Windows":
        if platform.machine().endswith('64'):
            url = "https://stockfishchess.org/files/stockfish-16-win.zip"
            output_file = "stockfish/stockfish-16-win.zip"
    elif platform.system() == "Darwin":  # macOS
        if platform.machine() == 'arm64':
            url = "https://stockfishchess.org/files/stockfish-16-mac-arm64.zip"
            output_file = "stockfish/stockfish-16-mac-arm64.zip"
        else:
            url = "https://stockfishchess.org/files/stockfish-16-mac-x86-64.zip"
            output_file = "stockfish/stockfish-16-mac-x86-64.zip"
    else:  # Linux
        if platform.machine().endswith('64'):
            url = "https://stockfishchess.org/files/stockfish-16-linux.zip"
            output_file = "stockfish/stockfish-16-linux.zip"
    
    if not url or not output_file:
        messagebox.showerror(
            "Unsupported Platform",
            "Automatic Stockfish download is not supported for your platform. "
            "Please download Stockfish 16 manually from https://stockfishchess.org/download/ "
            "and place it in a location accessible by this application."
        )
        return False
    
    try:
        import urllib.request
        import zipfile
        import shutil
        
        # Download the file
        messagebox.showinfo(
            "Downloading Stockfish",
            "Downloading Stockfish 16. This may take a moment..."
        )
        
        # Create a progress window
        progress_window = tk.Toplevel()
        progress_window.title("Downloading Stockfish")
        progress_window.geometry("300x100")
        progress_window.resizable(False, False)
        
        progress_label = tk.Label(progress_window, text="Downloading Stockfish 16...")
        progress_label.pack(pady=10)
        
        progress_bar = tk.Canvas(progress_window, width=250, height=20, bg="white")
        progress_bar.pack(pady=10)
        
        # Function to update progress bar
        def update_progress(current, total):
            progress_bar.delete("progress")
            width = int(250 * (current / total))
            progress_bar.create_rectangle(0, 0, width, 20, fill="blue", tags="progress")
            progress_window.update()
        
        # Download with progress
        def download_with_progress():
            try:
                urllib.request.urlretrieve(
                    url, 
                    output_file, 
                    lambda count, block_size, total_size: update_progress(
                        min(count * block_size, total_size), 
                        total_size
                    )
                )
                
                # Extract the zip file
                with zipfile.ZipFile(output_file, 'r') as zip_ref:
                    zip_ref.extractall("stockfish")
                
                # Find the stockfish executable in the extracted files
                stockfish_exe = None
                for root, dirs, files in os.walk("stockfish"):
                    for file in files:
                        if file.startswith("stockfish") and (
                            file.endswith(".exe") or 
                            "." not in file  # Unix executable without extension
                        ):
                            stockfish_exe = os.path.join(root, file)
                            # Make it executable on Unix
                            if platform.system() != "Windows":
                                os.chmod(stockfish_exe, 0o755)
                            break
                    if stockfish_exe:
                        break
                
                # Clean up
                os.remove(output_file)
                
                progress_window.destroy()
                
                if stockfish_exe:
                    messagebox.showinfo(
                        "Download Complete",
                        f"Stockfish 16 has been downloaded and extracted successfully."
                    )
                    return True
                else:
                    messagebox.showerror(
                        "Extraction Error",
                        "Could not find Stockfish executable in the downloaded package."
                    )
                    return False
            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Download Error", f"Error downloading Stockfish: {e}")
                return False
        
        # Run download in a separate thread
        download_thread = threading.Thread(target=download_with_progress)
        download_thread.start()
        
        # Wait for download to complete
        while download_thread.is_alive():
            progress_window.update()
            time.sleep(0.1)
        
        return True
    except Exception as e:
        messagebox.showerror("Download Error", f"Error downloading Stockfish: {e}")
        return False

def main():
    """Main entry point for the application."""
    # Check if Stockfish is installed
    if not check_stockfish():
        # Offer to download Stockfish
        response = messagebox.askyesno(
            "Stockfish Not Found",
            "Stockfish chess engine was not found on your system. "
            "Would you like to download Stockfish 16 now?"
        )
        
        if response:
            success = download_stockfish()
            if not success:
                messagebox.showwarning(
                    "Stockfish Required",
                    "This application requires Stockfish 16 to function properly. "
                    "Please download and install Stockfish 16 from https://stockfishchess.org/download/ "
                    "and try again."
                )
                return
        else:
            messagebox.showwarning(
                "Stockfish Required",
                "This application requires Stockfish 16 to function properly. "
                "Please download and install Stockfish 16 from https://stockfishchess.org/download/ "
                "and try again."
            )
            return
    
    # Create the main window
    root = tk.Tk()
    app = ChessVariantGUI(root)
    app.run()

if __name__ == "__main__":
    main()
