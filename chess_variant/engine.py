"""
Stockfish engine integration for the custom chess variant.
"""
import chess
import chess.engine
import os
import platform
from typing import Optional, Tuple, Dict, Any

class StockfishEngine:
    """
    Wrapper for the Stockfish chess engine with custom evaluation for the variant rules.
    """
    def __init__(self, depth: int = 20, nodes: int = None, time_limit: float = None):
        """
        Initialize the Stockfish engine.
        
        Args:
            depth: Search depth
            nodes: Node limit (optional)
            time_limit: Time limit in seconds (optional)
        """
        self.depth = depth
        self.nodes = nodes
        self.time_limit = time_limit
        self.engine = None
        self.engine_path = self._find_engine()
        
    def _find_engine(self) -> str:
        """Find the Stockfish executable based on the operating system."""
        # Default paths to check
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
                return path
        
        # If not found, assume it's in PATH
        return "stockfish"
    
    def start(self):
        """Start the Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            # Set engine options for optimal performance
            self.engine.configure({"Threads": 4, "Hash": 128})
        except Exception as e:
            raise Exception(f"Failed to start Stockfish engine: {e}")
    
    def stop(self):
        """Stop the Stockfish engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def get_best_move(self, board: chess.Board) -> Tuple[chess.Move, Dict[str, Any]]:
        """
        Get the best move from Stockfish for the current position.
        
        Args:
            board: Current chess position
            
        Returns:
            Tuple of (best move, info dictionary)
        """
        if not self.engine:
            self.start()
        
        # Create limit based on provided parameters
        limit_kwargs = {}
        if self.depth:
            limit_kwargs["depth"] = self.depth
        if self.nodes:
            limit_kwargs["nodes"] = self.nodes
        if self.time_limit:
            limit_kwargs["time"] = self.time_limit
        
        limit = chess.engine.Limit(**limit_kwargs)
        
        # Get the best move
        result = self.engine.play(board, limit)
        
        # Get additional info about the position
        info = self.engine.analyse(board, limit)
        
        return result.move, info
    
    def evaluate_position(self, board: chess.Board) -> int:
        """
        Evaluate the current position.
        
        Args:
            board: Current chess position
            
        Returns:
            Score in centipawns from white's perspective
        """
        if not self.engine:
            self.start()
        
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        
        # Extract score
        score = info["score"].white().score(mate_score=10000)
        return score
    
    def get_top_moves(self, board: chess.Board, num_moves: int = 3) -> list:
        """
        Get the top N moves from Stockfish for the current position.
        
        Args:
            board: Current chess position
            num_moves: Number of top moves to return
            
        Returns:
            List of (move, score, info) tuples
        """
        if not self.engine:
            self.start()
        
        limit = chess.engine.Limit(depth=self.depth)
        
        # Get multipv analysis
        analysis = self.engine.analyse(
            board, 
            limit=limit,
            multipv=num_moves
        )
        
        results = []
        for pv_info in analysis:
            move = pv_info["pv"][0] if "pv" in pv_info and pv_info["pv"] else None
            score = pv_info["score"].white().score(mate_score=10000) if "score" in pv_info else None
            results.append((move, score, pv_info))
        
        return results
    
    def __del__(self):
        """Ensure engine is properly closed when object is deleted."""
        self.stop()
