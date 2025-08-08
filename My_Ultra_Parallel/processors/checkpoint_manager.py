"""
Checkpoint manager for saving and loading processing state.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manages checkpoint save and load operations."""
    
    def __init__(self, checkpoint_file: str = "modular_checkpoint.pkl"):
        self.checkpoint_file = checkpoint_file
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file."""
        try:
            if Path(self.checkpoint_file).exists():
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                logging.info(f"📂 Loaded checkpoint from: {self.checkpoint_file}")
                return checkpoint_data
            else:
                logging.info(f"📂 No checkpoint file found: {self.checkpoint_file}")
                return None
        except Exception as e:
            logging.error(f"❌ Failed to load checkpoint: {e}")
            return None
    
    def save_checkpoint(self, data: Dict[str, Any]) -> bool:
        """Save checkpoint data to file."""
        try:
            # Ensure directory exists
            Path(self.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
            
            logging.info(f"💾 Checkpoint saved to: {self.checkpoint_file}")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to save checkpoint: {e}")
            return False 