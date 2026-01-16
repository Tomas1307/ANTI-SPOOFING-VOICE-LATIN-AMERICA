"""
Checkpoint Manager for Augmentation Pipeline

Allows resuming interrupted augmentation by tracking progress.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

class CheckpointManager:
    """
    Manages checkpoint system for resumable partitioning.
    
    Saves progress periodically to allow resumption after interruption.
    """
    
    def __init__(self, checkpoint_file: str = "partition_checkpoint.json"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.data = {
            'processed_speakers': [],
            'stats': {
                'bonafide_files': 0,
                'spoof_files': {},
                'skipped_speakers': []
            }
        }
    
    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_file.exists()
    
    def load(self) -> Dict:
        """Load checkpoint data from file."""
        if self.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.data = json.load(f)
            return self.data
        return self.data
    
    def save(self, processed_speakers: List[str], stats: Dict):
        """
        Save current progress to checkpoint.
        
        Args:
            processed_speakers: List of successfully processed speaker IDs
            stats: Current statistics dictionary
        """
        self.data['processed_speakers'] = processed_speakers
        self.data['stats'] = {
            'bonafide_files': stats['bonafide_files'],
            'spoof_files': dict(stats['spoof_files']),
            'skipped_speakers': stats['skipped_speakers']
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_processed_speakers(self) -> Set[str]:
        """Get set of already processed speaker IDs."""
        return set(self.data['processed_speakers'])
    
    def clear(self):
        """Remove checkpoint file."""
        if self.exists():
            self.checkpoint_file.unlink()