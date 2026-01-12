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
    Manages checkpoints for augmentation pipeline to enable resuming.
    
    Checkpoint file format:
    {
        "factor": "3x",
        "total_files": 18204,
        "processed_files": 5234,
        "processed_speakers": ["arf_00295", "clm_12345", ...],
        "last_audio_id": 25170,
        "timestamp": "2026-01-12T15:30:00",
        "output_dir": "data/augmented/3x"
    }
    """
    
    def __init__(self, checkpoint_file: str = "augmentation_checkpoint.json"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_file: Path to checkpoint file.
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_data = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict:
        """Load existing checkpoint or create empty one."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            "factor": None,
            "total_files": 0,
            "processed_files": 0,
            "processed_speakers": [],
            "last_audio_id": 0,
            "timestamp": None,
            "output_dir": None
        }
    
    def save_checkpoint(
        self,
        factor: str,
        total_files: int,
        processed_files: int,
        processed_speakers: List[str],
        last_audio_id: int,
        output_dir: str
    ):
        """
        Save current progress to checkpoint.
        
        Args:
            factor: Augmentation factor (e.g., "3x").
            total_files: Total number of files to process.
            processed_files: Number of files processed so far.
            processed_speakers: List of speaker IDs already processed.
            last_audio_id: Last audio ID counter value.
            output_dir: Output directory path.
        """
        self.checkpoint_data = {
            "factor": factor,
            "total_files": total_files,
            "processed_files": processed_files,
            "processed_speakers": processed_speakers,
            "last_audio_id": last_audio_id,
            "timestamp": datetime.now().isoformat(),
            "output_dir": output_dir
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint_data, f, indent=2)
        
        print(f"\n✓ Checkpoint saved: {processed_files}/{total_files} files processed")
    
    def should_resume(self, factor: str, output_dir: str) -> bool:
        """
        Check if there's a valid checkpoint to resume from.
        
        Args:
            factor: Current augmentation factor.
            output_dir: Current output directory.
            
        Returns:
            True if checkpoint exists and matches current run.
        """
        if not self.checkpoint_file.exists():
            return False
        
        if (self.checkpoint_data["factor"] == factor and 
            self.checkpoint_data["output_dir"] == output_dir and
            self.checkpoint_data["processed_files"] > 0):
            return True
        
        return False
    
    def get_processed_speakers(self) -> Set[str]:
        """
        Get set of already processed speaker IDs.
        
        Returns:
            Set of speaker IDs that were already processed.
        """
        return set(self.checkpoint_data.get("processed_speakers", []))
    
    def get_last_audio_id(self) -> int:
        """
        Get last used audio ID counter.
        
        Returns:
            Last audio ID counter value.
        """
        return self.checkpoint_data.get("last_audio_id", 0)
    
    def print_status(self):
        """Print current checkpoint status."""
        if not self.checkpoint_file.exists():
            print("No checkpoint found. Starting fresh.")
            return
        
        print("\n" + "="*70)
        print("CHECKPOINT STATUS")
        print("="*70)
        print(f"Factor: {self.checkpoint_data['factor']}")
        print(f"Progress: {self.checkpoint_data['processed_files']}/{self.checkpoint_data['total_files']}")
        print(f"Speakers processed: {len(self.checkpoint_data['processed_speakers'])}")
        print(f"Last audio ID: {self.checkpoint_data['last_audio_id']}")
        print(f"Last saved: {self.checkpoint_data['timestamp']}")
        print(f"Output: {self.checkpoint_data['output_dir']}")
        print("="*70 + "\n")
    
    def clear_checkpoint(self):
        """Delete checkpoint file."""
        if self.checkpoint_file.exists():
            os.remove(self.checkpoint_file)
            print(f"✓ Checkpoint cleared: {self.checkpoint_file}")
    
    def get_progress_percentage(self) -> float:
        """
        Calculate progress percentage.
        
        Returns:
            Progress as percentage (0-100).
        """
        total = self.checkpoint_data.get("total_files", 0)
        processed = self.checkpoint_data.get("processed_files", 0)
        
        if total == 0:
            return 0.0
        
        return (processed / total) * 100


def main():
    """Demo of checkpoint manager."""
    manager = CheckpointManager()
    
    # Simulate saving checkpoint
    manager.save_checkpoint(
        factor="3x",
        total_files=18204,
        processed_files=5000,
        processed_speakers=["arf_00295", "clm_12345"],
        last_audio_id=25000,
        output_dir="data/augmented/3x"
    )
    
    # Check status
    manager.print_status()
    
    # Check if should resume
    if manager.should_resume("3x", "data/augmented/3x"):
        print(f"✓ Can resume from checkpoint ({manager.get_progress_percentage():.1f}% complete)")
        print(f"  Processed speakers: {len(manager.get_processed_speakers())}")
        print(f"  Last audio ID: {manager.get_last_audio_id()}")


if __name__ == "__main__":
    main()