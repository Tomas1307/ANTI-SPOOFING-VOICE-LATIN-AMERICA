"""
Checkpoint Manager Utility
==========================

Reusable checkpoint system for resumable data processing pipelines.

Used by:
- dataset_partition_creation.py (speaker partitioning)
- augmentation_pipeline.py (data augmentation)
- checkpoint_tool.py (CLI management)
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict


class CheckpointManager:
    """Generic checkpoint manager for resumable processing."""
    
    def __init__(self, checkpoint_file: str = "checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.data = {
            'processed_items': [],
            'total_items': 0,
            'stats': {},
            'metadata': {}
        }
    
    def exists(self) -> bool:
        return self.checkpoint_file.exists()
    
    def load(self) -> Dict:
        if self.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.data = json.load(f)
        return self.data
    
    def save(self, data: Dict):
        self.data = data
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_processed_items(self) -> Set[str]:
        return set(self.data.get('processed_items', []))
    
    def get_progress_percentage(self) -> float:
        total = self.data.get('total_items', 0)
        processed = len(self.data.get('processed_items', []))
        return (processed / total * 100.0) if total > 0 else 0.0
    
    def clear(self):
        if self.exists():
            self.checkpoint_file.unlink()
    
    def print_status(self):
        if not self.exists():
            print(f"No checkpoint: {self.checkpoint_file}")
            return
        
        print(f"\nCheckpoint: {self.checkpoint_file}")
        print("=" * 50)
        
        processed = len(self.data.get('processed_items', []))
        total = self.data.get('total_items', 0)
        
        print(f"Processed: {processed}")
        print(f"Total:     {total}")
        
        if total > 0:
            print(f"Progress:  {self.get_progress_percentage():.1f}%")
        
        print("=" * 50)


class PartitionCheckpointManager(CheckpointManager):
    """Checkpoint manager for dataset partitioning."""
    
    def __init__(self, checkpoint_file: str = "partition_checkpoint.json"):
        super().__init__(checkpoint_file)
        self.data = {
            'processed_speakers': [],
            'speaker_splits': {},
            'stats': {
                'train': {'speakers': 0, 'bonafide': 0, 'spoof': {}},
                'val': {'speakers': 0, 'bonafide': 0, 'spoof': {}},
                'test': {'speakers': 0, 'bonafide': 0, 'spoof': {}}
            }
        }
    
    def save_partition(
        self,
        processed_speakers: List[str],
        speaker_splits: Dict,
        stats: Dict
    ):
        """Save partition state."""
        # Convert defaultdicts to regular dicts
        stats_serializable = {}
        for split, split_stats in stats.items():
            stats_serializable[split] = {
                'speakers': split_stats.get('speakers', 0),
                'bonafide': split_stats.get('bonafide', 0),
                'spoof': dict(split_stats.get('spoof', {}))
            }
        
        self.save({
            'processed_speakers': processed_speakers,
            'speaker_splits': speaker_splits,
            'stats': stats_serializable
        })
    
    def get_processed_speakers(self) -> Set[str]:
        return set(self.data.get('processed_speakers', []))
    
    def load_partition(self) -> Optional[Dict]:
        """Load partition state."""
        if not self.exists():
            return None
        
        loaded = self.load()
        
        # Convert back to defaultdict
        for split in ['train', 'val', 'test']:
            if split in loaded.get('stats', {}):
                loaded['stats'][split]['spoof'] = defaultdict(
                    int,
                    loaded['stats'][split].get('spoof', {})
                )
        
        return loaded


class AugmentationCheckpointManager(CheckpointManager):
    """Checkpoint manager for augmentation."""
    
    def __init__(self, checkpoint_file: str = "augmentation_checkpoint.json"):
        super().__init__(checkpoint_file)
        self.data = {
            'processed_files': [],
            'total_files': 0,
            'stats': {
                'clean_copied': 0,
                'augmented_created': 0
            },
            'metadata': {
                'factor': None,
                'output_dir': None
            }
        }
    
    def should_resume(self, factor: str, output_dir: str) -> bool:
        """Check if checkpoint matches current config."""
        if not self.exists():
            return False
        
        data = self.load()
        meta = data.get('metadata', {})
        
        return (
            meta.get('factor') == factor and
            meta.get('output_dir') == output_dir
        )