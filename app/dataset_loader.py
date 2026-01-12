"""
Dataset Loader

Handles loading of original voice files, MUSAN noise files, and RIR files
for the augmentation pipeline. Provides organized access to all required
audio resources.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

import app.utils as utils


class DatasetLoader:
    """
    Loader for voice dataset and augmentation resources.
    
    Provides methods to load:
    - Original voice files from partition_dataset_by_speaker
    - MUSAN noise files (noise, speech, music)
    - RIR files (small, medium, large rooms)
    
    Attributes:
        voices_root: Path to partition_dataset_by_speaker directory.
        musan_root: Path to MUSAN dataset directory.
        rir_root: Path to RIR dataset directory.
    """
    
    def __init__(
        self,
        voices_root: str = "data/partition_dataset_by_speaker",
        musan_root: str = "data/noise_dataset/musan",
        rir_root: str = "data/RIR"
    ):
        """
        Initialize dataset loader.
        
        Args:
            voices_root: Root directory for original voice files.
            musan_root: Root directory for MUSAN noise dataset.
            rir_root: Root directory for RIR files.
        """
        self.voices_root = Path(voices_root)
        self.musan_root = Path(musan_root)
        self.rir_root = Path(rir_root)
        
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that all required paths exist."""
        if not self.voices_root.exists():
            raise FileNotFoundError(f"Voices root not found: {self.voices_root}")
        
        if not self.musan_root.exists():
            print(f"Warning: MUSAN root not found: {self.musan_root}")
        
        if not self.rir_root.exists():
            print(f"Warning: RIR root not found: {self.rir_root}")
    
    def load_train_files(self) -> List[Dict[str, str]]:
        """
        Load all training audio files from partition_dataset_by_speaker.
        
        Returns:
            List of dictionaries containing file metadata:
            - filepath: Full path to audio file
            - speaker_id: Speaker identifier
            - split: 'train'
            - filename: Original filename
        """
        train_files = []
        
        speakers = [d for d in self.voices_root.iterdir() if d.is_dir()]
        
        print(f"\nLoading train files from {len(speakers)} speakers...")
        
        for speaker_dir in speakers:
            speaker_id = speaker_dir.name
            train_dir = speaker_dir / "train"
            
            if not train_dir.exists():
                print(f"Warning: No train directory for speaker {speaker_id}")
                continue
            
            audio_files = utils.get_audio_files_recursive(str(train_dir))
            
            for audio_file in audio_files:
                train_files.append({
                    "filepath": audio_file,
                    "speaker_id": speaker_id,
                    "split": "train",
                    "filename": Path(audio_file).name
                })
        
        print(f"Loaded {len(train_files)} training files")
        
        return train_files
    
    def load_val_files(self) -> List[Dict[str, str]]:
        """
        Load all validation audio files.
        
        Returns:
            List of dictionaries containing file metadata.
        """
        val_files = []
        
        speakers = [d for d in self.voices_root.iterdir() if d.is_dir()]
        
        print(f"\nLoading validation files from {len(speakers)} speakers...")
        
        for speaker_dir in speakers:
            speaker_id = speaker_dir.name
            val_dir = speaker_dir / "val"
            
            if not val_dir.exists():
                continue
            
            audio_files = utils.get_audio_files_recursive(str(val_dir))
            
            for audio_file in audio_files:
                val_files.append({
                    "filepath": audio_file,
                    "speaker_id": speaker_id,
                    "split": "val",
                    "filename": Path(audio_file).name
                })
        
        print(f"Loaded {len(val_files)} validation files")
        
        return val_files
    
    def load_test_files(self) -> List[Dict[str, str]]:
        """
        Load all test audio files.
        
        Returns:
            List of dictionaries containing file metadata.
        """
        test_files = []
        
        speakers = [d for d in self.voices_root.iterdir() if d.is_dir()]
        
        print(f"\nLoading test files from {len(speakers)} speakers...")
        
        for speaker_dir in speakers:
            speaker_id = speaker_dir.name
            test_dir = speaker_dir / "test"
            
            if not test_dir.exists():
                continue
            
            audio_files = utils.get_audio_files_recursive(str(test_dir))
            
            for audio_file in audio_files:
                test_files.append({
                    "filepath": audio_file,
                    "speaker_id": speaker_id,
                    "split": "test",
                    "filename": Path(audio_file).name
                })
        
        print(f"Loaded {len(test_files)} test files")
        
        return test_files
    
    def get_dataset_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with counts for each split.
        """
        train_files = self.load_train_files()
        val_files = self.load_val_files()
        test_files = self.load_test_files()
        
        return {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
            "total": len(train_files) + len(val_files) + len(test_files)
        }
    
    def get_speaker_distribution(self) -> Dict[str, int]:
        """
        Get distribution of files per speaker in training set.
        
        Returns:
            Dictionary mapping speaker_id to file count.
        """
        train_files = self.load_train_files()
        
        speaker_counts = {}
        for file_info in train_files:
            speaker_id = file_info["speaker_id"]
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        
        return speaker_counts
    
    def print_summary(self):
        """Print comprehensive dataset summary."""
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        
        stats = self.get_dataset_statistics()
        print(f"\nFile counts:")
        print(f"  Train: {stats['train']:,}")
        print(f"  Val:   {stats['val']:,}")
        print(f"  Test:  {stats['test']:,}")
        print(f"  Total: {stats['total']:,}")
        
        speaker_dist = self.get_speaker_distribution()
        print(f"\nSpeaker statistics:")
        print(f"  Total speakers: {len(speaker_dist)}")
        print(f"  Avg files per speaker: {sum(speaker_dist.values()) / len(speaker_dist):.1f}")
        print(f"  Min files: {min(speaker_dist.values())}")
        print(f"  Max files: {max(speaker_dist.values())}")
        
        country_counts = {}
        for speaker_id in speaker_dist.keys():
            country_code = speaker_id[:2]
            country_map = {
                "ar": "Argentina",
                "cl": "Chile",
                "co": "Colombia",
                "pe": "Peru",
                "ve": "Venezuela"
            }
            country = country_map.get(country_code, "Unknown")
            country_counts[country] = country_counts.get(country, 0) + 1
        
        print(f"\nSpeakers by country:")
        for country, count in sorted(country_counts.items()):
            print(f"  {country}: {count}")
        
        print("="*70 + "\n")


def test_dataset_loader():
    """Test dataset loader functionality."""
    loader = DatasetLoader(
        voices_root="data/partition_dataset_by_speaker",
        musan_root="data/noise_dataset/musan",
        rir_root="data/RIR"
    )
    
    loader.print_summary()
    
    train_files = loader.load_train_files()
    print(f"\nSample train file:")
    if train_files:
        sample = train_files[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_dataset_loader()
