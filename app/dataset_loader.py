"""
Dataset Loader

Handles loading of original voice files, MUSAN noise files, and RIR files
for the augmentation pipeline. Provides organized access to all required
audio resources.

Supports bonafide/spoof separation for balanced augmentation.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


class DatasetLoader:
    """
    Loader for voice dataset and augmentation resources.
    
    Provides methods to load:
    - Original voice files from partition_dataset_by_speaker (separated by bonafide/spoof)
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
        rir_root: str = "data/noise_dataset/RIR"
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
    
    def _get_audio_files_recursive(self, directory: str):
        """Get all audio files recursively."""
        audio_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.wav', '.flac')):
                    audio_files.append(os.path.join(root, file))
        return sorted(audio_files)
    
    def load_train_files(self) -> List[Dict[str, str]]:
        """
        Load all training audio files from partition_dataset_by_speaker.

        Supports two directory structures:
        1. Split-first: voices_root/train/speaker_id/files (preferred)
        2. Speaker-first: voices_root/speaker_id/train/files (legacy)

        Returns:
            List of dictionaries containing file metadata:
            - filepath: Full path to audio file
            - speaker_id: Speaker identifier
            - split: 'train'
            - filename: Original filename
            - file_type: 'bonafide' or 'spoof'
        """
        train_files = []

        # Check for split-first structure (voices_root/train/speaker_id/)
        train_dir = self.voices_root / "train"

        if train_dir.exists() and train_dir.is_dir():
            # Split-first structure
            speakers = [d for d in train_dir.iterdir() if d.is_dir()]
            print(f"\nLoading train files from {len(speakers)} speakers (split-first structure)...")

            for speaker_dir in speakers:
                speaker_id = speaker_dir.name
                audio_files = self._get_audio_files_recursive(str(speaker_dir))

                for audio_file in audio_files:
                    filename = Path(audio_file).name
                    file_type = "bonafide" if filename.startswith("bonafide_") else "spoof"

                    train_files.append({
                        "filepath": audio_file,
                        "speaker_id": speaker_id,
                        "split": "train",
                        "filename": filename,
                        "file_type": file_type
                    })
        else:
            # Speaker-first structure (legacy)
            speakers = [d for d in self.voices_root.iterdir() if d.is_dir()]
            print(f"\nLoading train files from {len(speakers)} speakers (speaker-first structure)...")

            for speaker_dir in speakers:
                speaker_id = speaker_dir.name
                speaker_train_dir = speaker_dir / "train"

                if not speaker_train_dir.exists():
                    print(f"Warning: No train directory for speaker {speaker_id}")
                    continue

                audio_files = self._get_audio_files_recursive(str(speaker_train_dir))

                for audio_file in audio_files:
                    filename = Path(audio_file).name
                    file_type = "bonafide" if filename.startswith("bonafide_") else "spoof"

                    train_files.append({
                        "filepath": audio_file,
                        "speaker_id": speaker_id,
                        "split": "train",
                        "filename": filename,
                        "file_type": file_type
                    })

        print(f"Loaded {len(train_files)} training files")

        return train_files
    
    def load_bonafide_train_files(self) -> List[Dict[str, str]]:
        """
        Load only bonafide training files.
        
        Returns:
            List of bonafide file metadata dictionaries.
        """
        all_files = self.load_train_files()
        bonafide_files = [f for f in all_files if f["file_type"] == "bonafide"]
        
        print(f"Filtered {len(bonafide_files)} bonafide training files")
        
        return bonafide_files
    
    def load_spoof_train_files(self) -> List[Dict[str, str]]:
        """
        Load only spoof training files.
        
        Returns:
            List of spoof file metadata dictionaries.
        """
        all_files = self.load_train_files()
        spoof_files = [f for f in all_files if f["file_type"] == "spoof"]
        
        print(f"Filtered {len(spoof_files)} spoof training files")
        
        return spoof_files
    
    def load_dev_files(self) -> List[Dict[str, str]]:
        """
        Load all dev (validation) audio files.

        Supports two directory structures:
        1. Split-first: voices_root/dev/speaker_id/files (preferred)
        2. Speaker-first: voices_root/speaker_id/dev/files (legacy)

        Returns:
            List of dictionaries containing file metadata.
        """
        dev_files = []

        # Check for split-first structure (voices_root/dev/speaker_id/)
        dev_dir = self.voices_root / "dev"

        if dev_dir.exists() and dev_dir.is_dir():
            # Split-first structure
            speakers = [d for d in dev_dir.iterdir() if d.is_dir()]
            print(f"\nLoading dev files from {len(speakers)} speakers (split-first structure)...")

            for speaker_dir in speakers:
                speaker_id = speaker_dir.name
                audio_files = self._get_audio_files_recursive(str(speaker_dir))

                for audio_file in audio_files:
                    filename = Path(audio_file).name
                    file_type = "bonafide" if filename.startswith("bonafide_") else "spoof"

                    dev_files.append({
                        "filepath": audio_file,
                        "speaker_id": speaker_id,
                        "split": "dev",
                        "filename": filename,
                        "file_type": file_type
                    })
        else:
            # Speaker-first structure (legacy)
            speakers = [d for d in self.voices_root.iterdir() if d.is_dir()]
            print(f"\nLoading dev files from {len(speakers)} speakers (speaker-first structure)...")

            for speaker_dir in speakers:
                speaker_id = speaker_dir.name
                speaker_dev_dir = speaker_dir / "dev"

                if not speaker_dev_dir.exists():
                    continue

                audio_files = self._get_audio_files_recursive(str(speaker_dev_dir))

                for audio_file in audio_files:
                    filename = Path(audio_file).name
                    file_type = "bonafide" if filename.startswith("bonafide_") else "spoof"

                    dev_files.append({
                        "filepath": audio_file,
                        "speaker_id": speaker_id,
                        "split": "dev",
                        "filename": filename,
                        "file_type": file_type
                    })

        print(f"Loaded {len(dev_files)} dev files")

        return dev_files

    def load_bonafide_dev_files(self) -> List[Dict[str, str]]:
        """
        Load only bonafide dev files.

        Returns:
            List of bonafide file metadata dictionaries.
        """
        all_files = self.load_dev_files()
        bonafide_files = [f for f in all_files if f["file_type"] == "bonafide"]

        print(f"Filtered {len(bonafide_files)} bonafide dev files")

        return bonafide_files

    def load_spoof_dev_files(self) -> List[Dict[str, str]]:
        """
        Load only spoof dev files.

        Returns:
            List of spoof file metadata dictionaries.
        """
        all_files = self.load_dev_files()
        spoof_files = [f for f in all_files if f["file_type"] == "spoof"]

        print(f"Filtered {len(spoof_files)} spoof dev files")

        return spoof_files
    
    def load_eval_files(self) -> List[Dict[str, str]]:
        """
        Load all eval (test) audio files.

        Supports two directory structures:
        1. Split-first: voices_root/eval/speaker_id/files (preferred)
        2. Speaker-first: voices_root/speaker_id/eval/files (legacy)

        Returns:
            List of dictionaries containing file metadata.
        """
        eval_files = []

        # Check for split-first structure (voices_root/eval/speaker_id/)
        eval_dir = self.voices_root / "eval"

        if eval_dir.exists() and eval_dir.is_dir():
            # Split-first structure
            speakers = [d for d in eval_dir.iterdir() if d.is_dir()]
            print(f"\nLoading eval files from {len(speakers)} speakers (split-first structure)...")

            for speaker_dir in speakers:
                speaker_id = speaker_dir.name
                audio_files = self._get_audio_files_recursive(str(speaker_dir))

                for audio_file in audio_files:
                    filename = Path(audio_file).name
                    file_type = "bonafide" if filename.startswith("bonafide_") else "spoof"

                    eval_files.append({
                        "filepath": audio_file,
                        "speaker_id": speaker_id,
                        "split": "eval",
                        "filename": filename,
                        "file_type": file_type
                    })
        else:
            # Speaker-first structure (legacy)
            speakers = [d for d in self.voices_root.iterdir() if d.is_dir()]
            print(f"\nLoading eval files from {len(speakers)} speakers (speaker-first structure)...")

            for speaker_dir in speakers:
                speaker_id = speaker_dir.name
                speaker_eval_dir = speaker_dir / "eval"

                if not speaker_eval_dir.exists():
                    continue

                audio_files = self._get_audio_files_recursive(str(speaker_eval_dir))

                for audio_file in audio_files:
                    filename = Path(audio_file).name
                    file_type = "bonafide" if filename.startswith("bonafide_") else "spoof"

                    eval_files.append({
                        "filepath": audio_file,
                        "speaker_id": speaker_id,
                        "split": "eval",
                        "filename": filename,
                        "file_type": file_type
                    })

        print(f"Loaded {len(eval_files)} eval files")

        return eval_files

    def load_bonafide_eval_files(self) -> List[Dict[str, str]]:
        """
        Load only bonafide eval files.

        Returns:
            List of bonafide file metadata dictionaries.
        """
        all_files = self.load_eval_files()
        bonafide_files = [f for f in all_files if f["file_type"] == "bonafide"]

        print(f"Filtered {len(bonafide_files)} bonafide eval files")

        return bonafide_files

    def load_spoof_eval_files(self) -> List[Dict[str, str]]:
        """
        Load only spoof eval files.

        Returns:
            List of spoof file metadata dictionaries.
        """
        all_files = self.load_eval_files()
        spoof_files = [f for f in all_files if f["file_type"] == "spoof"]

        print(f"Filtered {len(spoof_files)} spoof eval files")

        return spoof_files
    
    def get_dataset_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the loaded dataset including bonafide/spoof breakdown.
        
        Returns:
            Dictionary with counts for each split and file type.
        """
        train_files = self.load_train_files()
        
        train_bonafide = len([f for f in train_files if f["file_type"] == "bonafide"])
        train_spoof = len([f for f in train_files if f["file_type"] == "spoof"])
        
        return {
            "train": {
                "total": len(train_files),
                "bonafide": train_bonafide,
                "spoof": train_spoof
            },
            "total": len(train_files)
        }
    
    def print_summary(self):
        """Print comprehensive dataset summary with bonafide/spoof breakdown."""
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        
        stats = self.get_dataset_statistics()
        
        print(f"\nFile counts:")
        print(f"  Train:  {stats['train']['total']:,} total")
        print(f"    - Bonafide: {stats['train']['bonafide']:,} ({stats['train']['bonafide']/stats['train']['total']*100:.1f}%)")
        print(f"    - Spoof:    {stats['train']['spoof']:,} ({stats['train']['spoof']/stats['train']['total']*100:.1f}%)")
        print("="*70 + "\n")


if __name__ == "__main__":
    loader = DatasetLoader(
        voices_root="data/partition_dataset_by_speaker",
        musan_root="data/noise_dataset/musan",
        rir_root="data/noise_dataset/RIR"
    )
    
    loader.print_summary()