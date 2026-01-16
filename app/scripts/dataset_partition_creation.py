"""
Speaker-Independent Dataset Partition Creation for Anti-Spoofing
================================================================

CRITICAL DIFFERENCE FROM SPEAKER-DEPENDENT APPROACH:
---------------------------------------------------

This script creates SPEAKER-INDEPENDENT partitions where entire speakers
are assigned to train/val/test splits, NOT individual files.

SPEAKER-INDEPENDENT (This script):
    partition_dataset_by_speaker/
    ├── train/                    ← 80% of SPEAKERS
    │   ├── speaker_001/
    │   │   ├── bonafide_speaker_001_0000.wav
    │   │   ├── spoof_cyclegan_speaker_001_0000.wav
    │   │   └── ... (ALL files for this speaker)
    │   ├── speaker_002/
    │   └── ...
    ├── val/                      ← 10% of SPEAKERS
    │   ├── speaker_101/
    │   └── ...
    └── test/                     ← 10% of SPEAKERS
        ├── speaker_151/
        └── ...

WHY SPEAKER-INDEPENDENT?
------------------------

❌ WRONG (Speaker-Dependent):
   - Speaker A has files in train, val, AND test
   - Model learns speaker-specific features
   - Test performance is INFLATED (not true generalization)

✅ CORRECT (Speaker-Independent):
   - Speaker A ONLY in train OR val OR test
   - Model must generalize to UNSEEN speakers
   - TRUE evaluation of anti-spoofing capability

EXPECTED INPUT STRUCTURE:
------------------------

    FinalDataset_16khz/
    ├── Real/
    │   ├── Argentina/{speaker_id}/*.wav
    │   ├── Chile/{speaker_id}/*.wav
    │   └── ...
    ├── CycleGAN/{Country}-{Country}/{source}-{target}/*.wav
    ├── Diff/...
    ├── StarGAN/...
    ├── TTS/{Country}/TTS-{speaker_id}_*.wav
    └── TTS-VC/...

USAGE:
------

    python dataset_partition_creation.py \\
        --source_dir Latin_America_Spanish_anti_spoofing_dataset/FinalDataset_16khz \\
        --output_dir partition_dataset_by_speaker \\
        --split_ratios 0.8 0.1 0.1 \\
        --seed 42

FEATURES:
---------

✓ Checkpoint system for resumable processing
✓ Detailed progress tracking
✓ Per-technique spoof statistics
✓ Automatic directory validation
✓ Reproducible random seed
"""

import os
import shutil
import random
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm
from app.utils.checkpoint_manager import PartitionCheckpointManager


class SpeakerIndependentPartitioner:
    """
    Creates speaker-independent train/val/test partitions.
    
    Key principle: Each speaker appears in ONLY ONE split.
    """
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        checkpoint_file: str = "partition_checkpoint.json",
        save_interval: int = 10
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.split_ratios = split_ratios
        self.seed = seed
        self.save_interval = save_interval
        
        # Checkpoint manager
        self.checkpoint_manager = PartitionCheckpointManager(checkpoint_file)
        
        # Validate
        if not abs(sum(split_ratios) - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
        random.seed(seed)
        
        # Statistics
        self.stats = {
            'train': {'speakers': 0, 'bonafide': 0, 'spoof': defaultdict(int)},
            'val': {'speakers': 0, 'bonafide': 0, 'spoof': defaultdict(int)},
            'test': {'speakers': 0, 'bonafide': 0, 'spoof': defaultdict(int)}
        }
        
        self.processed_speakers = []
        self.speaker_splits = {}
    
    def _collect_all_speakers(self) -> List[str]:
        """Collect all speaker IDs from Real/ directory."""
        print(f"\n{'='*70}")
        print("STEP 1: Collecting Speakers")
        print(f"{'='*70}\n")
        
        speakers = set()
        real_dir = self.source_dir / "Real"
        
        if not real_dir.exists():
            raise FileNotFoundError(f"Real directory not found: {real_dir}")
        
        for country_dir in real_dir.iterdir():
            if country_dir.is_dir():
                for speaker_dir in country_dir.iterdir():
                    if speaker_dir.is_dir():
                        speakers.add(speaker_dir.name)
        
        speakers = sorted(list(speakers))
        print(f"Found {len(speakers)} unique speakers")
        
        return speakers
    
    def _split_speakers_into_sets(self, speakers: List[str]) -> Dict[str, List[str]]:
        """
        Split speakers into train/val/test sets.
        
        Returns:
            Dictionary mapping split name to list of speaker IDs
        """
        print(f"\n{'='*70}")
        print("STEP 2: Splitting Speakers into Sets")
        print(f"{'='*70}\n")
        
        shuffled = speakers.copy()
        random.shuffle(shuffled)
        
        n_total = len(speakers)
        n_train = int(n_total * self.split_ratios[0])
        n_val = int(n_total * self.split_ratios[1])
        
        splits = {
            'train': shuffled[:n_train],
            'val': shuffled[n_train:n_train + n_val],
            'test': shuffled[n_train + n_val:]
        }
        
        print(f"Train: {len(splits['train'])} speakers ({len(splits['train'])/n_total*100:.1f}%)")
        print(f"Val:   {len(splits['val'])} speakers ({len(splits['val'])/n_total*100:.1f}%)")
        print(f"Test:  {len(splits['test'])} speakers ({len(splits['test'])/n_total*100:.1f}%)")
        
        return splits
    
    def _get_bonafide_files(self, speaker_id: str) -> List[Path]:
        """Get all bonafide files for a speaker."""
        files = []
        real_dir = self.source_dir / "Real"
        
        for country_dir in real_dir.iterdir():
            if not country_dir.is_dir():
                continue
            speaker_dir = country_dir / speaker_id
            if speaker_dir.exists():
                files.extend(list(speaker_dir.glob("*.wav")))
        
        return files
    
    def _get_spoof_files(self, speaker_id: str) -> Dict[str, List[Path]]:
        """Get all spoof files where speaker is SOURCE."""
        spoof_files = {
            'cyclegan': [],
            'diff': [],
            'stargan': [],
            'tts': [],
            'ttsvc': []
        }
        
        # CycleGAN
        cyclegan_dir = self.source_dir / "CycleGAN"
        if cyclegan_dir.exists():
            for pair_dir in cyclegan_dir.rglob(f"{speaker_id}-*"):
                if pair_dir.is_dir():
                    spoof_files['cyclegan'].extend(list(pair_dir.glob("*.wav")))
        
        # Diff
        diff_dir = self.source_dir / "Diff"
        if diff_dir.exists():
            for pair_dir in diff_dir.rglob(f"{speaker_id}-*"):
                if pair_dir.is_dir():
                    spoof_files['diff'].extend(list(pair_dir.glob("*.wav")))
        
        # StarGAN
        stargan_dir = self.source_dir / "StarGAN"
        if stargan_dir.exists():
            for pair_dir in stargan_dir.rglob(f"{speaker_id}-*"):
                if pair_dir.is_dir():
                    spoof_files['stargan'].extend(list(pair_dir.glob("*.wav")))
        
        # TTS
        tts_dir = self.source_dir / "TTS"
        if tts_dir.exists():
            for country_dir in tts_dir.iterdir():
                if country_dir.is_dir():
                    files = list(country_dir.glob(f"TTS-{speaker_id}_*.wav"))
                    spoof_files['tts'].extend(files)
        
        # TTS-VC
        ttsvc_dir = self.source_dir / "TTS-VC"
        if ttsvc_dir.exists():
            for technique_dir in ttsvc_dir.iterdir():
                if technique_dir.is_dir():
                    for pair_dir in technique_dir.rglob("*"):
                        if pair_dir.is_dir():
                            files = list(pair_dir.glob(f"*{speaker_id}_*.wav"))
                            spoof_files['ttsvc'].extend(files)
        
        return spoof_files
    
    def _copy_speaker_files(self, speaker_id: str, split: str):
        """Copy all files for a speaker to the corresponding split."""
        speaker_dir = self.output_dir / split / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy bonafide files
        bonafide_files = self._get_bonafide_files(speaker_id)
        for idx, src_file in enumerate(bonafide_files):
            dst_file = speaker_dir / f"bonafide_{speaker_id}_{idx:04d}.wav"
            shutil.copy2(src_file, dst_file)
            self.stats[split]['bonafide'] += 1
        
        # Copy spoof files
        spoof_files = self._get_spoof_files(speaker_id)
        for technique, files in spoof_files.items():
            for idx, src_file in enumerate(files):
                dst_file = speaker_dir / f"spoof_{technique}_{speaker_id}_{idx:04d}.wav"
                shutil.copy2(src_file, dst_file)
                self.stats[split]['spoof'][technique] += 1
        
        # Update speaker count
        self.stats[split]['speakers'] += 1
    
    def run(self):
        """Execute speaker-independent partitioning."""
        print(f"\n{'='*70}")
        print("SPEAKER-INDEPENDENT DATASET PARTITIONING")
        print(f"{'='*70}\n")
        print(f"Source:       {self.source_dir}")
        print(f"Output:       {self.output_dir}")
        print(f"Split ratios: {self.split_ratios}")
        print(f"Seed:         {self.seed}")
        
        # Check for checkpoint
        resume_from_checkpoint = False
        already_processed = set()
        
        if self.checkpoint_manager.exists():
            print(f"\nFound checkpoint: {self.checkpoint_manager.checkpoint_file}")
            response = input("Resume from checkpoint? (yes/no): ")
            
            if response.lower() == 'yes':
                resume_from_checkpoint = True
                checkpoint_data = self.checkpoint_manager.load_partition()
                already_processed = self.checkpoint_manager.get_processed_speakers()
                
                # Restore state
                self.processed_speakers = checkpoint_data['processed_speakers']
                self.speaker_splits = checkpoint_data['speaker_splits']
                self.stats = checkpoint_data['stats']
                
                print(f"Resuming: {len(already_processed)} speakers already processed")
            else:
                print("Starting fresh")
        
        # Check output directory
        if self.output_dir.exists() and not resume_from_checkpoint:
            print(f"\nWARNING: Output directory exists: {self.output_dir}")
            response = input("Continue and overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        
        # Collect and split speakers
        all_speakers = self._collect_all_speakers()
        
        if not all_speakers:
            print("\nNo speakers found!")
            return
        
        # Create or load speaker splits
        if not resume_from_checkpoint:
            self.speaker_splits = self._split_speakers_into_sets(all_speakers)
        
        # Process each split
        print(f"\n{'='*70}")
        print("STEP 3: Copying Files")
        print(f"{'='*70}\n")
        
        for split in ['train', 'val', 'test']:
            split_speakers = self.speaker_splits[split]
            
            # Filter out already processed
            pending = [s for s in split_speakers if s not in already_processed]
            
            if not pending:
                print(f"\n{split.upper()}: All {len(split_speakers)} speakers already processed")
                continue
            
            print(f"\n{split.upper()}: Processing {len(pending)} speakers...")
            
            for i, speaker_id in enumerate(tqdm(pending, desc=f"  {split}"), 1):
                self._copy_speaker_files(speaker_id, split)
                self.processed_speakers.append(speaker_id)
                
                # Save checkpoint periodically
                if i % self.save_interval == 0:
                    self.checkpoint_manager.save_partition(
                        self.processed_speakers,
                        self.speaker_splits,
                        self.stats
                    )
        
        # Save final state
        self.checkpoint_manager.save_partition(
            self.processed_speakers,
            self.speaker_splits,
            self.stats
        )
        
        # Save speaker splits to output
        splits_file = self.output_dir / "speaker_splits.json"
        with open(splits_file, 'w') as f:
            json.dump(self.speaker_splits, f, indent=2)
        
        # Print final report
        self._print_final_report()
        
        # Clear checkpoint
        print("\nClearing checkpoint...")
        self.checkpoint_manager.clear()
        print("Done!")
    
    def _print_final_report(self):
        """Print comprehensive final report."""
        print(f"\n{'='*70}")
        print("PARTITIONING COMPLETE")
        print(f"{'='*70}\n")
        
        total_speakers = sum(s['speakers'] for s in self.stats.values())
        total_bonafide = sum(s['bonafide'] for s in self.stats.values())
        total_spoof = sum(
            sum(s['spoof'].values()) for s in self.stats.values()
        )
        
        print(f"Total speakers: {total_speakers}")
        print(f"Total bonafide: {total_bonafide:,}")
        print(f"Total spoof:    {total_spoof:,}")
        print(f"Total files:    {total_bonafide + total_spoof:,}")
        
        print(f"\nSplit Statistics:")
        for split in ['train', 'val', 'test']:
            s = self.stats[split]
            total = s['bonafide'] + sum(s['spoof'].values())
            speaker_pct = (s['speakers'] / total_speakers * 100) if total_speakers > 0 else 0
            
            print(f"\n  {split.upper()}:")
            print(f"    Speakers:  {s['speakers']} ({speaker_pct:.1f}%)")
            print(f"    Bonafide:  {s['bonafide']:,}")
            
            spoof_total = sum(s['spoof'].values())
            print(f"    Spoof:     {spoof_total:,}")
            
            for technique, count in sorted(s['spoof'].items()):
                if count > 0:
                    print(f"      └─ {technique}: {count:,}")
            
            print(f"    Total:     {total:,}")
        
        print(f"\nOutput: {self.output_dir}")
        print(f"Speaker splits saved: {self.output_dir / 'speaker_splits.json'}")
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Speaker-Independent Dataset Partitioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--source_dir",
        type=str,
        default="Latin_America_Spanish_anti_spoofing_dataset/FinalDataset_16khz",
        help="Path to FinalDataset_16khz directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="partition_dataset_by_speaker",
        help="Output directory for partitions"
    )
    
    parser.add_argument(
        "--split_ratios",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test ratios (must sum to 1.0)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="partition_checkpoint.json",
        help="Checkpoint file for resumption"
    )
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N speakers"
    )
    
    args = parser.parse_args()
    
    partitioner = SpeakerIndependentPartitioner(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
        checkpoint_file=args.checkpoint_file,
        save_interval=args.save_interval
    )
    
    try:
        partitioner.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        print("Progress saved to checkpoint.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()