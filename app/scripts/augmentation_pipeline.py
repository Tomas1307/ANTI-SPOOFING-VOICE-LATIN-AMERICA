"""
Dataset Partition Creation for Anti-Spoofing Voice Cloning
===========================================================

This script creates a partitioned dataset structure for training anti-spoofing
voice cloning models. It combines BONAFIDE (real human voices) and SPOOF 
(synthetic/converted voices) samples into train/val/test splits organized by speaker.

PARTITIONING STRATEGY:
---------------------

For each speaker, we create the following structure:

    partition_dataset_by_speaker/
    └── {speaker_id}/
        ├── train/      (80% of each type)
        ├── val/        (10% of each type)
        └── test/       (10% of each type)

Within each split (train/val/test), we include:

1. BONAFIDE samples:
   - Original recordings from Real/{Country}/{speaker_id}/
   - Split: 80% train, 10% val, 10% test

2. SPOOF samples (where speaker is SOURCE):
   - CycleGAN: Voice conversion attacks (speaker → target)
   - Diff: Diffusion-based conversion attacks
   - StarGAN: Multi-domain conversion attacks
   - TTS: Text-to-speech synthesis
   - TTS-VC: Hybrid TTS + conversion attacks
   
   Each spoof technique is split independently (80/10/10) to maintain 
   technique-specific balance and prevent data leakage.

WHY THIS STRUCTURE?
-------------------

1. Speaker-centric: All data (bonafide + spoof) for a speaker is grouped together
2. Binary classification: Model learns to detect "real vs fake" regardless of technique
3. Technique balance: 80/10/10 split within each spoof type prevents leakage
4. Train efficiency: Model sees diverse attacks during training

EXPECTED INPUT STRUCTURE:
-------------------------

    FinalDataset_16khz/
    ├── Real/
    │   ├── Argentina/{speaker_id}/*.wav
    │   ├── Chile/{speaker_id}/*.wav
    │   ├── Colombia/{speaker_id}/*.wav
    │   ├── Peru/{speaker_id}/*.wav
    │   └── Venezuela/{speaker_id}/*.wav
    ├── CycleGAN/{Country}-{Country}/{source_speaker}-{target_speaker}/*.wav
    ├── Diff/{Country}-{Country}/{source_speaker}-{target_speaker}/*.wav
    ├── StarGAN/... (similar structure)
    ├── TTS/{Country}/TTS-{speaker_id}_*.wav
    └── TTS-VC/TTS-{technique}/{Country}-{Country}/TTS-{technique}-{source}_{timestamp}_TTS-{target}_{index}.wav

USAGE:
------

    python dataset_partition_creation.py \\
        --source_dir Latin_America_Spanish_anti_spoofing_dataset/FinalDataset_16khz \\
        --output_dir partition_dataset_by_speaker \\
        --split_ratios 0.8 0.1 0.1 \\
        --seed 42

NOTES:
------

- Files are COPIED (not moved) to preserve original dataset
- Random seed ensures reproducibility
- Only speakers with bonafide samples are included
- Spoof samples are optional (if none found, only bonafide is partitioned)
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import augmenters
from app.augmenter.rir_augmenter import RIRAugmenter
from app.augmenter.codec_augmenter import CodecAugmenter
from app.augmenter.rawboost_augmenter import RawBoostAugmenter

# Import utilities
from app.utils.augmentation_calculator import AugmentationModeCalculator, AugmentationFactors
from app.dataset_loader import DatasetLoader
from app.config.augmentation_config import AugmentationConfigManager
from app.schema import AugmentationType
import app.utils.utils as utils


class AugmentationPipeline:
    """
    Complete augmentation pipeline with all features integrated.
    
    Features:
    - Simple mode: uniform factor for all files
    - Balanced mode: calculated factors for target ratio
    - Real augmentation (RIR, Codec, RawBoost)
    - ASVspoof format output
    - Protocol file generation
    """
    
    def __init__(
        self,
        voices_root: str = "data/partition_dataset_by_speaker",
        musan_root: str = "data/noise_dataset/musan",
        rir_root: str = "data/noise_dataset/RIR",
        output_root: str = "data/augmented",
        mode: str = "simple",
        factor: str = "3x",
        target_ratio: float = 0.50,
        seed: int = 42
    ):
        """
        Initialize complete augmentation pipeline.
        
        Args:
            voices_root: Path to partitioned dataset
            musan_root: Path to MUSAN noise dataset
            rir_root: Path to RIR files
            output_root: Output directory
            mode: "simple" or "balanced"
            factor: Augmentation factor (e.g., "3x", "5x")
            target_ratio: Target bonafide ratio for balanced mode
            seed: Random seed
        """
        self.voices_root = Path(voices_root)
        self.musan_root = Path(musan_root)
        self.rir_root = Path(rir_root)
        self.output_root = Path(output_root)
        self.mode = mode
        self.factor = factor
        self.target_ratio = target_ratio
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Parse factor
        self.factor_num = int(factor.replace('x', ''))
        
        # Create output directory name
        if mode == "simple":
            self.output_dir = self.output_root / f"augmented_{factor}_simple"
        else:
            ratio_str = f"{int(target_ratio*100)}{int((1-target_ratio)*100)}"
            self.output_dir = self.output_root / f"augmented_{factor}_balanced_{ratio_str}"
        
        # Initialize components
        self.loader = DatasetLoader(
            voices_root=str(self.voices_root),
            musan_root=str(self.musan_root),
            rir_root=str(self.rir_root)
        )
        
        self.calculator = AugmentationModeCalculator()
        
        # Load augmentation config
        config_manager = AugmentationConfigManager.get_instance()
        self.strategy = config_manager.get_strategy(factor)
        
        # Initialize augmenters
        print("\nInitializing augmenters...")
        
        self.rir_augmenter = RIRAugmenter(
            config=self.strategy.rir_noise_config,
            rir_root=str(self.rir_root),
            noise_root=str(self.musan_root)
        )
        
        self.codec_augmenter = CodecAugmenter(
            config=self.strategy.codec_config
        )
        
        self.rawboost_augmenter = RawBoostAugmenter(
            config=self.strategy.rawboost_config
        )
        
        # Statistics
        self.stats = {
            'train': {'bonafide': 0, 'spoof': 0, 'total': 0},
            'dev': {'bonafide': 0, 'spoof': 0, 'total': 0},
            'eval': {'bonafide': 0, 'spoof': 0, 'total': 0}
        }
        
        # Audio ID counters
        self.audio_id_counter = {
            'train': 1,
            'dev': 1,
            'eval': 1
        }
        
        # Protocol entries
        self.protocol_entries = {
            'train': [],
            'dev': [],
            'eval': []
        }
    
    def _select_augmentation_type(self) -> AugmentationType:
        """
        Select augmentation type based on strategy distribution.
        
        Returns:
            Selected AugmentationType
        """
        types = list(self.strategy.type_distribution.keys())
        probabilities = [self.strategy.type_distribution[t] for t in types]
        
        return random.choices(types, weights=probabilities, k=1)[0]
    
    def _apply_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        aug_type: AugmentationType
    ) -> Tuple[np.ndarray, str]:
        """
        Apply specific augmentation to audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            aug_type: Type of augmentation
            
        Returns:
            Tuple of (augmented_audio, system_id)
        """
        if aug_type == AugmentationType.RIR_NOISE:
            augmented, metadata = self.rir_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.rir_augmenter.get_augmentation_label(
                metadata['room_size'],
                metadata['noise_source'],
                metadata['snr_db']
            )
        
        elif aug_type == AugmentationType.CODEC:
            augmented, metadata = self.codec_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.codec_augmenter.get_augmentation_label(
                metadata['codec_sr'],
                metadata['packet_loss']
            )
        
        elif aug_type == AugmentationType.RAWBOOST:
            augmented, metadata = self.rawboost_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.rawboost_augmenter.get_augmentation_label(
                metadata['operations']
            )
        
        else:
            augmented = audio
            system_id = "-"
        
        return augmented, system_id
    
    def _generate_audio_id(self, split: str) -> str:
        """
        Generate ASVspoof-style audio ID.
        
        Args:
            split: 'train', 'dev', or 'eval'
            
        Returns:
            Audio ID (e.g., "LA_T_0000001")
        """
        prefix_map = {
            'train': 'LA_T',
            'dev': 'LA_D',
            'eval': 'LA_E'
        }
        
        prefix = prefix_map[split]
        audio_id = f"{prefix}_{self.audio_id_counter[split]:07d}"
        self.audio_id_counter[split] += 1
        
        return audio_id
    
    def _save_audio_and_protocol(
        self,
        audio: np.ndarray,
        sr: int,
        split: str,
        speaker_id: str,
        system_id: str,
        key: str
    ):
        """
        Save audio as FLAC and add protocol entry.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            split: 'train', 'dev', or 'eval'
            speaker_id: Speaker ID
            system_id: Augmentation system ID or '-'
            key: 'bonafide' or 'spoof'
        """
        # Generate audio ID
        audio_id = self._generate_audio_id(split)
        
        # Create output path
        flac_dir = self.output_dir / "LA" / f"ASVspoof2019_LA_{split}" / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = flac_dir / f"{audio_id}.flac"
        
        # Save audio
        utils.save_audio_flac(audio, str(audio_path), sr=sr)
        
        # Add protocol entry
        protocol_entry = f"{speaker_id} {audio_id} {system_id} {key}"
        self.protocol_entries[split].append(protocol_entry)
        
        # Update stats
        self.stats[split][key] += 1
        self.stats[split]['total'] += 1
    
    def _process_file(
        self,
        file_info: Dict,
        split: str,
        n_copies: int,
        key: str
    ):
        """
        Process a single file with augmentation.
        
        Args:
            file_info: File metadata
            split: 'train', 'dev', or 'eval'
            n_copies: Number of copies to create
            key: 'bonafide' or 'spoof'
        """
        # Load audio
        audio, sr = utils.load_audio(file_info['filepath'])
        speaker_id = file_info['speaker_id']
        
        # Save original (always)
        self._save_audio_and_protocol(
            audio=audio,
            sr=sr,
            split=split,
            speaker_id=speaker_id,
            system_id="-",
            key=key
        )
        
        # Create augmented copies
        for _ in range(n_copies - 1):  # -1 because we saved original
            # Select augmentation type
            aug_type = self._select_augmentation_type()
            
            # Apply augmentation
            augmented, system_id = self._apply_augmentation(audio, sr, aug_type)
            
            # Save augmented
            self._save_audio_and_protocol(
                audio=augmented,
                sr=sr,
                split=split,
                speaker_id=speaker_id,
                system_id=system_id,
                key=key
            )
    
    def _process_split(
        self,
        split: str,
        bonafide_factor: int,
        spoof_factor: int
    ):
        """
        Process a complete split (train/dev/eval).
        
        Args:
            split: 'train', 'dev', or 'eval'
            bonafide_factor: Copies per bonafide file
            spoof_factor: Copies per spoof file
        """
        print(f"\nProcessing {split} split...")
        
        # Load files for this split
        if split == 'train':
            bonafide_files = self.loader.load_bonafide_train_files()
            spoof_files = self.loader.load_spoof_train_files()
        elif split == 'dev':
            bonafide_files = self.loader.load_bonafide_val_files()
            spoof_files = self.loader.load_spoof_val_files()
        elif split == 'eval':
            bonafide_files = self.loader.load_bonafide_test_files()
            spoof_files = self.loader.load_spoof_test_files()
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"  Bonafide: {len(bonafide_files)} files x {bonafide_factor}")
        print(f"  Spoof:    {len(spoof_files)} files x {spoof_factor}")
        
        # Process bonafide
        print(f"  Processing bonafide...")
        for file_info in tqdm(bonafide_files, desc=f"  {split} bonafide"):
            self._process_file(file_info, split, bonafide_factor, 'bonafide')
        
        # Process spoof
        print(f"  Processing spoof...")
        for file_info in tqdm(spoof_files, desc=f"  {split} spoof"):
            self._process_file(file_info, split, spoof_factor, 'spoof')
    
    def _write_protocol_files(self):
        """Write protocol files for all splits."""
        print("\nWriting protocol files...")
        
        protocol_map = {
            'train': 'ASVspoof2019.LA.cm.train.trn.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
        }
        
        for split, filename in protocol_map.items():
            if self.protocol_entries[split]:
                protocol_dir = self.output_dir / "LA" / f"ASVspoof2019_LA_{split}"
                protocol_dir.mkdir(parents=True, exist_ok=True)
                
                protocol_path = protocol_dir / filename
                
                with open(protocol_path, 'w') as f:
                    for entry in self.protocol_entries[split]:
                        f.write(entry + '\n')
                
                print(f"  Written: {filename} ({len(self.protocol_entries[split])} entries)")
    
    def _print_final_report(self, factors: AugmentationFactors):
        """
        Print final augmentation report.
        
        Args:
            factors: Calculated augmentation factors
        """
        print("\n" + "="*70)
        print("AUGMENTATION COMPLETE")
        print("="*70)
        
        print(f"\nMode: {self.mode.upper()}")
        print(f"Output: {self.output_dir}")
        
        print(f"\nFactors Applied (train only):")
        print(f"  Bonafide: {factors.bonafide_factor}x")
        print(f"  Spoof:    {factors.spoof_factor}x")
        print(f"  Total:    {factors.total_factor:.2f}x")
        
        if self.mode == "balanced":
            print(f"\nBalance:")
            print(f"  Target:   {factors.target_ratio[0]:.1f}% / {factors.target_ratio[1]:.1f}%")
            print(f"  Achieved: {factors.final_ratio[0]:.1f}% / {factors.final_ratio[1]:.1f}%")
        
        print(f"\nFinal Dataset:")
        for split in ['train', 'dev', 'eval']:
            if self.stats[split]['total'] > 0:
                bonafide = self.stats[split]['bonafide']
                spoof = self.stats[split]['spoof']
                total = self.stats[split]['total']
                bonafide_pct = (bonafide / total) * 100 if total > 0 else 0
                spoof_pct = (spoof / total) * 100 if total > 0 else 0
                
                print(f"\n  {split.upper()}:")
                print(f"    Total:    {total:,} files")
                print(f"    Bonafide: {bonafide:,} ({bonafide_pct:.1f}%)")
                print(f"    Spoof:    {spoof:,} ({spoof_pct:.1f}%)")
        
        print("\n" + "="*70 + "\n")
    
    def run(self):
        """Execute the complete augmentation pipeline."""
        print("\n" + "="*70)
        print("AUGMENTATION PIPELINE - ASVspoof2019 LA Format")
        print("="*70)
        
        print(f"\nConfiguration:")
        print(f"  Mode:         {self.mode}")
        print(f"  Factor:       {self.factor}")
        if self.mode == "balanced":
            print(f"  Target ratio: {self.target_ratio:.0%} bonafide")
        print(f"  Seed:         {self.seed}")
        print(f"  Output:       {self.output_dir}")
        
        # Load dataset stats
        print("\nLoading dataset...")
        stats = self.loader.get_dataset_statistics()
        
        n_bonafide = stats['train']['bonafide']
        n_spoof = stats['train']['spoof']
        
        print(f"  Train bonafide: {n_bonafide:,}")
        print(f"  Train spoof:    {n_spoof:,}")
        
        # Calculate augmentation factors
        print("\nCalculating augmentation factors...")
        
        if self.mode == "simple":
            factors = self.calculator.calculate_simple_mode(
                n_bonafide=n_bonafide,
                n_spoof=n_spoof,
                factor=self.factor_num
            )
        else:  # balanced
            factors = self.calculator.calculate_balanced_mode(
                n_bonafide=n_bonafide,
                n_spoof=n_spoof,
                target_ratio=self.target_ratio,
                min_total_factor=self.factor_num
            )
        
        # Print calculation summary
        self.calculator.print_calculation_summary(
            n_bonafide=n_bonafide,
            n_spoof=n_spoof,
            factors=factors,
            mode=self.mode
        )
        
        # Process train with augmentation
        self._process_split(
            split='train',
            bonafide_factor=factors.bonafide_factor,
            spoof_factor=factors.spoof_factor
        )
        
        # Process dev/eval WITHOUT augmentation (just copy)
        self._process_split(split='dev', bonafide_factor=1, spoof_factor=1)
        self._process_split(split='eval', bonafide_factor=1, spoof_factor=1)
        
        # Write protocol files
        self._write_protocol_files()
        
        # Print final report
        self._print_final_report(factors)


def main():
    """Test the pipeline."""
    # Simple mode test
    pipeline = AugmentationPipeline(
        mode="simple",
        factor="3x",
        seed=42
    )
    
    # Uncomment to run
    # pipeline.run()
    
    print("Pipeline initialized successfully!")


if __name__ == "__main__":
    main()