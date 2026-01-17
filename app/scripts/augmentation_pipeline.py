"""
Augmentation Pipeline for Anti-Spoofing Dataset
================================================

Creates augmented dataset in ASVspoof2019 LA format with state-of-the-art practices:

AUGMENTATION STRATEGY:
---------------------

1. SPEAKER-INDEPENDENT SPLITS:
   - Train/Val/Test use DIFFERENT speakers (no speaker overlap)
   - Prevents model from learning speaker-specific features
   - Ensures true generalization evaluation

2. BALANCED MODE:
   - Calculates different augmentation factors for bonafide vs spoof
   - Achieves target ratio (e.g., 50/50, 60/40) while maintaining clean data percentage
   - Example: If target is 50/50 with 3x minimum:
     * Bonafide: 5x augmentation (to increase representation)
     * Spoof: 2x augmentation (already have more)
     * Result: ~50/50 ratio achieved

3. CLEAN DATA PRESERVATION (25% Rule):
   - TRAIN: 25% clean (original) audio, 75% augmented
     * Prevents model from losing sensitivity to original voices
     * Clean audio always includes ALL originals (both bonafide and spoof)
     * Augmented copies calculated to reach 25% clean ratio
   - VAL: 100% clean (NO augmentation)
     * Pure evaluation during training
   - TEST: 100% clean (NO augmentation)
     * Final evaluation on unseen speakers

4. AUGMENTATION TYPES (Applied to train only):
   - RIR + Noise (60%): Room acoustics + background noise
   - Codec Degradation (30%): Telephone/VoIP artifacts
   - RawBoost (10%): Signal-dependent distortions

OUTPUT STRUCTURE (ASVspoof2019 LA Format):
------------------------------------------

    data/augmented_{factor}_balanced_{ratio}/
    └── LA/
        ├── ASVspoof2019_LA_train/
        │   ├── flac/
        │   │   ├── LA_T_0000001.flac
        │   │   └── ...
        │   └── ASVspoof2019.LA.cm.train.trn.txt
        ├── ASVspoof2019_LA_dev/
        │   ├── flac/
        │   │   ├── LA_D_0000001.flac
        │   │   └── ...
        │   └── ASVspoof2019.LA.cm.dev.trl.txt
        └── ASVspoof2019_LA_eval/
            ├── flac/
            │   ├── LA_E_0000001.flac
            │   └── ...
            └── ASVspoof2019.LA.cm.eval.trl.txt

PROTOCOL FILE FORMAT:
--------------------

    SPEAKER_ID AUDIO_ID SYSTEM_ID KEY
    arf_00295 LA_T_0000001 - bonafide
    arf_00295 LA_T_0000002 RIR_SMALL_NOI_SNR15 bonafide
    clf_00123 LA_T_0000003 - spoof
    clf_00123 LA_T_0000004 CODEC_8K_LOSS2PCT spoof

    Where:
    - SPEAKER_ID: Speaker identifier
    - AUDIO_ID: Unique file ID (LA_T_* for train, LA_D_* for dev, LA_E_* for eval)
    - SYSTEM_ID: "-" for original/clean, or augmentation label for augmented
    - KEY: "bonafide" or "spoof"

USAGE:
------

    # Balanced mode with 50/50 ratio and minimum 3x augmentation
    python run_augmentation.py \\
        --target_ratio 0.50 \\
        --min_factor 3x \\
        --seed 42

    # Balanced mode with 60/40 ratio (more bonafide) and 5x minimum
    python run_augmentation.py \\
        --target_ratio 0.60 \\
        --min_factor 5x

EXAMPLE CALCULATION:
-------------------

Input (train split):
  - 80 bonafide originals
  - 176 spoof originals
  - Total: 256 originals

Target: 50/50 ratio with minimum 3x augmentation and 25% clean

Step 1: Calculate total needed for 3x minimum
  - 256 * 3 = 768 total files minimum

Step 2: Calculate targets for 50/50
  - Bonafide target: 768 * 0.50 = 384 files
  - Spoof target: 768 * 0.50 = 384 files

Step 3: Calculate augmentation factors
  - Bonafide factor: 384 / 80 = 4.8 → 5x
  - Spoof factor: 384 / 176 = 2.2 → 2x

Step 4: Calculate actual totals
  - Bonafide: 80 * 5 = 400 files (53.2%)
  - Spoof: 176 * 2 = 352 files (46.8%)
  - Total: 752 files (2.9x actual)

Step 5: Verify clean data ratio
  - Clean: 256 originals
  - Total: 752 files
  - Clean ratio: 256/752 = 34.0% (> 25% target ✓)

Note: We ALWAYS include ALL originals, so clean ratio may exceed 25%
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
    State-of-the-art augmentation pipeline for anti-spoofing.
    
    Features:
    - Balanced mode with calculated factors
    - Speaker-independent splits
    - 25% clean data preservation in train
    - ASVspoof2019 LA format output
    """
    
    def __init__(
        self,
        voices_root: str = "data/partition_dataset_by_speaker",
        musan_root: str = "data/noise_dataset/musan",
        rir_root: str = "data/noise_dataset/RIR",
        output_root: str = "data/augmented",
        target_ratio: float = 0.50,
        min_factor: str = "3x",
        seed: int = 42
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            voices_root: Path to speaker-independent partitioned dataset
            musan_root: Path to MUSAN noise dataset
            rir_root: Path to RIR files
            output_root: Output directory
            target_ratio: Target bonafide ratio (0.0-1.0)
            min_factor: Minimum total augmentation factor (e.g., "3x", "5x")
            seed: Random seed for reproducibility
        """
        self.voices_root = Path(voices_root)
        self.musan_root = Path(musan_root)
        self.rir_root = Path(rir_root)
        self.output_root = Path(output_root)
        self.target_ratio = target_ratio
        self.min_factor = min_factor
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Parse factor
        self.factor_num = int(min_factor.replace('x', ''))
        
        # Create output directory name
        ratio_str = f"{int(target_ratio*100)}{int((1-target_ratio)*100)}"
        self.output_dir = self.output_root / f"augmented_{min_factor}_balanced_{ratio_str}"
        
        # Initialize components
        self.loader = DatasetLoader(
            voices_root=str(self.voices_root),
            musan_root=str(self.musan_root),
            rir_root=str(self.rir_root)
        )
        
        self.calculator = AugmentationModeCalculator()
        
        # Load augmentation config
        config_manager = AugmentationConfigManager.get_instance()
        self.strategy = config_manager.get_strategy(min_factor)
        
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
            'train': {'bonafide': 0, 'spoof': 0, 'total': 0, 'clean': 0, 'augmented': 0},
            'dev': {'bonafide': 0, 'spoof': 0, 'total': 0, 'clean': 0, 'augmented': 0},
            'eval': {'bonafide': 0, 'spoof': 0, 'total': 0, 'clean': 0, 'augmented': 0}
        }
        
        # Audio ID counters
        self.audio_id_counter = {'train': 1, 'dev': 1, 'eval': 1}
        
        # Protocol entries
        self.protocol_entries = {'train': [], 'dev': [], 'eval': []}
    
    def _select_augmentation_type(self) -> AugmentationType:
        """Select augmentation type based on strategy distribution."""
        types = list(self.strategy.type_distribution.keys())
        probabilities = [self.strategy.type_distribution[t] for t in types]
        return random.choices(types, weights=probabilities, k=1)[0]
    
    def _apply_augmentation(
        self,
        audio: np.ndarray,
        sr: int,
        aug_type: AugmentationType
    ) -> Tuple[np.ndarray, str]:
        """Apply specific augmentation to audio."""
        if aug_type == AugmentationType.RIR_NOISE:
            augmented, metadata = self.rir_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.rir_augmenter.get_augmentation_label(
                metadata['room_size'], metadata['noise_source'], metadata['snr_db']
            )
        elif aug_type == AugmentationType.CODEC:
            augmented, metadata = self.codec_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.codec_augmenter.get_augmentation_label(
                metadata['codec_sr'], metadata['packet_loss']
            )
        elif aug_type == AugmentationType.RAWBOOST:
            augmented, metadata = self.rawboost_augmenter.augment(audio, sr, return_metadata=True)
            system_id = self.rawboost_augmenter.get_augmentation_label(metadata['operations'])
        else:
            augmented = audio
            system_id = "-"
        
        return augmented, system_id
    
    def _generate_audio_id(self, split: str) -> str:
        """Generate ASVspoof-style audio ID."""
        prefix_map = {'train': 'LA_T', 'dev': 'LA_D', 'eval': 'LA_E'}
        prefix = prefix_map[split]
        audio_id = f"{prefix}_{self.audio_id_counter[split]:07d}"
        self.audio_id_counter[split] += 1
        return audio_id
    
    def _save_audio_and_protocol(
        self, audio: np.ndarray, sr: int, split: str,
        speaker_id: str, system_id: str, key: str
    ):
        """Save audio as FLAC and add protocol entry."""
        audio_id = self._generate_audio_id(split)
        
        flac_dir = self.output_dir / "LA" / f"ASVspoof2019_LA_{split}" / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = flac_dir / f"{audio_id}.flac"
        utils.save_audio_flac(audio, str(audio_path), sr=sr)
        
        protocol_entry = f"{speaker_id} {audio_id} {system_id} {key}"
        self.protocol_entries[split].append(protocol_entry)
        
        self.stats[split][key] += 1
        self.stats[split]['total'] += 1
        
        if system_id == "-":
            self.stats[split]['clean'] += 1
        else:
            self.stats[split]['augmented'] += 1
    
    def _process_file(
        self, file_info: Dict, split: str, n_copies: int, key: str
    ):
        """Process a single file with augmentation."""
        audio, sr = utils.load_audio(file_info['filepath'])
        speaker_id = file_info['speaker_id']
        
        # Save original (always)
        self._save_audio_and_protocol(audio, sr, split, speaker_id, "-", key)
        
        # Create augmented copies
        for _ in range(n_copies - 1):
            aug_type = self._select_augmentation_type()
            augmented, system_id = self._apply_augmentation(audio, sr, aug_type)
            self._save_audio_and_protocol(augmented, sr, split, speaker_id, system_id, key)
    
    def _process_split(
        self, split: str, bonafide_factor: int, spoof_factor: int
    ):
        """Process a complete split."""
        print(f"\nProcessing {split} split...")
        
        if split == 'train':
            bonafide_files = self.loader.load_bonafide_train_files()
            spoof_files = self.loader.load_spoof_train_files()
        elif split == 'dev':
            bonafide_files = self.loader.load_bonafide_val_files()
            spoof_files = self.loader.load_spoof_val_files()
        elif split == 'eval':
            bonafide_files = self.loader.load_bonafide_test_files()
            spoof_files = self.loader.load_spoof_test_files()
        
        print(f"  Bonafide: {len(bonafide_files)} files x {bonafide_factor}")
        print(f"  Spoof:    {len(spoof_files)} files x {spoof_factor}")
        
        for file_info in tqdm(bonafide_files, desc=f"  {split} bonafide"):
            self._process_file(file_info, split, bonafide_factor, 'bonafide')
        
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
                protocol_path = protocol_dir / filename
                
                with open(protocol_path, 'w') as f:
                    for entry in self.protocol_entries[split]:
                        f.write(entry + '\n')
                
                print(f"  Written: {filename} ({len(self.protocol_entries[split])} entries)")
    
    def _print_final_report(self, factors: AugmentationFactors):
        """Print comprehensive final report."""
        print("\n" + "="*70)
        print("AUGMENTATION COMPLETE")
        print("="*70)
        
        print(f"\nConfiguration:")
        print(f"  Mode:         BALANCED")
        print(f"  Target ratio: {self.target_ratio:.0%} bonafide")
        print(f"  Min factor:   {self.min_factor}")
        print(f"  Seed:         {self.seed}")
        print(f"  Output:       {self.output_dir}")
        
        print(f"\nCalculated Factors (train only):")
        print(f"  Bonafide: {factors.bonafide_factor}x")
        print(f"  Spoof:    {factors.spoof_factor}x")
        print(f"  Total:    {factors.total_factor:.2f}x")
        
        print(f"\nBalance Achievement:")
        print(f"  Target:   {factors.target_ratio[0]:.1f}% / {factors.target_ratio[1]:.1f}%")
        print(f"  Achieved: {factors.final_ratio[0]:.1f}% / {factors.final_ratio[1]:.1f}%")
        deviation = abs(factors.final_ratio[0] - factors.target_ratio[0])
        print(f"  Deviation: ±{deviation:.1f}%")
        
        print(f"\nFinal Dataset Statistics:")
        for split in ['train', 'dev', 'eval']:
            if self.stats[split]['total'] > 0:
                s = self.stats[split]
                bonafide_pct = (s['bonafide'] / s['total']) * 100
                spoof_pct = (s['spoof'] / s['total']) * 100
                clean_pct = (s['clean'] / s['total']) * 100
                
                print(f"\n  {split.upper()}:")
                print(f"    Total:    {s['total']:,} files")
                print(f"    Bonafide: {s['bonafide']:,} ({bonafide_pct:.1f}%)")
                print(f"    Spoof:    {s['spoof']:,} ({spoof_pct:.1f}%)")
                print(f"    Clean:    {s['clean']:,} ({clean_pct:.1f}%)")
                print(f"    Augmented: {s['augmented']:,} ({100-clean_pct:.1f}%)")
        
        print("="*70 + "\n")
    
    def run(self):
        """Execute the complete augmentation pipeline."""
        print("\n" + "="*70)
        print("ANTI-SPOOFING AUGMENTATION PIPELINE")
        print("="*70)
        
        print(f"\nConfiguration:")
        print(f"  Target ratio: {self.target_ratio:.0%} bonafide")
        print(f"  Min factor:   {self.min_factor}")
        print(f"  Seed:         {self.seed}")
        
        # Load dataset stats
        print("\nLoading dataset...")
        stats = self.loader.get_dataset_statistics()
        
        n_bonafide = stats['train']['bonafide']
        n_spoof = stats['train']['spoof']
        
        print(f"  Train bonafide: {n_bonafide:,}")
        print(f"  Train spoof:    {n_spoof:,}")
        
        # Calculate factors
        print("\nCalculating balanced augmentation factors...")
        factors = self.calculator.calculate_balanced_mode(
            n_bonafide=n_bonafide,
            n_spoof=n_spoof,
            target_ratio=self.target_ratio,
            min_total_factor=self.factor_num
        )
        
        self.calculator.print_calculation_summary(n_bonafide, n_spoof, factors, "balanced")
        
        # Process splits
        self._process_split('train', factors.bonafide_factor, factors.spoof_factor)
        self._process_split('dev', 1, 1)  # No augmentation
        self._process_split('eval', 1, 1)  # No augmentation
        
        # Write protocols
        self._write_protocol_files()
        
        # Final report
        self._print_final_report(factors)


if __name__ == "__main__":
    pipeline = AugmentationPipeline(target_ratio=0.50, min_factor="3x")
    print("Pipeline initialized successfully!")