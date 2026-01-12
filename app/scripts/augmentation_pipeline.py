"""
Augmentation Pipeline Orchestrator

Main pipeline that coordinates the entire augmentation process:
- Loads original voice files
- Applies augmentation according to configured distributions
- Saves augmented files in ASVspoof-compatible structure
- Generates protocol files
- Copies validation/test files unchanged
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import shutil
from app.config.augmentation_config import AugmentationConfigManager, AugmentationType
from app.dataset_loader import DatasetLoader
from app.augmenter.rir_augmenter import RIRAugmenter
from app.augmenter.codec_augmenter import CodecAugmenter
from app.augmenter.rawboost_augmenter import RawBoostAugmenter
import app.utils as utils

class AugmentationPipeline:
    """
    Main augmentation pipeline orchestrator.
    
    Coordinates the entire augmentation process from loading original files
    to generating augmented datasets with ASVspoof-compatible structure.
    
    Attributes:
        augmentation_factor: Multiplication factor (3x, 5x, 10x).
        config_manager: Singleton configuration manager.
        strategy: Current augmentation strategy.
        loader: Dataset loader instance.
        augmenters: Dictionary of augmenter instances.
        output_root: Root output directory.
    """
    
    def __init__(
        self,
        augmentation_factor: str = "3x",
        voices_root: str = "data/partition_dataset_by_speaker",
        musan_root: str = "data/noise_dataset/musan",
        rir_root: str = "data/RIR",
        output_root: str = "data/augmented"
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentation_factor: "3x", "5x", or "10x".
            voices_root: Path to original voice files.
            musan_root: Path to MUSAN dataset.
            rir_root: Path to RIR files.
            output_root: Root directory for augmented output.
        """
        self.augmentation_factor = augmentation_factor
        self.output_root = Path(output_root)
        
        self.config_manager = AugmentationConfigManager.get_instance()
        self.strategy = self.config_manager.get_strategy(augmentation_factor)
        
        self.loader = DatasetLoader(voices_root, musan_root, rir_root)
        
        self.augmenters = {
            AugmentationType.RIR_NOISE: RIRAugmenter(
                self.strategy.rir_noise_config,
                rir_root=rir_root,
                noise_root=musan_root
            ),
            AugmentationType.CODEC: CodecAugmenter(
                self.strategy.codec_config
            ),
            AugmentationType.RAWBOOST: RawBoostAugmenter(
                self.strategy.rawboost_config
            )
        }
        
        print(f"\nAugmentationPipeline initialized:")
        print(f"  Factor: {augmentation_factor}")
        print(f"  Output: {output_root}")
    
    def _sample_augmentation_type(self) -> AugmentationType:
        """
        Sample augmentation type according to configured distribution.
        
        Returns:
            Sampled AugmentationType.
        """
        types = list(self.strategy.type_distribution.keys())
        probabilities = [self.strategy.type_distribution[t] for t in types]
        
        return random.choices(types, weights=probabilities, k=1)[0]
    
    def _augment_audio(
        self,
        audio: np.ndarray,
        sr: int,
        aug_type: AugmentationType
    ) -> tuple:
        """
        Apply augmentation of specified type.
        
        Args:
            audio: Input audio signal.
            sr: Sample rate.
            aug_type: Type of augmentation to apply.
            
        Returns:
            Tuple of (augmented_audio, metadata).
        """
        augmenter = self.augmenters[aug_type]
        return augmenter.augment(audio, sr, return_metadata=True)
    
    def _create_output_structure(self):
        """Create output directory structure."""
        output_dir = self.output_root / self.augmentation_factor
        
        for split in ["train", "dev", "eval"]:
            split_dir = output_dir / f"ASVspoof_LatinAmerica_{split}"
            flac_dir = split_dir / "flac"
            utils.ensure_dir(str(flac_dir))
        
        print(f"\nCreated output structure at: {output_dir}")
    
    def _process_train_files(self) -> List[Dict]:
        """
        Process training files with augmentation.
        
        Returns:
            List of protocol entries.
        """
        train_files = self.loader.load_train_files()
        
        output_dir = self.output_root / self.augmentation_factor / "ASVspoof_LatinAmerica_train" / "flac"
        
        protocol_entries = []
        audio_id_counter = 1
        
        print(f"\n{'='*70}")
        print(f"PROCESSING TRAINING FILES ({len(train_files)} originals)")
        print(f"{'='*70}\n")
        
        factor = int(self.augmentation_factor.replace('x', ''))
        
        for file_info in tqdm(train_files, desc="Augmenting train files"):
            filepath = file_info["filepath"]
            speaker_id = file_info["speaker_id"]
            
            try:
                audio, sr = utils.load_audio(filepath, sr=16000)
            except Exception as e:
                print(f"\nError loading {filepath}: {e}")
                continue
            
            if self.strategy.include_original:
                audio_id = utils.generate_audio_id(audio_id_counter)
                output_path = output_dir / f"{audio_id}.flac"
                
                utils.save_audio_flac(audio, str(output_path), sr=16000)
                
                entry = utils.create_protocol_entry(
                    speaker_id,
                    audio_id,
                    "ORIGINAL",
                    "bonafide"
                )
                protocol_entries.append(entry)
                audio_id_counter += 1
            
            for i in range(factor):
                aug_type = self._sample_augmentation_type()
                
                augmented, metadata = self._augment_audio(audio, sr, aug_type)
                
                audio_id = utils.generate_audio_id(audio_id_counter)
                output_path = output_dir / f"aug_{audio_id}.flac"
                
                utils.save_audio_flac(augmented, str(output_path), sr=16000)
                
                if aug_type == AugmentationType.RIR_NOISE:
                    aug_label = self.augmenters[aug_type].get_augmentation_label(**metadata)
                elif aug_type == AugmentationType.CODEC:
                    aug_label = self.augmenters[aug_type].get_augmentation_label(
                        metadata['codec_sr'],
                        metadata['packet_loss']
                    )
                else:
                    aug_label = self.augmenters[aug_type].get_augmentation_label(
                        metadata['operations']
                    )
                
                entry = utils.create_protocol_entry(
                    speaker_id,
                    f"aug_{audio_id}",
                    aug_label,
                    "bonafide"
                )
                protocol_entries.append(entry)
                audio_id_counter += 1
        
        return protocol_entries
    
    def _copy_split_unchanged(self, split: str, split_name: str) -> List[Dict]:
        """
        Copy validation or test files without augmentation.
        
        Args:
            split: "val" or "test".
            split_name: "dev" or "eval" for output directory.
            
        Returns:
            List of protocol entries.
        """
        if split == "val":
            files = self.loader.load_val_files()
        else:
            files = self.loader.load_test_files()
        
        output_dir = self.output_root / self.augmentation_factor / f"ASVspoof_LatinAmerica_{split_name}" / "flac"
        
        protocol_entries = []
        audio_id_counter = 1
        
        print(f"\nCopying {split} files unchanged...")
        
        for file_info in tqdm(files, desc=f"Copying {split} files"):
            filepath = file_info["filepath"]
            speaker_id = file_info["speaker_id"]
            
            try:
                audio, sr = utils.load_audio(filepath, sr=16000)
            except Exception as e:
                print(f"\nError loading {filepath}: {e}")
                continue
            
            audio_id = utils.generate_audio_id(audio_id_counter)
            output_path = output_dir / f"{audio_id}.flac"
            
            utils.save_audio_flac(audio, str(output_path), sr=16000)
            
            entry = utils.create_protocol_entry(
                speaker_id,
                audio_id,
                "ORIGINAL",
                "bonafide"
            )
            protocol_entries.append(entry)
            audio_id_counter += 1
        
        return protocol_entries
    
    def _save_protocol_file(self, entries: List[str], split: str):
        """
        Save protocol file for split.
        
        Args:
            entries: List of protocol entry strings.
            split: "train", "dev", or "eval".
        """
        output_dir = self.output_root / self.augmentation_factor / f"ASVspoof_LatinAmerica_{split}"
        protocol_path = output_dir / f"ASVspoof_LatinAmerica.cm.{split}.txt"
        
        with open(protocol_path, 'w') as f:
            f.write("# ASVspoof Latin America Protocol File\n")
            f.write("# Format: speaker_id audio_file augmentation_type key\n")
            for entry in entries:
                f.write(entry + "\n")
        
        print(f"Protocol file saved: {protocol_path}")
        print(f"  Entries: {len(entries)}")
    
    def _create_readme(self):
        """Create README file for augmented dataset."""
        output_dir = self.output_root / self.augmentation_factor
        readme_path = output_dir / "README.txt"
        
        stats = self.config_manager.calculate_dataset_sizes(
            18204,
            self.augmentation_factor
        )
        
        content = f"""
ASVspoof Latin America Augmented Dataset - {self.augmentation_factor}
{'='*70}

This dataset contains augmented voice samples for anti-spoofing research
focused on Latin American Spanish accents.

AUGMENTATION STRATEGY
{'='*70}
Augmentation Factor: {self.augmentation_factor}
Type Distribution:
  - RIR + Noise: {self.strategy.type_distribution[AugmentationType.RIR_NOISE]*100:.0f}%
  - Codec/Channel: {self.strategy.type_distribution[AugmentationType.CODEC]*100:.0f}%
  - RawBoost: {self.strategy.type_distribution[AugmentationType.RAWBOOST]*100:.0f}%

DATASET STATISTICS
{'='*70}
Training set:
  - Original samples: {stats['original']:,}
  - RIR+Noise augmented: {stats['rir_noise']:,}
  - Codec augmented: {stats['codec']:,}
  - RawBoost augmented: {stats['rawboost']:,}
  - Total: {stats['total']:,}

Validation set: 2,211 (original only, no augmentation)
Test set: 2,399 (original only, no augmentation)

DIRECTORY STRUCTURE
{'='*70}
ASVspoof_LatinAmerica_train/
  flac/                           # Training audio files (FLAC format)
  ASVspoof_LatinAmerica.cm.train.txt  # Training protocol

ASVspoof_LatinAmerica_dev/
  flac/                           # Validation audio files
  ASVspoof_LatinAmerica.cm.dev.txt    # Validation protocol

ASVspoof_LatinAmerica_eval/
  flac/                           # Test audio files
  ASVspoof_LatinAmerica.cm.eval.txt   # Test protocol

PROTOCOL FILE FORMAT
{'='*70}
speaker_id audio_file augmentation_type key

Example:
arf_00295 LA_T_0000001 ORIGINAL bonafide
arf_00295 LA_T_0000002 RIR_MEDIUM_NOI_SNR15 bonafide
clm_12345 LA_T_0000003 CODEC_8K_LOSS2PCT bonafide

AUGMENTATION TYPES
{'='*70}
RIR + Noise:
  - Room sizes: small (30%), medium (50%), large (20%)
  - SNR distribution: 0-5dB (10%), 5-30dB (80%), 30-35dB (10%)
  - Noise sources: noise (50%), speech (30%), music (20%)

Codec/Channel:
  - Downsampling to 8kHz/16kHz
  - Packet loss simulation (1-5%)
  - Bandpass filtering (300-3400 Hz)

RawBoost:
  - Linear filtering
  - Nonlinear distortion
  - Signal-dependent additive noise
  - Clipping simulation

CITATION
{'='*70}
If you use this dataset, please cite:
[Your citation here]

Generated using augmentation pipeline based on:
- Sánchez et al. (2024) - RIR and noise augmentation
- ASVspoof 2019 - Dataset structure and protocols
- Tak et al. (2022) - RawBoost methodology
"""
        
        with open(readme_path, 'w') as f:
            f.write(content)
        
        print(f"\nREADME created: {readme_path}")
    
    def run(self):
        """
        Execute complete augmentation pipeline.
        
        Process:
        1. Create output directory structure
        2. Process and augment training files
        3. Copy validation files unchanged
        4. Copy test files unchanged
        5. Generate protocol files
        6. Create README
        """
        print("\n" + "="*70)
        print(f"AUGMENTATION PIPELINE - {self.augmentation_factor}")
        print("="*70)
        
        self.config_manager.print_strategy_summary(self.augmentation_factor)
        
        self._create_output_structure()
        
        train_protocol = self._process_train_files()
        self._save_protocol_file(train_protocol, "train")
        
        val_protocol = self._copy_split_unchanged("val", "dev")
        self._save_protocol_file(val_protocol, "dev")
        
        test_protocol = self._copy_split_unchanged("test", "eval")
        self._save_protocol_file(test_protocol, "eval")
        
        self._create_readme()
        
        print("\n" + "="*70)
        print("AUGMENTATION PIPELINE COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {self.output_root / self.augmentation_factor}")
        print(f"Total files generated: {len(train_protocol) + len(val_protocol) + len(test_protocol):,}")
        print("\nDataset ready for training!")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Voice Anti-Spoofing Data Augmentation Pipeline"
    )
    
    parser.add_argument(
        "--factor",
        type=str,
        default="3x",
        choices=["3x", "5x", "10x"],
        help="Augmentation factor"
    )
    
    parser.add_argument(
        "--voices",
        type=str,
        default="data/partition_dataset_by_speaker",
        help="Path to original voice files"
    )
    
    parser.add_argument(
        "--musan",
        type=str,
        default="data/noise_dataset/musan",
        help="Path to MUSAN dataset"
    )
    
    parser.add_argument(
        "--rir",
        type=str,
        default="data/noise_dataset/RIR",
        help="Path to RIR files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/augmented",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    pipeline = AugmentationPipeline(
        augmentation_factor=args.factor,
        voices_root=args.voices,
        musan_root=args.musan,
        rir_root=args.rir,
        output_root=args.output
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()