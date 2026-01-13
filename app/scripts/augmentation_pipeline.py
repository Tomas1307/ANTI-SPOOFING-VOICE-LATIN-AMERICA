"""
Augmentation Pipeline Orchestrator

Main pipeline that coordinates the entire augmentation process:
- Loads original voice files
- Applies augmentation according to configured distributions
- Saves augmented files in ASVspoof-compatible structure
- Generates protocol files
- Copies validation/test files unchanged
- Supports checkpointing for resuming interrupted runs
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
from app.scripts.checkpoint_manager import CheckpointManager 
import app.utils as utils


class AugmentationPipeline:
    """
    Main augmentation pipeline orchestrator with checkpoint support.
    
    Coordinates the entire augmentation process from loading original files
    to generating augmented datasets with ASVspoof-compatible structure.
    
    Supports resuming from checkpoints if interrupted.
    
    Attributes:
        augmentation_factor: Multiplication factor (3x, 5x, 10x, etc.).
        config_manager: Singleton configuration manager.
        strategy: Current augmentation strategy.
        loader: Dataset loader instance.
        augmenters: Dictionary of augmenter instances.
        output_root: Root output directory.
        checkpoint_manager: Checkpoint manager for resume capability.
    """
    
    def __init__(
        self,
        augmentation_factor: str = "3x",
        voices_root: str = "data/partition_dataset_by_speaker",
        musan_root: str = "data/noise_dataset/musan",
        rir_root: str = "data/noise_dataset/RIR",
        output_root: str = "data/augmented"
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentation_factor: "3x", "5x", "7x", "10x", etc.
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
        
        checkpoint_file = f"checkpoint_{augmentation_factor}.json"
        self.checkpoint_manager = CheckpointManager(checkpoint_file)
        
        print(f"\nAugmentationPipeline initialized:")
        print(f"  Factor: {augmentation_factor}")
        print(f"  Output: {output_root}")
        
        output_dir_str = str(self.output_root / augmentation_factor)
        if self.checkpoint_manager.should_resume(augmentation_factor, output_dir_str):
            self.checkpoint_manager.print_status()
            response = input("\n  Resume from checkpoint? [Y/n]: ").strip().lower()
            self.resume_from_checkpoint = response in ['', 'y', 'yes']
            
            if self.resume_from_checkpoint:
                print("✓ Will resume from checkpoint")
            else:
                print("✓ Starting fresh (checkpoint will be overwritten)")
                self.checkpoint_manager.clear_checkpoint()
        else:
            self.resume_from_checkpoint = False
    
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
            flac_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreated output structure at: {output_dir}")
    
    def _process_train_files(self) -> List[Dict]:
        """
        Process training files with augmentation and checkpoint support.
        
        Returns:
            List of protocol entries.
        """
        all_train_files = self.loader.load_train_files()
        
        if self.resume_from_checkpoint:
            processed_speakers = self.checkpoint_manager.get_processed_speakers()
            train_files = [f for f in all_train_files 
                          if f["speaker_id"] not in processed_speakers]
            audio_id_counter = self.checkpoint_manager.get_last_audio_id() + 1
            print(f"\n✓ Resuming: {len(train_files)} files remaining (skipped {len(processed_speakers)} speakers)")
        else:
            train_files = all_train_files
            audio_id_counter = 1
        
        output_dir = self.output_root / self.augmentation_factor / "ASVspoof_LatinAmerica_train" / "flac"
        
        protocol_entries = []
        factor = int(self.augmentation_factor.replace('x', ''))
        
        processed_count = 0
        total_original_files = len(all_train_files)
        processed_speakers_list = list(self.checkpoint_manager.get_processed_speakers())
        checkpoint_interval = 50  # Save every 50 files
        
        print(f"\n{'='*70}")
        print(f"PROCESSING TRAINING FILES ({len(train_files)} originals)")
        print(f"{'='*70}\n")
        
        for file_info in tqdm(train_files, desc="Augmenting train files"):
            filepath = file_info["filepath"]
            speaker_id = file_info["speaker_id"]
            
            try:
                audio, sr = utils.load_audio(filepath, sr=16000)
            except Exception as e:
                print(f"\nError loading {filepath}: {e}")
                continue
            
            # Save original
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
            
            # Generate augmented copies
            for i in range(factor):
                aug_type = self._sample_augmentation_type()
                
                augmented, metadata = self._augment_audio(audio, sr, aug_type)
                
                audio_id = utils.generate_audio_id(audio_id_counter)
                output_path = output_dir / f"{audio_id}.flac"
                
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
                    audio_id,
                    aug_label,
                    "bonafide"
                )
                protocol_entries.append(entry)
                audio_id_counter += 1
            
            processed_count += 1
            if speaker_id not in processed_speakers_list:
                processed_speakers_list.append(speaker_id)
            
            if processed_count % checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    factor=self.augmentation_factor,
                    total_files=total_original_files,
                    processed_files=len(processed_speakers_list),
                    processed_speakers=processed_speakers_list,
                    last_audio_id=audio_id_counter - 1,
                    output_dir=str(self.output_root / self.augmentation_factor)
                )
        
        self.checkpoint_manager.save_checkpoint(
            factor=self.augmentation_factor,
            total_files=total_original_files,
            processed_files=len(processed_speakers_list),
            processed_speakers=processed_speakers_list,
            last_audio_id=audio_id_counter - 1,
            output_dir=str(self.output_root / self.augmentation_factor)
        )
        
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

Generated with checkpoint support - can resume if interrupted.

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

Validation set: 2,211 (original only)
Test set: 2,399 (original only)

Generated: {output_dir}
"""
        
        with open(readme_path, 'w') as f:
            f.write(content)
        
        print(f"\nREADME created: {readme_path}")
    
    def run(self):
        """
        Execute complete augmentation pipeline with checkpoint support.
        
        Process:
        1. Create output directory structure
        2. Process and augment training files (with checkpointing)
        3. Copy validation files unchanged
        4. Copy test files unchanged
        5. Generate protocol files
        6. Create README
        7. Clear checkpoint on success
        """
        print("\n" + "="*70)
        print(f"AUGMENTATION PIPELINE - {self.augmentation_factor}")
        print("="*70)
        
        self.config_manager.print_strategy_summary(self.augmentation_factor)
        
        self._create_output_structure()
        
        try:
            train_protocol = self._process_train_files()
            self._save_protocol_file(train_protocol, "train")
            
            val_protocol = self._copy_split_unchanged("val", "dev")
            self._save_protocol_file(val_protocol, "dev")
            
            test_protocol = self._copy_split_unchanged("test", "eval")
            self._save_protocol_file(test_protocol, "eval")
            
            self._create_readme()
            
            # 🆕 NUEVO: Clear checkpoint on successful completion
            self.checkpoint_manager.clear_checkpoint()
            
            print("\n" + "="*70)
            print("AUGMENTATION PIPELINE COMPLETE")
            print("="*70)
            print(f"\nOutput directory: {self.output_root / self.augmentation_factor}")
            print(f"Total files generated: {len(train_protocol) + len(val_protocol) + len(test_protocol):,}")
            print("\nDataset ready for training!")
            
        except KeyboardInterrupt:
            print("\n\n  Pipeline interrupted! Checkpoint saved.")
            print("Run again to resume from checkpoint.")
            raise
        except Exception as e:
            print(f"\n\n Error occurred: {e}")
            print("Checkpoint saved. Fix the error and run again to resume.")
            raise