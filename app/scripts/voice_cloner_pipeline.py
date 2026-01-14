"""
Complete Voice Cloning Training Pipeline

This script orchestrates the entire training pipeline from data transcription
to model training, with comprehensive validation and checkpoint management.

Directory Structure:
    data/
    ├── partition_dataset_by_speaker/
    │   └── speaker_id/
    │       ├── train/*.wav
    │       ├── val/*.wav
    │       └── test/*.wav
    └── embeddings/
        ├── metadata.csv
        └── speaker_embeddings.pt

Usage:
    python voice_cloner_pipeline.py --all
    python voice_cloner_pipeline.py --step 1
    python voice_cloner_pipeline.py --step 2
    python voice_cloner_pipeline.py --step 3
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from app.cloner.transcriber import DatasetTranscriber
from app.cloner.speaker_encoder import SpeakerEncoder
from app.cloner.train_voice_cloner import VoiceClonerTrainer
from app.cloner.pipeline_validator import PipelineValidator
import pandas as pd
import torch


class VoiceCloningPipeline:
    """
    Orchestrates the complete voice cloning training pipeline.
    
    Manages execution order, validation, checkpoint detection, and error
    handling for all pipeline stages.
    
    Attributes:
        validator: PipelineValidator instance for data verification.
        embeddings_dir: Directory for storing metadata and embeddings.
        dataset_root: Root directory for audio dataset.
    """
    
    def __init__(self):
        self.validator = PipelineValidator()
        self.embeddings_dir = Path("app/embedding")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_root = Path("data/partition_dataset_by_speaker")
    
    def _print_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "="*70)
        print(title.center(70))
        print("="*70 + "\n")
    
    def _print_status(self, message: str, status: str = "info"):
        """
        Print formatted status message.
        
        Args:
            message: Status message to display.
            status: Message type (info, success, warning, error).
        """
        symbols = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗"
        }
        symbol = symbols.get(status, "•")
        print(f"{symbol} {message}")
    
    def validate_prerequisites(self) -> bool:
        """
        Validate all pipeline prerequisites before execution.
        
        Returns:
            True if all validations pass, False otherwise.
        """
        self._print_header("PREREQUISITE VALIDATION")
        
        valid, error = self.validator.validate_dataset_structure()
        if not valid:
            self._print_status(f"Dataset validation failed: {error}", "error")
            return False
        self._print_status("Dataset structure valid", "success")
        
        return True
    
    def step_1_transcription(self, force: bool = False) -> bool:
        """
        Execute transcription step to generate metadata.csv.
        
        Args:
            force: If True, regenerate even if metadata exists.
            
        Returns:
            True if transcription completed successfully, False otherwise.
        """
        self._print_header("STEP 1: AUDIO TRANSCRIPTION")
        
        metadata_path = self.embeddings_dir / "metadata.csv"
        
        if metadata_path.exists() and not force:
            self._print_status(f"Metadata already exists: {metadata_path}", "info")
            response = input("Regenerate? (y/N): ").strip().lower()
            if response != 'y':
                self._print_status("Using existing metadata", "info")
                return True
        
        self._print_status("Starting transcription with Whisper...", "info")
        self._print_status("This may take 10-60 minutes depending on dataset size", "info")
        
        try:
            transcriber = DatasetTranscriber(model_size="tiny")
            
            transcriber.transcribe_dataset(
                root_path=str(self.dataset_root),
                output_csv=str(metadata_path),
                save_interval=500
            )
            
            valid, error = self.validator.validate_metadata()
            if not valid:
                self._print_status(f"Metadata validation failed: {error}", "error")
                return False
            
            self._print_status(f"Transcription complete: {metadata_path}", "success")
            return True
            
        except Exception as e:
            self._print_status(f"Transcription failed: {e}", "error")
            return False
        except KeyboardInterrupt:
            self._print_status("Transcription interrupted", "warning")
            if metadata_path.exists():
                self._print_status("Partial results saved, you can resume", "info")
            return False
    
    def step_2_embeddings(self, force: bool = False) -> bool:
        """
        Execute embedding generation step to create speaker_embeddings.pt.
        
        Args:
            force: If True, regenerate even if embeddings exist.
            
        Returns:
            True if embedding generation completed successfully, False otherwise.
        """
        self._print_header("STEP 2: SPEAKER EMBEDDING GENERATION")
        
        embeddings_path = self.embeddings_dir / "speaker_embeddings.pt"
        
        if embeddings_path.exists() and not force:
            self._print_status(f"Embeddings already exist: {embeddings_path}", "info")
            response = input("Regenerate? (y/N): ").strip().lower()
            if response != 'y':
                self._print_status("Using existing embeddings", "info")
                return True
        
        self._print_status("Generating embeddings with WavLM...", "info")
        self._print_status("This may take 10-30 minutes depending on speaker count", "info")
        
        try:
            encoder = SpeakerEncoder()
            
            embeddings = encoder.process_dataset(
                root_path=str(self.dataset_root),
                checkpoint_file=str(embeddings_path),
                save_interval=5,
                use_train_only=True
            )
            
            valid, error = self.validator.validate_embeddings()
            if not valid:
                self._print_status(f"Embeddings validation failed: {error}", "error")
                return False
            
            self._print_status(f"Embedding generation complete: {embeddings_path}", "success")
            return True
            
        except Exception as e:
            self._print_status(f"Embedding generation failed: {e}", "error")
            return False
        except KeyboardInterrupt:
            self._print_status("Embedding generation interrupted", "warning")
            if embeddings_path.exists():
                self._print_status("Partial results saved, you can resume", "info")
            return False
    
    def step_3_training(self) -> bool:
        """
        Execute model training step.
        
        Returns:
            True if training completed successfully, False otherwise.
        """
        self._print_header("STEP 3: MODEL TRAINING")
        
        valid, error = self.validator.validate_metadata()
        if not valid:
            self._print_status(f"Metadata validation failed: {error}", "error")
            self._print_status("Run step 1 (transcription) first", "error")
            return False
        
        valid, error = self.validator.validate_embeddings()
        if not valid:
            self._print_status(f"Embeddings validation failed: {error}", "error")
            self._print_status("Run step 2 (embedding generation) first", "error")
            return False
        
        self._print_status("Starting model training...", "info")
        self._print_status("This may take hours or days depending on configuration", "info")
        self._print_status("Monitor progress: tensorboard --logdir speech_tts_finetuned", "info")
        
        try:
            metadata_path = str(self.embeddings_dir / "metadata.csv")
            embeddings_path = str(self.embeddings_dir / "speaker_embeddings.pt")
            output_dir = "speech_tts_finetuned"
            
            trainer = VoiceClonerTrainer(
                metadata_path=metadata_path,
                embeddings_path=embeddings_path,
                output_dir=output_dir,
                use_presplit=True
            )
            
            processed_data = trainer.load_and_process_data()
            
            trainer.train(
                dataset=processed_data,
                batch_size=8,
                max_steps=30000,
                learning_rate=1e-5,
                warmup_steps=500,
                save_steps=500,
                eval_steps=500,
                logging_steps=50
            )
            
            self._print_status("Training completed successfully", "success")
            return True
            
        except Exception as e:
            self._print_status(f"Training failed: {e}", "error")
            return False
        except KeyboardInterrupt:
            self._print_status("Training interrupted", "warning")
            self._print_status("Progress saved in checkpoints, you can resume", "info")
            return False
    
    def run_full_pipeline(self, skip_transcription: bool = False, skip_embeddings: bool = False):
        """
        Execute all pipeline steps in sequence with validation.
        
        Args:
            skip_transcription: If True, skip step 1 if metadata exists.
            skip_embeddings: If True, skip step 2 if embeddings exist.
        """
        self._print_header("VOICE CLONING TRAINING PIPELINE")
        
        print("Pipeline stages:")
        print("  1. Audio Transcription (Whisper)")
        print("  2. Speaker Embedding Generation (WavLM)")
        print("  3. Model Training (SpeechT5)")
        print("\nEstimated total time: 1-3 hours (depends on dataset size)")
        
        if not self.validate_prerequisites():
            self._print_status("Prerequisite validation failed, aborting", "error")
            return
        
        if not skip_transcription:
            if not self.step_1_transcription():
                self._print_status("Pipeline aborted at step 1", "error")
                return
        else:
            self._print_status("Skipping transcription (using existing metadata)", "info")
        
        if not skip_embeddings:
            if not self.step_2_embeddings():
                self._print_status("Pipeline aborted at step 2", "error")
                return
        else:
            self._print_status("Skipping embeddings (using existing file)", "info")
        
        if not self.step_3_training():
            self._print_status("Pipeline completed with errors", "warning")
            return
        
        self._print_header("PIPELINE COMPLETED SUCCESSFULLY")
        
        print("Generated files:")
        print(f"  • {self.validator.metadata_file}")
        print(f"  • {self.validator.embeddings_file}")
        print(f"  • {self.validator.output_dir}/")
        print("\nYour model is ready for inference!")


def main():
    """
    Main entry point with argument parsing.
    
    Supports running individual steps or the complete pipeline with
    various configuration options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Voice Cloning Training Pipeline with Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run complete pipeline:
    python voice_cloner_pipeline.py --all
  
  Run only transcription:
    python voice_cloner_pipeline.py --step 1
  
  Run only embeddings:
    python voice_cloner_pipeline.py --step 2
  
  Run only training:
    python voice_cloner_pipeline.py --step 3
  
  Skip transcription (use existing metadata):
    python voice_cloner_pipeline.py --all --skip-transcription
  
  Force regeneration:
    python voice_cloner_pipeline.py --step 1 --force
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all pipeline steps in sequence'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3],
        help='Run specific step (1=transcription, 2=embeddings, 3=training)'
    )
    
    parser.add_argument(
        '--skip-transcription',
        action='store_true',
        help='Skip transcription if metadata exists'
    )
    
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip embeddings if file exists'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if files exist'
    )
    
    args = parser.parse_args()
    
    pipeline = VoiceCloningPipeline()
    
    try:
        if args.all:
            pipeline.run_full_pipeline(
                skip_transcription=args.skip_transcription,
                skip_embeddings=args.skip_embeddings
            )
        elif args.step == 1:
            if pipeline.validate_prerequisites():
                pipeline.step_1_transcription(force=args.force)
        elif args.step == 2:
            if pipeline.validate_prerequisites():
                pipeline.step_2_embeddings(force=args.force)
        elif args.step == 3:
            if pipeline.validate_prerequisites():
                pipeline.step_3_training()
        else:
            parser.print_help()
            print("\nTIP: Use --all to run the complete pipeline")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()