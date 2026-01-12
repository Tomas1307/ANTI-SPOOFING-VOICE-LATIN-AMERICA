#!/usr/bin/env python3
"""
Run Augmentation Pipeline

Main execution script for voice anti-spoofing data augmentation.

Usage:
    python run_augmentation.py --factor 3x
    python run_augmentation.py --factor 7x --output data/augmented_custom
"""

import sys
import argparse
from pathlib import Path
from app.scripts.augmentation_pipeline import AugmentationPipeline

def main():
    """Execute augmentation pipeline with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Anti-Spoofing Data Augmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate 3x augmented dataset:
    python run_augmentation.py --factor 3x
  
  Generate 5x augmented dataset:
    python run_augmentation.py --factor 5x
  
  Generate 7x augmented dataset (custom):
    python run_augmentation.py --factor 7x
  
  Custom output directory:
    python run_augmentation.py --factor 3x --output data/my_augmented
  
  Custom paths:
    python run_augmentation.py --factor 3x \\
        --voices data/my_voices \\
        --musan data/my_noise \\
        --rir data/my_rir
        """
    )
    
    parser.add_argument(
        "--factor",
        type=str,
        default="3x",
        help="Augmentation factor (e.g., 3x, 5x, 7x, 10x, 15x, etc.)"
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
        help="Output root directory"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VOICE ANTI-SPOOFING DATA AUGMENTATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Augmentation factor: {args.factor}")
    print(f"  Voices: {args.voices}")
    print(f"  MUSAN: {args.musan}")
    print(f"  RIR: {args.rir}")
    print(f"  Output: {args.output}")
    print()
    
    try:
        pipeline = AugmentationPipeline(
            augmentation_factor=args.factor,
            voices_root=args.voices,
            musan_root=args.musan,
            rir_root=args.rir,
            output_root=args.output
        )
        
        pipeline.run()
        
        print("\n" + "="*70)
        print("SUCCESS: Augmentation completed!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nAugmentation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()