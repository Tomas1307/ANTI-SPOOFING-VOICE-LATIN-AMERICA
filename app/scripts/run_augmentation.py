#!/usr/bin/env python3
"""
Run Augmentation Pipeline

Main execution script for voice anti-spoofing data augmentation.
Supports both simple and balanced augmentation modes.

Usage:
    # Simple mode (uniform 3x augmentation)
    python run_augmentation.py --mode simple --factor 3x
    
    # Balanced mode (50/50 ratio with minimum 3x)
    python run_augmentation.py --mode balanced --factor 3x --target_ratio 0.50
    
    # Balanced mode (60/40 ratio with minimum 5x)
    python run_augmentation.py --mode balanced --factor 5x --target_ratio 0.60
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
  Simple mode (uniform 3x augmentation for all files):
    python run_augmentation.py --mode simple --factor 3x
  
  Simple mode with 5x:
    python run_augmentation.py --mode simple --factor 5x
  
  Balanced mode (50/50 bonafide/spoof with minimum 3x total):
    python run_augmentation.py --mode balanced --factor 3x --target_ratio 0.50
  
  Balanced mode (60/40 with minimum 5x):
    python run_augmentation.py --mode balanced --factor 5x --target_ratio 0.60
  
  Balanced mode (70/30 with minimum 3x):
    python run_augmentation.py --mode balanced --factor 3x --target_ratio 0.70
  
  Custom paths:
    python run_augmentation.py --mode simple --factor 3x \\
        --voices data/my_voices \\
        --musan data/my_noise \\
        --rir data/my_rir \\
        --output data/my_augmented
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="simple",
        choices=["simple", "balanced"],
        help="Augmentation mode: 'simple' (uniform factor) or 'balanced' (calculated factors)"
    )
    
    # Augmentation factor
    parser.add_argument(
        "--factor",
        type=str,
        default="3x",
        help="Augmentation factor (e.g., 3x, 5x, 7x, 10x). In simple mode, applied uniformly. In balanced mode, used as minimum total factor."
    )
    
    # Target ratio (for balanced mode)
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.50,
        help="Target bonafide ratio for balanced mode (0.0-1.0). E.g., 0.50 = 50/50, 0.60 = 60/40, 0.70 = 70/30"
    )
    
    # Data paths
    parser.add_argument(
        "--voices",
        type=str,
        default="data/partition_dataset_by_speaker",
        help="Path to partitioned voice dataset"
    )
    
    parser.add_argument(
        "--musan",
        type=str,
        default="data/noise_dataset/musan",
        help="Path to MUSAN noise dataset"
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
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Validate target_ratio
    if args.mode == "balanced" and not (0.0 < args.target_ratio < 1.0):
        print(f"ERROR: target_ratio must be between 0.0 and 1.0, got {args.target_ratio}")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*70)
    print("VOICE ANTI-SPOOFING DATA AUGMENTATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Mode:         {args.mode}")
    print(f"  Factor:       {args.factor}")
    
    if args.mode == "balanced":
        bonafide_pct = int(args.target_ratio * 100)
        spoof_pct = 100 - bonafide_pct
        print(f"  Target ratio: {bonafide_pct}/{spoof_pct} (bonafide/spoof)")
    
    print(f"  Voices:       {args.voices}")
    print(f"  MUSAN:        {args.musan}")
    print(f"  RIR:          {args.rir}")
    print(f"  Output:       {args.output}")
    print(f"  Seed:         {args.seed}")
    print()
    
    try:
        # Create pipeline
        pipeline = AugmentationPipeline(
            voices_root=args.voices,
            musan_root=args.musan,
            rir_root=args.rir,
            output_root=args.output,
            mode=args.mode,
            factor=args.factor,
            target_ratio=args.target_ratio,
            seed=args.seed
        )
        
        # Run pipeline
        pipeline.run()
        
        print("\n" + "="*70)
        print("SUCCESS: Augmentation completed!")
        print("="*70)
        print(f"\nOutput directory: {pipeline.output_dir}")
        print("\nNext steps:")
        print("  1. Verify output structure:")
        print(f"     ls {pipeline.output_dir}/LA/")
        print("  2. Check protocol files:")
        print(f"     head {pipeline.output_dir}/LA/ASVspoof2019_LA_train/*.txt")
        print("  3. Count generated files:")
        print(f"     find {pipeline.output_dir} -name '*.flac' | wc -l")
        print()
        
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