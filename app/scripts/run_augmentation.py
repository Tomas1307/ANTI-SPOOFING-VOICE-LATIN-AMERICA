#!/usr/bin/env python3
"""
Run Augmentation Pipeline - Balanced Mode Only

State-of-the-art augmentation for anti-spoofing with:
- Speaker-independent splits
- Balanced bonafide/spoof ratios
- 25% clean data preservation in train
- Val/Test 100% clean (no augmentation)

Usage:
    # 50/50 balance with 3x minimum
    python run_augmentation.py --target_ratio 0.50 --min_factor 3x
    
    # 60/40 balance (more bonafide) with 5x minimum
    python run_augmentation.py --target_ratio 0.60 --min_factor 5x
"""

import sys
import argparse
from pathlib import Path
from app.scripts.augmentation_pipeline import AugmentationPipeline


def main():
    """Execute balanced augmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Anti-Spoofing Augmentation Pipeline (Balanced Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  50/50 balance with 3x minimum:
    python run_augmentation.py --target_ratio 0.50 --min_factor 3x
  
  60/40 balance (more bonafide) with 5x minimum:
    python run_augmentation.py --target_ratio 0.60 --min_factor 5x
  
  70/30 balance with 3x minimum:
    python run_augmentation.py --target_ratio 0.70 --min_factor 3x
  
  Custom paths:
    python run_augmentation.py --target_ratio 0.50 --min_factor 3x \\
        --voices data/my_partition \\
        --musan data/my_noise \\
        --rir data/my_rir \\
        --output data/my_output

Strategy:
  - Train: Augmented with calculated factors to achieve target ratio
  - Val:   100% clean (no augmentation) for pure evaluation
  - Test:  100% clean (no augmentation) for final testing
  - Clean data in train: ~25-35% (all originals always included)
        """
    )
    
    # Target ratio
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.50,
        help="Target bonafide ratio (0.0-1.0). Examples: 0.50=50/50, 0.60=60/40, 0.70=70/30"
    )
    
    # Minimum augmentation factor
    parser.add_argument(
        "--min_factor",
        type=str,
        default="3x",
        help="Minimum total augmentation factor (e.g., 3x, 5x, 10x)"
    )
    
    # Data paths
    parser.add_argument(
        "--voices",
        type=str,
        default="data/partition_dataset_by_speaker",
        help="Path to speaker-independent partitioned dataset"
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
    if not (0.0 < args.target_ratio < 1.0):
        print(f"ERROR: target_ratio must be between 0.0 and 1.0, got {args.target_ratio}")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*70)
    print("ANTI-SPOOFING DATA AUGMENTATION - BALANCED MODE")
    print("="*70)
    print(f"\nConfiguration:")
    
    bonafide_pct = int(args.target_ratio * 100)
    spoof_pct = 100 - bonafide_pct
    print(f"  Target ratio: {bonafide_pct}/{spoof_pct} (bonafide/spoof)")
    print(f"  Min factor:   {args.min_factor}")
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
            target_ratio=args.target_ratio,
            min_factor=args.min_factor,
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
        print("  4. Verify clean data ratio in train:")
        print(f"     grep ' - bonafide' {pipeline.output_dir}/LA/ASVspoof2019_LA_train/*.txt | wc -l")
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