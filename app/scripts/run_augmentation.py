#!/usr/bin/env python3
"""
Run Augmentation Pipeline - Uniform Factor Only

State-of-the-art augmentation for anti-spoofing with:
- Speaker-independent splits
- Uniform augmentation factor across bonafide/spoof (no corpus-side
  rebalancing; class imbalance is corrected at training time)
- Val/Test 100% clean (no augmentation)

Usage:
    # Uniform 3x factor
    python run_augmentation.py --min_factor 3x

    # Uniform 5x factor
    python run_augmentation.py --min_factor 5x
"""

import sys
import argparse
from app.scripts.augmentation_pipeline import AugmentationPipeline


def main():
    """Execute the augmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Anti-Spoofing Augmentation Pipeline (Uniform Factor)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Uniform 3x factor:
    python run_augmentation.py --min_factor 3x

  Uniform 5x factor:
    python run_augmentation.py --min_factor 5x

  Custom paths:
    python run_augmentation.py --min_factor 3x \\
        --voices data/my_partition \\
        --musan data/my_noise \\
        --rir data/my_rir \\
        --output data/my_output

Strategy:
  - Train: one clean copy per original plus (factor - 1) augmented copies,
    for both classes alike
  - Val:   100% clean (no augmentation) for pure evaluation
  - Test:  100% clean (no augmentation) for final testing
  - Natural class ratio is preserved; rebalancing is left to training time
        """
    )

    # Augmentation factor
    parser.add_argument(
        "--min_factor",
        type=str,
        default="3x",
        help="Augmentation factor applied uniformly to both classes (e.g., 3x, 5x, 10x)"
    )

    # Uniform loudness target
    parser.add_argument(
        "--loudness_dbfs",
        type=float,
        default=-23.0,
        help="Target RMS level (dBFS) applied uniformly to every emitted clip"
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

    try:
        # Create pipeline (logger is set up inside)
        pipeline = AugmentationPipeline(
            voices_root=args.voices,
            musan_root=args.musan,
            rir_root=args.rir,
            output_root=args.output,
            min_factor=args.min_factor,
            loudness_target_dbfs=args.loudness_dbfs,
            seed=args.seed
        )

        # Log run configuration through the pipeline logger
        logger = pipeline.logger

        logger.info("\n" + "="*70)
        logger.info("ANTI-SPOOFING DATA AUGMENTATION - UNIFORM FACTOR")
        logger.info("="*70)
        logger.info(f"\nRun Configuration:")
        logger.info(f"  Min factor:   {args.min_factor}")
        logger.info(f"  Loudness:     {args.loudness_dbfs} dBFS")
        logger.info(f"  Voices:       {args.voices}")
        logger.info(f"  MUSAN:        {args.musan}")
        logger.info(f"  RIR:          {args.rir}")
        logger.info(f"  Output:       {args.output}")
        logger.info(f"  Seed:         {args.seed}")
        logger.info("")

        # Run pipeline
        pipeline.run()

        logger.info("\n" + "="*70)
        logger.info("SUCCESS: Augmentation completed!")
        logger.info("="*70)
        logger.info(f"\nOutput directory: {pipeline.output_dir}")
        logger.info(f"Log saved to: {pipeline.log_path}")
        logger.info("")

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
