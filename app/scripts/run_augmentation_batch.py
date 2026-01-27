#!/usr/bin/env python3
"""
Batch Augmentation Runner

Runs the augmentation pipeline sequentially for multiple min_factor values.
Each run produces its own output directory and log file ({min_factor}.txt).

Usage:
    python -m app.scripts.run_augmentation_batch
    python -m app.scripts.run_augmentation_batch --target_ratio 0.60
    python -m app.scripts.run_augmentation_batch --factors 2x 3x 5x
"""

import sys
import argparse
import time
from app.scripts.augmentation_pipeline import AugmentationPipeline

DEFAULT_FACTORS = ["2x", "3x", "5x", "10x"]


def run_batch(
    factors: list,
    target_ratio: float = 0.50,
    voices: str = "data/partition_dataset_by_speaker",
    musan: str = "data/noise_dataset/musan",
    rir: str = "data/noise_dataset/RIR",
    output: str = "data/augmented",
    seed: int = 42,
):
    """
    Run augmentation pipeline for each factor sequentially.

    Args:
        factors: List of augmentation factors (e.g. ["2x", "3x", "5x", "10x"]).
        target_ratio: Target bonafide ratio.
        voices: Path to speaker-partitioned dataset.
        musan: Path to MUSAN noise dataset.
        rir: Path to RIR files.
        output: Output root directory.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping each factor to its result status ("success" or error message).
    """
    results = {}

    print("\n" + "=" * 70)
    print("BATCH AUGMENTATION RUNNER")
    print("=" * 70)
    print(f"  Factors to run: {factors}")
    print(f"  Target ratio:   {target_ratio}")
    print(f"  Seed:           {seed}")
    print("=" * 70)

    for i, factor in enumerate(factors, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(factors)}] Starting augmentation with min_factor={factor}")
        print(f"{'#' * 70}")

        start_time = time.time()

        try:
            pipeline = AugmentationPipeline(
                voices_root=voices,
                musan_root=musan,
                rir_root=rir,
                output_root=output,
                target_ratio=target_ratio,
                min_factor=factor,
                seed=seed,
            )

            logger = pipeline.logger
            bonafide_pct = int(target_ratio * 100)
            spoof_pct = 100 - bonafide_pct

            logger.info("\n" + "=" * 70)
            logger.info("ANTI-SPOOFING DATA AUGMENTATION - BALANCED MODE")
            logger.info("=" * 70)
            logger.info(f"\nRun Configuration:")
            logger.info(f"  Target ratio: {bonafide_pct}/{spoof_pct} (bonafide/spoof)")
            logger.info(f"  Min factor:   {factor}")
            logger.info(f"  Voices:       {voices}")
            logger.info(f"  MUSAN:        {musan}")
            logger.info(f"  RIR:          {rir}")
            logger.info(f"  Output:       {output}")
            logger.info(f"  Seed:         {seed}")
            logger.info("")

            pipeline.run()

            elapsed = time.time() - start_time
            logger.info("\n" + "=" * 70)
            logger.info(f"SUCCESS: Augmentation with {factor} completed!")
            logger.info(f"  Elapsed: {elapsed:.1f}s")
            logger.info("=" * 70)
            logger.info(f"\nOutput directory: {pipeline.output_dir}")
            logger.info(f"Log saved to: {pipeline.log_path}")
            logger.info("")

            results[factor] = "success"
            print(f"\n  [{factor}] completed in {elapsed:.1f}s -> {pipeline.log_path}")

        except Exception as e:
            elapsed = time.time() - start_time
            results[factor] = str(e)
            print(f"\n  [{factor}] FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    for factor, status in results.items():
        marker = "OK" if status == "success" else "FAIL"
        print(f"  [{marker}] {factor}: {status}")
    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch augmentation runner for multiple min_factor values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Default (2x, 3x, 5x, 10x):
    python -m app.scripts.run_augmentation_batch

  Custom factors:
    python -m app.scripts.run_augmentation_batch --factors 2x 3x

  With different ratio:
    python -m app.scripts.run_augmentation_batch --target_ratio 0.60
        """,
    )

    parser.add_argument(
        "--factors",
        nargs="+",
        default=DEFAULT_FACTORS,
        help=f"List of min_factor values to run (default: {DEFAULT_FACTORS})",
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.50,
        help="Target bonafide ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--voices",
        type=str,
        default="data/partition_dataset_by_speaker",
        help="Path to speaker-independent partitioned dataset",
    )
    parser.add_argument(
        "--musan",
        type=str,
        default="data/noise_dataset/musan",
        help="Path to MUSAN noise dataset",
    )
    parser.add_argument(
        "--rir",
        type=str,
        default="data/noise_dataset/RIR",
        help="Path to RIR files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/augmented",
        help="Output root directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if not (0.0 < args.target_ratio < 1.0):
        print(f"ERROR: target_ratio must be between 0.0 and 1.0, got {args.target_ratio}")
        sys.exit(1)

    for f in args.factors:
        if not f.endswith("x"):
            print(f"ERROR: Invalid factor format '{f}'. Expected format: '2x', '3x', etc.")
            sys.exit(1)
        try:
            int(f[:-1])
        except ValueError:
            print(f"ERROR: Invalid factor format '{f}'. Expected format: '2x', '3x', etc.")
            sys.exit(1)

    results = run_batch(
        factors=args.factors,
        target_ratio=args.target_ratio,
        voices=args.voices,
        musan=args.musan,
        rir=args.rir,
        output=args.output,
        seed=args.seed,
    )

    if any(v != "success" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
