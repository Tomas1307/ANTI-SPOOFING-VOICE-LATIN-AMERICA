"""
Ablation: sweep VALLEY_SCORE_THRESHOLD and measure word-selection yield.

Reuses the production WordSelector unmodified against the already-generated
clone audio and forced-alignment metadata from the partial-spoof pipeline
(Steps 1-3 outputs, already on disk from the production run). For each
threshold in the sweep, writes its own throwaway output directory so the
real corpus's word_selection_metadata.json is never touched.

Usage:
    python3 app/scripts/ablation_valley_threshold.py <production_output_dir>

    <production_output_dir> is the attack system's partial-spoof output
    directory that already contains alignment_metadata.json from Steps 1-3,
    e.g. data/partial_spoof_output/omnivoice/canonical
"""
import json
import shutil
import sys
from pathlib import Path

from loguru import logger

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.steps.step_04_select_words import WordSelector

THRESHOLDS = [0.3, 0.5, 0.65, 0.8, 1.0]


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    production_output_dir = Path(sys.argv[1])
    system_name = production_output_dir.parent.name if production_output_dir.name in (
        "canonical", "jittered"
    ) else production_output_dir.name
    ablation_dir = Path("data/ablation_valley_threshold") / system_name

    alignment_src = production_output_dir / "alignment_metadata.json"
    if not alignment_src.exists():
        raise FileNotFoundError(
            f"{alignment_src} not found. This ablation reuses the production "
            "Step 1-3 outputs (transcription, cloning, forced alignment); "
            "run the partial-spoof pipeline through Step 3 first."
        )

    ablation_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for threshold in THRESHOLDS:
        run_dir = ablation_dir / f"threshold_{threshold:.2f}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # WordSelector reads alignment_metadata.json from output_dir; copy
        # (not symlink, for portability) the already-computed alignment once.
        run_alignment = run_dir / "alignment_metadata.json"
        if not run_alignment.exists():
            shutil.copy(alignment_src, run_alignment)

        settings.VALLEY_SCORE_THRESHOLD = threshold
        logger.info(f"=== Threshold {threshold:.2f} ===")

        selector = WordSelector(output_dir=run_dir)
        result = selector.execute()

        with open(run_alignment, "r", encoding="utf-8") as f:
            n_utterances = len(json.load(f))

        row = {
            "threshold": threshold,
            "total_selection_plans": result.total_selections,
            "tier_counts": result.tier_counts,
            "n_utterances_aligned": n_utterances,
        }
        results.append(row)
        logger.info(f"Threshold {threshold:.2f}: {row}")

    summary_path = ablation_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Valley score threshold ablation summary ===")
    print(f"{'Threshold':>10} | {'Total plans':>12} | Tier counts")
    for row in results:
        print(f"{row['threshold']:>10.2f} | {row['total_selection_plans']:>12} | {row['tier_counts']}")
    print(f"\nFull results written to {summary_path}")


if __name__ == "__main__":
    main()
