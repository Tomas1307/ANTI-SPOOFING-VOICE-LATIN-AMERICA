"""
Master pipeline orchestrator for Mozilla Common Voice speaker selection.

This script runs all 4 pipeline steps sequentially:
1. Extract HABLA speaker embeddings
2. Extract Common Voice speaker embeddings
3. Filter by cosine similarity
4. Balanced sampling

Usage:
    python pipeline/select_mozilla_speakers/run_pipeline.py
"""
import subprocess
import sys
import time
from pathlib import Path


def print_section(title: str):
    """Print formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def run_step(script_name: str, description: str):
    """Run a pipeline step script.

    Args:
        script_name: Name of the Python script to run
        description: Human-readable description of the step
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)

    print(f"Running: {script_name}")
    print()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path(__file__).parent.parent.parent,
        check=False
    )

    if result.returncode != 0:
        print()
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    """Run the complete speaker selection pipeline."""
    start_time = time.time()

    print_section("Mozilla Common Voice Speaker Selection Pipeline")
    print("This pipeline selects 15,340 acoustically diverse speakers from")
    print("Common Voice to balance HABLA 2.0 across 7 accents.")
    print()
    print("Strategy: Boost existing HABLA accents + add Mexico & Spain")
    print("  Colombia: +846 | Chile: +1,361 | Venezuela: +2,233")
    print("  Mexico: +5,450 | Spain: +5,450")
    print("  Target: 5,450 samples per accent (38,156 total)")
    print()

    # Step 1: Extract HABLA embeddings
    print_section("Step 1/4: Extract HABLA Speaker Embeddings")
    run_step("01_extract_habla_embeddings.py", "HABLA embedding extraction")

    # Step 2: Extract CV embeddings
    print_section("Step 2/4: Extract Common Voice Speaker Embeddings (CO/CL/VE/MX/ES)")
    run_step("02_extract_cv_embeddings.py", "CV embedding extraction")

    # Step 3: Filter by similarity
    print_section("Step 3/4: Filter by Cosine Similarity (threshold: 0.75)")
    run_step("03_filter_by_similarity.py", "Similarity filtering")

    # Step 4: Balanced sampling
    print_section("Step 4/4: Balanced Sampling (15,340 samples)")
    run_step("04_balanced_sampling.py", "Balanced sampling")

    # Complete
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print_section("Pipeline Complete!")
    print(f"Total runtime: {minutes}m {seconds}s")
    print()
    print("Output files:")
    print("  - data/mozilla_speaker_selection/habla_embeddings.npy")
    print("  - data/mozilla_speaker_selection/habla_speaker_ids.json")
    print("  - data/mozilla_speaker_selection/cv_speaker_embeddings.npy")
    print("  - data/mozilla_speaker_selection/cv_speaker_metadata.json")
    print("  - data/mozilla_speaker_selection/filtered_speakers.json")
    print("  - data/mozilla_speaker_selection/selected_15340.tsv")
    print("  - data/mozilla_speaker_selection/selection_stats.json")
    print()
    print("Final HABLA 2.0 Dataset:")
    print("  - 7 accents × 5,450 samples = 38,156 total")
    print("  - HABLA original: 22,816 + CV addition: 15,340")
    print()


if __name__ == "__main__":
    main()
