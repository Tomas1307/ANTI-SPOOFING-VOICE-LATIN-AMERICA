"""
Integrate selected Common Voice samples into HABLA dataset structure.

This script creates bonafide_dataset_by_speaker_v2/ by:
1. Copying all existing HABLA speakers
2. Adding CV speakers with proper ID generation and folder structure
3. Splitting CV samples into train/val/test per speaker

Input:
    data/mozilla_speaker_selection/selected_15340.tsv
    data/mozilla_speaker_selection/cv_speaker_metadata.json
    data/bonafide_dataset_by_speaker/ (existing HABLA)
    data/cv-corpus-24.0-2025-12-05/es/clips/ (CV audio files)

Output:
    data/bonafide_dataset_by_speaker_v2/ (HABLA + CV merged)
    data/mozilla_speaker_selection/cv_speaker_mapping.json (ID mapping)

Usage:
    python -m app.pipeline.select_mozilla_speakers.05_integrate_cv_samples
"""
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


# Input paths
SELECTED_TSV = Path("data/mozilla_speaker_selection/selected_15340.tsv")
CV_METADATA = Path("data/mozilla_speaker_selection/cv_speaker_metadata.json")
HABLA_DIR = Path("data/bonafide_dataset_by_speaker")
CV_CLIPS_DIR = Path("data/cv-corpus-24.0-2025-12-05/es/clips")

# Output paths
OUTPUT_DIR = Path("data/bonafide_dataset_by_speaker_v2")
MAPPING_OUTPUT = Path("data/mozilla_speaker_selection/cv_speaker_mapping.json")

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Accent codes for speaker ID generation
ACCENT_CODES = {
    "Colombia": "co",
    "Chile": "cl",
    "Venezuela": "ve",
    "Mexico": "mx",
    "Spain": "es",
}

# Gender codes
GENDER_CODES = {
    "male": "m",
    "female": "f",
    "other": "o",
}


def get_gender_code(gender: str) -> str:
    """Extract gender code from gender string."""
    gender_lower = gender.lower()
    if "female" in gender_lower:
        return "f"
    elif "male" in gender_lower:
        return "m"
    else:
        return "o"


def count_existing_habla_speakers() -> Dict[str, int]:
    """Count existing HABLA speakers per accent to continue numbering.

    Returns:
        Dict mapping accent code to max speaker number
        Example: {"ar": 20, "cl": 29, "co": 49, "pe": 18, "ve": 25}
    """
    speaker_counts = defaultdict(int)

    for speaker_dir in HABLA_DIR.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            # Extract accent code (first 2 chars before gender, e.g., "ar" from "arf_00001")
            accent_code = speaker_id[:2]
            # Extract number (last 5 digits)
            try:
                num = int(speaker_id.split("_")[1])
                speaker_counts[accent_code] = max(speaker_counts[accent_code], num)
            except (IndexError, ValueError):
                continue

    return dict(speaker_counts)


def generate_speaker_id(
    accent: str,
    gender: str,
    counter: Dict[str, Dict[str, int]]
) -> str:
    """Generate new speaker ID for CV speaker.

    Args:
        accent: Accent name (e.g., "Mexico")
        gender: Gender string (e.g., "male_masculine")
        counter: Nested dict tracking current count per accent+gender

    Returns:
        Speaker ID like "mxm_00001"
    """
    accent_code = ACCENT_CODES[accent]
    gender_code = get_gender_code(gender)
    key = f"{accent_code}{gender_code}"

    counter[accent_code][gender_code] += 1
    num = counter[accent_code][gender_code]

    return f"{key}_{num:05d}"


def split_samples(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Split samples into train/val/test.

    Args:
        samples: List of sample dicts

    Returns:
        Dict with keys "train", "val", "test"
    """
    n = len(samples)

    # Ensure at least 1 sample in each split if possible
    if n < 3:
        return {"train": samples, "val": [], "test": []}

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    # Ensure at least 1 in each if enough samples
    if train_end == 0:
        train_end = 1
    if val_end <= train_end:
        val_end = train_end + 1

    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def copy_habla_speakers():
    """Copy all existing HABLA speakers to v2 directory."""
    print("=" * 70)
    print("Step 1: Copy Existing HABLA Speakers")
    print("=" * 70)
    print()

    habla_speakers = sorted([d for d in HABLA_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(habla_speakers)} HABLA speakers")
    print("Copying to bonafide_dataset_by_speaker_v2/...")
    print()

    for speaker_dir in tqdm(habla_speakers, desc="Copying HABLA speakers"):
        dest_dir = OUTPUT_DIR / speaker_dir.name
        shutil.copytree(speaker_dir, dest_dir, dirs_exist_ok=True)

    print(f"✓ Copied {len(habla_speakers)} HABLA speakers")
    print()


def integrate_cv_speakers():
    """Add CV speakers to v2 directory."""
    print("=" * 70)
    print("Step 2: Integrate Common Voice Speakers")
    print("=" * 70)
    print()

    # Load metadata
    print("Loading CV metadata...")
    with open(CV_METADATA, "r", encoding="utf-8") as f:
        cv_metadata = json.load(f)

    # Load selected samples
    print(f"Loading selected samples from {SELECTED_TSV}...")
    samples_by_speaker = defaultdict(list)

    with open(SELECTED_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            client_id = row["client_id"]
            samples_by_speaker[client_id].append(row)

    print(f"Loaded {len(samples_by_speaker):,} CV speakers")
    print(f"Total samples: {sum(len(s) for s in samples_by_speaker.values()):,}")
    print()

    # Count existing HABLA speakers to continue numbering
    existing_counts = count_existing_habla_speakers()
    print("Existing HABLA speaker counts per accent:")
    for accent_code, count in sorted(existing_counts.items()):
        print(f"  {accent_code}: {count}")
    print()

    # Initialize counters for CV speaker ID generation
    cv_counters = defaultdict(lambda: defaultdict(int))
    for accent_code, count in existing_counts.items():
        cv_counters[accent_code]["m"] = count
        cv_counters[accent_code]["f"] = count
        cv_counters[accent_code]["o"] = count

    # Process each CV speaker
    speaker_mapping = {}
    accent_stats = Counter()
    gender_stats = Counter()

    print("Creating speaker directories and copying audio files...")
    print()

    for client_id, samples in tqdm(samples_by_speaker.items(), desc="Processing CV speakers"):
        # Get metadata
        metadata = cv_metadata.get(client_id)
        if not metadata:
            print(f"Warning: No metadata for {client_id}, skipping")
            continue

        accent = metadata["accent_category"]
        gender = metadata["gender"]

        # Generate speaker ID
        speaker_id = generate_speaker_id(accent, gender, cv_counters)
        speaker_mapping[client_id] = {
            "speaker_id": speaker_id,
            "accent": accent,
            "gender": gender,
            "sample_count": len(samples),
        }

        accent_stats[accent] += 1
        gender_stats[get_gender_code(gender)] += 1

        # Create speaker directory
        speaker_dir = OUTPUT_DIR / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # Split samples
        splits = split_samples(samples)

        # Copy audio files to respective splits
        for split_name, samples_in_split in splits.items():
            if not samples_in_split:
                continue

            split_dir = speaker_dir / split_name
            split_dir.mkdir(exist_ok=True)

            for sample in samples_in_split:
                audio_filename = sample["path"]
                src_path = CV_CLIPS_DIR / audio_filename

                if not src_path.exists():
                    continue

                dest_path = split_dir / audio_filename
                shutil.copy2(src_path, dest_path)

    print()
    print("=" * 70)
    print("Integration Complete")
    print("=" * 70)
    print(f"Total CV speakers added: {len(speaker_mapping):,}")
    print()
    print("CV speakers by accent:")
    for accent, count in sorted(accent_stats.items(), key=lambda x: -x[1]):
        print(f"  {accent:12s}: {count:,}")
    print()
    print("CV speakers by gender:")
    for gender, count in sorted(gender_stats.items(), key=lambda x: -x[1]):
        gender_name = {"m": "Male", "f": "Female", "o": "Other"}[gender]
        print(f"  {gender_name:8s}: {count:,}")
    print()

    # Save mapping
    print(f"Saving speaker mapping to {MAPPING_OUTPUT}...")
    with open(MAPPING_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(speaker_mapping, f, indent=2, ensure_ascii=False)

    print("✓ Done!")
    print()

    return speaker_mapping


def main():
    """Main integration pipeline."""
    print("=" * 70)
    print("Common Voice Integration Pipeline")
    print("=" * 70)
    print()
    print(f"Creating {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print()

    # Step 1: Copy HABLA speakers
    copy_habla_speakers()

    # Step 2: Add CV speakers
    speaker_mapping = integrate_cv_speakers()

    # Final summary
    total_speakers = len(list(OUTPUT_DIR.iterdir()))
    habla_speakers = len(list(HABLA_DIR.iterdir()))
    cv_speakers = len(speaker_mapping)

    print("=" * 70)
    print("Final Summary")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total speakers: {total_speakers:,}")
    print(f"  HABLA: {habla_speakers:,}")
    print(f"  CV:    {cv_speakers:,}")
    print()
    print("Next steps:")
    print("  1. Verify dataset structure: ls data/bonafide_dataset_by_speaker_v2/ | head")
    print("  2. Update config to use bonafide_dataset_by_speaker_v2/")
    print("  3. Regenerate partitions if needed")
    print()


if __name__ == "__main__":
    main()
