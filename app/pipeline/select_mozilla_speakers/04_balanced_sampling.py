"""
Perform stratified sampling to select 15,340 Common Voice samples.

This script loads the filtered Common Voice speakers and selects exactly:
- Colombia: 846 samples (boost to 5,450 total)
- Chile: 1,361 samples (boost to 5,450 total)
- Venezuela: 2,233 samples (boost to 5,450 total)
- Mexico: 5,450 samples (new accent)
- Spain: 5,450 samples (new accent, prioritize females)

Total: 15,340 samples to balance HABLA 2.0 across 7 accents.

Input:
    data/mozilla_speaker_selection/filtered_speakers.json
    data/mozilla_speaker_selection/cv_speaker_metadata.json
    data/cv_metadata/validated.tsv

Output:
    data/mozilla_speaker_selection/selected_15340.tsv: Subset of validated.tsv
    data/mozilla_speaker_selection/selection_stats.json: Demographics breakdown

Usage:
    python -m pipeline.select_mozilla_speakers.04_balanced_sampling
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


INPUT_DIR = Path("data/mozilla_speaker_selection")
CV_METADATA_DIR = Path("data/cv-corpus-24.0-2025-12-05/es")

FILTERED_SPEAKERS = INPUT_DIR / "filtered_speakers.json"
CV_SPEAKER_METADATA = INPUT_DIR / "cv_speaker_metadata.json"
VALIDATED_TSV = CV_METADATA_DIR / "validated.tsv"

SELECTED_OUTPUT = INPUT_DIR / "selected_15340.tsv"
STATS_OUTPUT = INPUT_DIR / "selection_stats.json"

# Fixed targets per accent to balance HABLA 2.0 to 5,450 samples each
ACCENT_TARGETS = {
    "Colombia": 846,      # 4,604 + 846 = 5,450
    "Chile": 1361,        # 4,089 + 1,361 = 5,450
    "Venezuela": 2233,    # 3,217 + 2,233 = 5,450
    "Mexico": 5450,       # 0 + 5,450 = 5,450
    "Spain": 5450,        # 0 + 5,450 = 5,450
}

TOTAL_TARGET = sum(ACCENT_TARGETS.values())  # 15,340
RANDOM_SEED = 42


def load_filtered_speakers() -> set:
    """Load filtered speaker IDs."""
    with open(FILTERED_SPEAKERS, "r", encoding="utf-8") as f:
        speakers = json.load(f)
    return set(speakers)


def load_cv_metadata() -> dict:
    """Load Common Voice speaker metadata."""
    with open(CV_SPEAKER_METADATA, "r", encoding="utf-8") as f:
        return json.load(f)


def get_accent_category(sample_dict: dict) -> str:
    """Get accent category from sample metadata.

    Uses the accent_category field added in step 2.
    """
    return sample_dict.get("accent_category", "Unknown")


def normalize_gender(gender: str) -> str:
    """Normalize gender string."""
    if not gender:
        return "Unknown"
    if "male" in gender.lower() and "female" not in gender.lower():
        return "Male"
    elif "female" in gender.lower():
        return "Female"
    else:
        return "Other"


def normalize_age(age: str) -> str:
    """Normalize age string."""
    if not age:
        return "Unknown"
    return age.capitalize()


def stratified_sample(
    samples_by_speaker: Dict[str, List[Dict]],
    cv_metadata: dict
) -> List[Dict]:
    """Perform stratified sampling with fixed targets per accent.

    Uses ACCENT_TARGETS to select exact counts per accent:
    - Colombia: 846
    - Chile: 1,361
    - Venezuela: 2,233
    - Mexico: 5,450
    - Spain: 5,450 (prioritize females)

    Within each accent, balances by gender and age.

    Args:
        samples_by_speaker: Dict mapping client_id to list of samples
        cv_metadata: Speaker metadata dict

    Returns:
        List of selected sample dicts
    """
    # Group samples by accent using cv_metadata (which has accent_category from step 2)
    samples_by_accent = defaultdict(list)

    for client_id, samples in samples_by_speaker.items():
        accent = cv_metadata.get(client_id, {}).get("accent_category", "Unknown")
        if accent in ACCENT_TARGETS:
            for sample in samples:
                sample["accent_category"] = accent
                samples_by_accent[accent].append(sample)

    print(f"Available samples by accent:")
    for accent in ["Colombia", "Chile", "Venezuela", "Mexico", "Spain"]:
        print(f"  {accent:12s}: {len(samples_by_accent[accent]):,} samples available")
    print()

    selected_samples = []

    for accent, target in ACCENT_TARGETS.items():
        available = samples_by_accent[accent]

        if len(available) < target:
            print(f"WARNING: {accent} has only {len(available)} samples, need {target}")
            selected = available  # Take all available
        else:
            # For Spain, prioritize females
            if accent == "Spain":
                print(f"  {accent:12s}: sampling {target:,} with female priority...")
                selected = sample_with_female_priority(available, target)
            else:
                # For others, balance gender and age
                print(f"  {accent:12s}: sampling {target:,} with gender/age balance...")
                selected = sample_balanced(available, target)

        selected_samples.extend(selected)
        print(f"  {accent:12s}: ✓ selected {len(selected):,} samples")

    print()
    print(f"Total selected: {len(selected_samples):,}")
    return selected_samples


def sample_with_female_priority(samples: List[Dict], target: int) -> List[Dict]:
    """Sample with priority for female speakers (for Spain gender imbalance)."""
    females = [s for s in samples if "female" in s.get("gender", "").lower()]
    males = [s for s in samples if "male" in s.get("gender", "").lower() and "female" not in s.get("gender", "").lower()]

    # Take all females if possible, fill rest with males
    if len(females) >= target // 2:
        # Enough females for 50/50
        selected_females = random.sample(females, target // 2)
        selected_males = random.sample(males, target - target // 2)
    else:
        # Take all females, fill with males
        selected_females = females
        selected_males = random.sample(males, target - len(selected_females))

    return selected_females + selected_males


def sample_balanced(samples: List[Dict], target: int) -> List[Dict]:
    """Sample with gender and age balance."""
    # Group by gender and age
    strata = defaultdict(list)
    for s in samples:
        gender = normalize_gender(s.get("gender", ""))
        age = normalize_age(s.get("age", ""))
        strata[(gender, age)].append(s)

    # Proportional sampling across strata
    total_available = len(samples)
    selected = []

    for (gender, age), stratum_samples in strata.items():
        stratum_proportion = len(stratum_samples) / total_available
        stratum_target = int(target * stratum_proportion)

        if len(stratum_samples) <= stratum_target:
            selected.extend(stratum_samples)
        else:
            selected.extend(random.sample(stratum_samples, stratum_target))

    # Fill to target if needed
    while len(selected) < target:
        remaining = [s for s in samples if s not in selected]
        if not remaining:
            break
        selected.append(random.choice(remaining))

    # Trim to target if overshot
    if len(selected) > target:
        selected = random.sample(selected, target)

    return selected


def compute_statistics(selected_samples: List[Dict], cv_metadata: dict) -> dict:
    """Compute demographic statistics for selected samples."""
    accents = [get_accent_category(s) for s in selected_samples]
    genders = [normalize_gender(s.get("gender", "")) for s in selected_samples]
    ages = [normalize_age(s.get("age", "")) for s in selected_samples]
    speakers = [s["client_id"] for s in selected_samples]

    stats = {
        "total_samples": len(selected_samples),
        "unique_speakers": len(set(speakers)),
        "accent_distribution": dict(Counter(accents)),
        "gender_distribution": dict(Counter(genders)),
        "age_distribution": dict(Counter(ages)),
    }

    return stats


def main():
    """Perform balanced sampling and save 20K selection."""
    print("=" * 70)
    print("Balanced Sampling: Select 20K Common Voice Samples")
    print("=" * 70)
    print()

    random.seed(RANDOM_SEED)

    # Load filtered speakers
    print(f"Loading filtered speakers from {FILTERED_SPEAKERS}...")
    filtered_speakers = load_filtered_speakers()
    print(f"  Filtered speakers: {len(filtered_speakers):,}")
    print()

    # Load CV metadata
    print(f"Loading CV metadata from {CV_SPEAKER_METADATA}...")
    cv_metadata = load_cv_metadata()
    print(f"  Total CV speakers in metadata: {len(cv_metadata):,}")
    print()

    # Parse validated.tsv and filter
    print(f"Parsing {VALIDATED_TSV}...")
    samples_by_speaker = defaultdict(list)
    total_rows = 0

    with open(VALIDATED_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total_rows += 1
            client_id = row["client_id"]

            if client_id in filtered_speakers:
                samples_by_speaker[client_id].append(row)

    print(f"  Total rows in validated.tsv: {total_rows:,}")
    print(f"  Filtered speakers with samples: {len(samples_by_speaker):,}")
    total_available = sum(len(samples) for samples in samples_by_speaker.values())
    print(f"  Total available samples: {total_available:,}")
    print()

    # Stratified sampling with fixed targets
    print(f"Performing stratified sampling (target: {TOTAL_TARGET:,})...")
    print("Per-accent targets:")
    for accent, target in ACCENT_TARGETS.items():
        print(f"  {accent:12s}: {target:,}")
    print()

    selected_samples = stratified_sample(samples_by_speaker, cv_metadata)
    print()

    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(selected_samples, cv_metadata)

    print()
    print("=" * 70)
    print("Selection Statistics")
    print("=" * 70)
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Unique speakers: {stats['unique_speakers']:,}")
    print()
    print("Accent distribution:")
    for accent, count in sorted(stats['accent_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {accent}: {count:,} ({count / stats['total_samples'] * 100:.1f}%)")
    print()
    print("Gender distribution:")
    for gender, count in sorted(stats['gender_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {gender}: {count:,} ({count / stats['total_samples'] * 100:.1f}%)")
    print()
    print("Age distribution:")
    for age, count in sorted(stats['age_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {age}: {count:,} ({count / stats['total_samples'] * 100:.1f}%)")
    print()

    # Save selected samples to TSV
    print(f"Saving selected samples to {SELECTED_OUTPUT}...")
    with open(SELECTED_OUTPUT, "w", encoding="utf-8", newline="") as f:
        fieldnames = selected_samples[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected_samples)

    # Save statistics
    print(f"Saving statistics to {STATS_OUTPUT}...")
    with open(STATS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print()
    print("Done!")
    print()
    print("Output files:")
    print(f"  - {SELECTED_OUTPUT}")
    print(f"  - {STATS_OUTPUT}")


if __name__ == "__main__":
    main()
