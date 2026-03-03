"""
Count available data from HABLA and Common Voice (Mexico + Spain only).

This script analyzes:
1. HABLA bonafide samples by accent (CO, CL, PE, VE, AR)
2. Common Voice Mexico samples (gender/age breakdown)
3. Common Voice Spain samples (gender/age breakdown)

Output:
    Prints detailed statistics to console

Usage:
    python pipeline/select_mozilla_speakers/00_count_available_data.py
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HABLA_DIR = Path("data/bonafide_dataset_by_speaker")
CV_METADATA_TSV = Path("data/cv_metadata/validated.tsv")


def count_habla_data():
    """Count HABLA samples by accent."""
    print("=" * 70)
    print("HABLA Dataset Analysis")
    print("=" * 70)
    print()

    accent_counts = Counter()
    gender_counts = Counter()
    total_speakers = 0
    total_samples = 0

    speaker_dirs = sorted([d for d in HABLA_DIR.iterdir() if d.is_dir()])

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        total_speakers += 1

        # Extract accent and gender from speaker ID (e.g., "arf_00295" -> ar=Argentina, f=female)
        accent_code = speaker_id[:2]
        gender_code = speaker_id[2]

        accent_map = {
            "ar": "Argentina",
            "cl": "Chile",
            "co": "Colombia",
            "pe": "Peru",
            "ve": "Venezuela",
        }

        gender_map = {
            "f": "Female",
            "m": "Male",
        }

        accent = accent_map.get(accent_code, "Unknown")
        gender = gender_map.get(gender_code, "Unknown")

        # Count samples from train/val/test
        speaker_samples = 0
        for split_dir in ["train", "val", "test"]:
            split_path = speaker_dir / split_dir
            if split_path.exists():
                split_samples = len(list(split_path.glob("*.wav")))
                speaker_samples += split_samples

        accent_counts[accent] += speaker_samples
        gender_counts[f"{accent}_{gender}"] += speaker_samples
        total_samples += speaker_samples

    print(f"Total speakers: {total_speakers}")
    print(f"Total samples: {total_samples:,}")
    print()

    print("Samples by accent:")
    for accent in ["Argentina", "Chile", "Colombia", "Peru", "Venezuela"]:
        count = accent_counts[accent]
        pct = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {accent:15s}: {count:6,} samples ({pct:5.2f}%)")

    print()
    print("Gender breakdown by accent:")
    for accent in ["Argentina", "Chile", "Colombia", "Peru", "Venezuela"]:
        male = gender_counts[f"{accent}_Male"]
        female = gender_counts[f"{accent}_Female"]
        total = male + female
        print(f"  {accent:15s}: {male:5,} M / {female:5,} F (total: {total:5,})")

    print()
    return accent_counts, total_samples


def count_cv_data():
    """Count Common Voice Mexico and Spain samples with demographics."""
    print("=" * 70)
    print("Common Voice: Mexico + Spain Analysis")
    print("=" * 70)
    print()

    mexico_samples = []
    spain_samples = []

    with open(CV_METADATA_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            accent = row.get("accents", "").strip().lower()

            # Check Mexico
            if "méxico" in accent or "mexico" in accent:
                mexico_samples.append({
                    "gender": row.get("gender", ""),
                    "age": row.get("age", ""),
                    "client_id": row["client_id"],
                })

            # Check Spain
            if any(kw in accent for kw in ["espa", "madrid", "castilla", "andaluc", "catalu", "galicia", "valencia", "canaria", "peninsular"]):
                spain_samples.append({
                    "gender": row.get("gender", ""),
                    "age": row.get("age", ""),
                    "client_id": row["client_id"],
                })

    # Mexico stats
    print("MEXICO")
    print("-" * 70)
    print(f"Total samples: {len(mexico_samples):,}")
    print(f"Unique speakers: {len(set(s['client_id'] for s in mexico_samples)):,}")
    print()

    mexico_genders = Counter(s["gender"] or "Unknown" for s in mexico_samples)
    print("Gender distribution:")
    for gender, count in mexico_genders.most_common():
        pct = count / len(mexico_samples) * 100
        print(f"  {gender:20s}: {count:6,} ({pct:5.2f}%)")

    print()
    mexico_ages = Counter(s["age"] or "Unknown" for s in mexico_samples)
    print("Age distribution:")
    for age, count in mexico_ages.most_common():
        pct = count / len(mexico_samples) * 100
        print(f"  {age:20s}: {count:6,} ({pct:5.2f}%)")

    print()
    print()

    # Spain stats
    print("SPAIN")
    print("-" * 70)
    print(f"Total samples: {len(spain_samples):,}")
    print(f"Unique speakers: {len(set(s['client_id'] for s in spain_samples)):,}")
    print()

    spain_genders = Counter(s["gender"] or "Unknown" for s in spain_samples)
    print("Gender distribution:")
    for gender, count in spain_genders.most_common():
        pct = count / len(spain_samples) * 100
        print(f"  {gender:20s}: {count:6,} ({pct:5.2f}%)")

    print()
    spain_ages = Counter(s["age"] or "Unknown" for s in spain_samples)
    print("Age distribution:")
    for age, count in spain_ages.most_common():
        pct = count / len(spain_samples) * 100
        print(f"  {age:20s}: {count:6,} ({pct:5.2f}%)")

    print()

    return len(mexico_samples), len(spain_samples)


def main():
    """Count all available data."""
    # Count HABLA
    habla_counts, habla_total = count_habla_data()

    # Count CV Mexico + Spain
    mexico_count, spain_count = count_cv_data()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"HABLA total samples: {habla_total:,}")
    print(f"  Argentina:  {habla_counts['Argentina']:,}")
    print(f"  Chile:      {habla_counts['Chile']:,}")
    print(f"  Colombia:   {habla_counts['Colombia']:,}")
    print(f"  Peru:       {habla_counts['Peru']:,}")
    print(f"  Venezuela:  {habla_counts['Venezuela']:,}")
    print()
    print(f"Common Voice available:")
    print(f"  Mexico:     {mexico_count:,} samples")
    print(f"  Spain:      {spain_count:,} samples")
    print()
    print("Next step: Decide how many samples to take from Mexico and Spain")
    print("to balance the final dataset across all 7 accents.")


if __name__ == "__main__":
    main()
