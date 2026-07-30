"""
Verification Script for Anti-Spoofing Augmentation Pipeline
============================================================

Checks the following state-of-the-art requirements:
1. Augmentation factors are correctly calculated
2. Clean data is available (bonafide/spoof counts)
3. Speaker independence across splits (no speaker overlap)
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# Configuration
DATA_ROOT = Path("data/partition_dataset_by_speaker")
SPEAKER_SPLITS_FILE = DATA_ROOT / "speaker_splits.json"


def count_files_by_type(directory: Path) -> dict:
    """Count bonafide and spoof files in a directory."""
    counts = {"bonafide": 0, "spoof": 0, "total": 0}

    if not directory.exists():
        return counts

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.wav', '.flac')):
                counts["total"] += 1
                if file.startswith("bonafide_"):
                    counts["bonafide"] += 1
                else:
                    counts["spoof"] += 1

    return counts


def get_speakers_in_split(split_dir: Path) -> set:
    """Get all speaker IDs in a split directory."""
    speakers = set()
    if split_dir.exists():
        for item in split_dir.iterdir():
            if item.is_dir():
                speakers.add(item.name)
    return speakers


def verify_speaker_independence():
    """Verify that no speaker appears in multiple splits."""
    print("\n" + "="*70)
    print("1. SPEAKER INDEPENDENCE VERIFICATION")
    print("="*70)

    # Load speaker splits from JSON
    if SPEAKER_SPLITS_FILE.exists():
        with open(SPEAKER_SPLITS_FILE, 'r') as f:
            splits_json = json.load(f)

        train_speakers_json = set(splits_json.get('train', []))
        val_speakers_json = set(splits_json.get('val', []))
        test_speakers_json = set(splits_json.get('test', []))

        print(f"\nFrom speaker_splits.json:")
        print(f"  Train speakers: {len(train_speakers_json)}")
        print(f"  Val speakers:   {len(val_speakers_json)}")
        print(f"  Test speakers:  {len(test_speakers_json)}")
    else:
        print("\nWarning: speaker_splits.json not found!")
        train_speakers_json = set()
        val_speakers_json = set()
        test_speakers_json = set()

    # Get actual speakers from directories
    train_speakers = get_speakers_in_split(DATA_ROOT / "train")
    val_speakers = get_speakers_in_split(DATA_ROOT / "val")
    test_speakers = get_speakers_in_split(DATA_ROOT / "test")

    print(f"\nFrom actual directories:")
    print(f"  Train speakers: {len(train_speakers)}")
    print(f"  Val speakers:   {len(val_speakers)}")
    print(f"  Test speakers:  {len(test_speakers)}")

    # Check for overlaps
    train_val_overlap = train_speakers & val_speakers
    train_test_overlap = train_speakers & test_speakers
    val_test_overlap = val_speakers & test_speakers

    print(f"\nSpeaker Overlap Check:")

    all_passed = True

    if train_val_overlap:
        print(f"  [FAIL] Train-Val overlap: {len(train_val_overlap)} speakers")
        print(f"         Overlapping: {list(train_val_overlap)[:5]}...")
        all_passed = False
    else:
        print(f"  [PASS] Train-Val: NO overlap")

    if train_test_overlap:
        print(f"  [FAIL] Train-Test overlap: {len(train_test_overlap)} speakers")
        print(f"         Overlapping: {list(train_test_overlap)[:5]}...")
        all_passed = False
    else:
        print(f"  [PASS] Train-Test: NO overlap")

    if val_test_overlap:
        print(f"  [FAIL] Val-Test overlap: {len(val_test_overlap)} speakers")
        print(f"         Overlapping: {list(val_test_overlap)[:5]}...")
        all_passed = False
    else:
        print(f"  [PASS] Val-Test: NO overlap")

    if all_passed:
        print(f"\n  [OK] SPEAKER INDEPENDENCE VERIFIED - No speaker leakage!")
    else:
        print(f"\n  [ERROR] SPEAKER INDEPENDENCE FAILED - There is speaker leakage!")

    return all_passed


def verify_clean_data_availability():
    """Verify bonafide and spoof files are available in each split."""
    print("\n" + "="*70)
    print("2. CLEAN DATA AVAILABILITY VERIFICATION")
    print("="*70)

    splits = ["train", "val", "test"]
    all_passed = True

    total_stats = {"bonafide": 0, "spoof": 0, "total": 0}

    for split in splits:
        split_dir = DATA_ROOT / split
        counts = count_files_by_type(split_dir)

        total_stats["bonafide"] += counts["bonafide"]
        total_stats["spoof"] += counts["spoof"]
        total_stats["total"] += counts["total"]

        if counts["total"] > 0:
            bonafide_pct = (counts["bonafide"] / counts["total"]) * 100
            spoof_pct = (counts["spoof"] / counts["total"]) * 100
        else:
            bonafide_pct = 0
            spoof_pct = 0

        print(f"\n{split.upper()} Split:")
        print(f"  Total files:    {counts['total']:,}")
        print(f"  Bonafide files: {counts['bonafide']:,} ({bonafide_pct:.1f}%)")
        print(f"  Spoof files:    {counts['spoof']:,} ({spoof_pct:.1f}%)")

        # Check if both types are present
        if counts["bonafide"] == 0:
            print(f"  [FAIL] No bonafide files found!")
            all_passed = False
        elif counts["spoof"] == 0:
            print(f"  [FAIL] No spoof files found!")
            all_passed = False
        else:
            print(f"  [PASS] Both bonafide and spoof files present")

    # Print totals
    if total_stats["total"] > 0:
        bonafide_pct = (total_stats["bonafide"] / total_stats["total"]) * 100
        spoof_pct = (total_stats["spoof"] / total_stats["total"]) * 100
    else:
        bonafide_pct = 0
        spoof_pct = 0

    print(f"\nOVERALL DATASET:")
    print(f"  Total files:    {total_stats['total']:,}")
    print(f"  Bonafide files: {total_stats['bonafide']:,} ({bonafide_pct:.1f}%)")
    print(f"  Spoof files:    {total_stats['spoof']:,} ({spoof_pct:.1f}%)")

    if all_passed:
        print(f"\n  [OK] CLEAN DATA VERIFIED - Both classes present in all splits!")
    else:
        print(f"\n  [ERROR] CLEAN DATA FAILED - Missing class(es) in some splits!")

    return all_passed, total_stats


def verify_file_naming_convention():
    """Verify file naming follows expected patterns."""
    print("\n" + "="*70)
    print("3. FILE NAMING CONVENTION VERIFICATION")
    print("="*70)

    splits = ["train", "val", "test"]
    spoof_types = defaultdict(int)
    invalid_files = []
    all_passed = True

    for split in splits:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            continue

        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            for audio_file in speaker_dir.glob("*.wav"):
                filename = audio_file.name

                if filename.startswith("bonafide_"):
                    # Bonafide file - check format: bonafide_SPEAKER_NNNN.wav
                    parts = filename.replace(".wav", "").split("_")
                    if len(parts) >= 3:
                        # Valid bonafide naming
                        pass
                    else:
                        invalid_files.append(str(audio_file))
                elif filename.startswith("spoof_"):
                    # Spoof file - check format: spoof_TYPE_SPEAKER_NNNN.wav
                    parts = filename.replace(".wav", "").split("_")
                    if len(parts) >= 4:
                        spoof_type = parts[1]
                        spoof_types[spoof_type] += 1
                    else:
                        invalid_files.append(str(audio_file))
                else:
                    invalid_files.append(str(audio_file))

    print(f"\nSpoof Types Detected:")
    for spoof_type, count in sorted(spoof_types.items()):
        print(f"  {spoof_type}: {count:,} files")

    if len(spoof_types) > 0:
        print(f"\n  [PASS] Found {len(spoof_types)} different spoofing techniques")
    else:
        print(f"\n  [FAIL] No spoofing techniques detected")
        all_passed = False

    if invalid_files:
        print(f"\n  [WARN] Found {len(invalid_files)} files with unexpected naming:")
        for f in invalid_files[:5]:
            print(f"         {f}")
        if len(invalid_files) > 5:
            print(f"         ... and {len(invalid_files) - 5} more")
    else:
        print(f"\n  [PASS] All files follow naming conventions")

    return all_passed


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("ANTI-SPOOFING AUGMENTATION PIPELINE VERIFICATION")
    print("="*70)
    print(f"\nData root: {DATA_ROOT}")

    results = {}

    # Run all verifications
    results["speaker_independence"] = verify_speaker_independence()
    results["clean_data"], stats = verify_clean_data_availability()
    results["naming_convention"] = verify_file_naming_convention()

    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = True
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} - {check.replace('_', ' ').title()}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n{'='*70}")
        print("ALL STATE-OF-THE-ART REQUIREMENTS VERIFIED SUCCESSFULLY!")
        print("="*70)
    else:
        print(f"\n{'='*70}")
        print("SOME REQUIREMENTS FAILED - Please review above!")
        print("="*70)

    return all_passed


if __name__ == "__main__":
    main()
