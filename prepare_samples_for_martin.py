#!/usr/bin/env python3
"""
Prepare Sample Dataset for Analysis

Extracts samples from N speakers (both bonafide and spoof) from the augmented
dataset for artifact analysis. Creates a subset with protocol file.

Usage:
    python prepare_samples_for_martin.py
    python prepare_samples_for_martin.py --n_speakers 15 --factor 5x
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict


def parse_protocol_file(protocol_path):
    """
    Parse ASVspoof protocol file.

    Returns:
        dict: {audio_id: (speaker_id, system_id, key)}
    """
    protocol = {}
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                speaker_id, audio_id, system_id, key = parts
                protocol[audio_id] = (speaker_id, system_id, key)
    return protocol


def group_by_speaker(protocol):
    """
    Group audio IDs by speaker and type (bonafide/spoof).

    Returns:
        dict: {speaker_id: {'bonafide': [audio_ids], 'spoof': [audio_ids]}}
    """
    speakers = defaultdict(lambda: {'bonafide': [], 'spoof': []})

    for audio_id, (speaker_id, system_id, key) in protocol.items():
        speakers[speaker_id][key].append(audio_id)

    return speakers


def select_speakers(speakers, n_speakers=10, require_both=True):
    """
    Select N speakers that have both bonafide and spoof samples.

    Args:
        speakers: Speaker grouping dict.
        n_speakers: Number of speakers to select.
        require_both: If True, only select speakers with both bonafide and spoof.

    Returns:
        list: Selected speaker IDs.
    """
    if require_both:
        valid_speakers = [
            speaker_id for speaker_id, samples in speakers.items()
            if samples['bonafide'] and samples['spoof']
        ]
    else:
        valid_speakers = list(speakers.keys())

    # Sort by total sample count (descending) to get speakers with most data
    valid_speakers.sort(
        key=lambda sid: len(speakers[sid]['bonafide']) + len(speakers[sid]['spoof']),
        reverse=True
    )

    return valid_speakers[:n_speakers]


def prepare_samples(
    source_dir,
    output_dir,
    n_speakers=10,
    factor="10x",
    require_both=True
):
    """
    Extract samples from N speakers and create subset dataset.

    Args:
        source_dir: Root of augmented dataset (e.g., data/augmented).
        output_dir: Output directory for subset (e.g., datos_martin).
        n_speakers: Number of speakers to extract.
        factor: Augmentation factor (e.g., "10x").
        require_both: Only select speakers with both bonafide and spoof.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Construct paths
    ratio_str = "5050"  # Default balanced ratio
    augmented_dir = source_dir / f"augmented_{factor}_balanced_{ratio_str}"

    if not augmented_dir.exists():
        print(f"ERROR: Augmented directory not found: {augmented_dir}")
        print(f"Available directories in {source_dir}:")
        if source_dir.exists():
            for d in source_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return

    train_dir = augmented_dir / "LA" / "ASVspoof2019_LA_train"
    flac_dir = train_dir / "flac"
    protocol_file = train_dir / "ASVspoof2019.LA.cm.train.trn.txt"

    if not protocol_file.exists():
        print(f"ERROR: Protocol file not found: {protocol_file}")
        return

    print(f"\nReading protocol from: {protocol_file}")
    protocol = parse_protocol_file(protocol_file)
    print(f"  Total samples: {len(protocol):,}")

    print(f"\nGrouping by speaker...")
    speakers = group_by_speaker(protocol)
    print(f"  Total speakers: {len(speakers)}")

    print(f"\nSelecting {n_speakers} speakers...")
    selected_speakers = select_speakers(speakers, n_speakers, require_both)

    if len(selected_speakers) < n_speakers:
        print(f"WARNING: Only {len(selected_speakers)} speakers available with both bonafide and spoof")

    print(f"  Selected speakers:")
    for speaker_id in selected_speakers:
        n_bonafide = len(speakers[speaker_id]['bonafide'])
        n_spoof = len(speakers[speaker_id]['spoof'])
        print(f"    {speaker_id}: {n_bonafide} bonafide, {n_spoof} spoof")

    # Collect audio IDs to copy
    audio_ids_to_copy = []
    for speaker_id in selected_speakers:
        audio_ids_to_copy.extend(speakers[speaker_id]['bonafide'])
        audio_ids_to_copy.extend(speakers[speaker_id]['spoof'])

    print(f"\nTotal samples to copy: {len(audio_ids_to_copy):,}")

    # Create output directories
    output_flac_dir = output_dir / "flac"
    output_flac_dir.mkdir(parents=True, exist_ok=True)

    # Copy FLAC files
    print(f"\nCopying FLAC files to {output_flac_dir}...")
    copied = 0
    missing = 0

    for audio_id in audio_ids_to_copy:
        src_file = flac_dir / f"{audio_id}.flac"
        dst_file = output_flac_dir / f"{audio_id}.flac"

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            copied += 1
        else:
            print(f"  WARNING: Missing file: {src_file}")
            missing += 1

    print(f"  Copied: {copied:,} files")
    if missing > 0:
        print(f"  Missing: {missing} files")

    # Create subset protocol file
    output_protocol = output_dir / "protocol.txt"
    print(f"\nWriting protocol file to {output_protocol}...")

    with open(output_protocol, 'w') as f:
        for audio_id in audio_ids_to_copy:
            if audio_id in protocol:
                speaker_id, system_id, key = protocol[audio_id]
                f.write(f"{speaker_id} {audio_id} {system_id} {key}\n")

    print(f"  Written: {len(audio_ids_to_copy):,} entries")

    # Create README
    readme_path = output_dir / "README.txt"
    print(f"\nWriting README to {readme_path}...")

    with open(readme_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("AUGMENTED VOICE SAMPLES FOR ARTIFACT ANALYSIS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Dataset Information:\n")
        f.write(f"  Source: {augmented_dir}\n")
        f.write(f"  Augmentation factor: {factor}\n")
        f.write(f"  Target ratio: 50/50 (bonafide/spoof)\n")
        f.write(f"  Number of speakers: {len(selected_speakers)}\n")
        f.write(f"  Total samples: {len(audio_ids_to_copy):,}\n\n")

        f.write("Selected Speakers:\n")
        for speaker_id in selected_speakers:
            n_bonafide = len(speakers[speaker_id]['bonafide'])
            n_spoof = len(speakers[speaker_id]['spoof'])
            f.write(f"  {speaker_id}: {n_bonafide} bonafide, {n_spoof} spoof\n")

        f.write("\n" + "="*70 + "\n")
        f.write("FILE STRUCTURE\n")
        f.write("="*70 + "\n\n")

        f.write("flac/\n")
        f.write("  Contains all audio files in FLAC format\n")
        f.write("  Filename format: LA_T_NNNNNNN.flac\n\n")

        f.write("protocol.txt\n")
        f.write("  Protocol file mapping audio IDs to metadata\n")
        f.write("  Format: SPEAKER_ID AUDIO_ID SYSTEM_ID KEY\n")
        f.write("  - SPEAKER_ID: Speaker identifier\n")
        f.write("  - AUDIO_ID: Unique file identifier\n")
        f.write("  - SYSTEM_ID: Augmentation type ('-' = clean/original)\n")
        f.write("  - KEY: 'bonafide' or 'spoof'\n\n")

        f.write("="*70 + "\n")
        f.write("AUGMENTATION TYPES\n")
        f.write("="*70 + "\n\n")

        f.write("1. RIR + Noise: Room acoustics + background noise\n")
        f.write("   Format: RIR_{ROOM}_{NOISE}_SNR{DB}\n")
        f.write("   - ROOM: SMALL, MEDIUM, LARGE\n")
        f.write("   - NOISE: NOI (noise), SPE (speech), MUS (music)\n")
        f.write("   - SNR: Signal-to-noise ratio in dB\n")
        f.write("   Example: RIR_SMALL_NOI_SNR12\n\n")

        f.write("2. Codec Degradation: Telephone/VoIP artifacts\n")
        f.write("   Format: CODEC_{SR}K_LOSS{PCT}PCT[_BP][_Q{BITS}]\n")
        f.write("   - SR: Sample rate (8K or 16K)\n")
        f.write("   - LOSS: Packet loss percentage\n")
        f.write("   - _BP: Bandpass filter applied (optional)\n")
        f.write("   - _Q{BITS}: Quantization (8 or 12 bits, optional)\n")
        f.write("   Example: CODEC_8K_LOSS2PCT_BP_Q8\n\n")

        f.write("3. RawBoost: Signal-dependent distortions\n")
        f.write("   Format: RAWBOOST_{OP1}[_{OP2}...]\n")
        f.write("   Operations:\n")
        f.write("   - LF: Linear FIR filtering\n")
        f.write("   - NL: Nonlinear tanh distortion\n")
        f.write("   - AN: Additive noise\n")
        f.write("   - GV: Gain variation\n")
        f.write("   - CL: Clipping\n")
        f.write("   Example: RAWBOOST_LF_AN_GV\n\n")

        f.write("="*70 + "\n")
        f.write("USAGE\n")
        f.write("="*70 + "\n\n")

        f.write("1. Load protocol.txt to get file metadata\n")
        f.write("2. Load corresponding FLAC files from flac/ directory\n")
        f.write("3. Analyze artifacts by comparing:\n")
        f.write("   - Clean samples (SYSTEM_ID = '-')\n")
        f.write("   - Augmented samples (various SYSTEM_ID values)\n")
        f.write("4. Group by SPEAKER_ID to analyze per-speaker effects\n\n")

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"  - {copied:,} FLAC files in flac/")
    print(f"  - Protocol file: protocol.txt")
    print(f"  - Documentation: README.txt")
    print("\nReady to share with Martin for artifact analysis!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract samples from augmented dataset for artifact analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Default (10 speakers from 10x augmentation):
    python prepare_samples_for_martin.py

  Extract 15 speakers:
    python prepare_samples_for_martin.py --n_speakers 15

  Use 5x augmentation:
    python prepare_samples_for_martin.py --factor 5x
        """
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/augmented",
        help="Source augmented dataset directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="datos_martin",
        help="Output directory for extracted samples"
    )

    parser.add_argument(
        "--n_speakers",
        type=int,
        default=10,
        help="Number of speakers to extract"
    )

    parser.add_argument(
        "--factor",
        type=str,
        default="10x",
        help="Augmentation factor (e.g., 2x, 3x, 5x, 10x)"
    )

    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Allow speakers with only bonafide or only spoof (not both)"
    )

    args = parser.parse_args()

    prepare_samples(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        n_speakers=args.n_speakers,
        factor=args.factor,
        require_both=not args.allow_incomplete
    )


if __name__ == "__main__":
    main()
