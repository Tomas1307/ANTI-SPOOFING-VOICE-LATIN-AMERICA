"""
Extract speaker embeddings from Common Voice Spanish dataset using ECAPA-TDNN.

This script extracts the Common Voice archive, parses validated.tsv, filters by
accent (LatAm + Spain), groups samples by speaker, and extracts one representative
512-dimensional ECAPA-TDNN embedding per speaker.

Input:
    data/cv-corpus-24.0-2025-12-05-es.tar.gz
    data/cv_metadata/validated.tsv

Output:
    data/mozilla_speaker_selection/cv_speaker_embeddings.npy: (N, 512) float32
    data/mozilla_speaker_selection/cv_speaker_metadata.json: Speaker metadata

Usage:
    python -m pipeline.select_mozilla_speakers.02_extract_cv_embeddings
"""
import csv
import json
import os
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm


CV_ARCHIVE = Path("data/cv-corpus-24.0-2025-12-05-es.tar.gz")
CV_EXTRACTED_DIR = Path("data/cv-corpus-24.0-2025-12-05")
CV_METADATA_TSV = CV_EXTRACTED_DIR / "es" / "validated.tsv"
OUTPUT_DIR = Path("data/mozilla_speaker_selection")
EMBEDDINGS_OUTPUT = OUTPUT_DIR / "cv_speaker_embeddings.npy"
METADATA_OUTPUT = OUTPUT_DIR / "cv_speaker_metadata.json"

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Accent keywords for filtering (ONLY: Colombia, Chile, Venezuela, Mexico, Spain)
# We exclude: Argentina (already in HABLA), Peru (already in HABLA),
# and other LatAm regions (Rioplatense, Caribe, Central America)

COLOMBIA_KEYWORDS = ["colombia", "bogot", "colombiano"]
CHILE_KEYWORDS = ["chile", "chileno"]
VENEZUELA_KEYWORDS = ["venezuel"]
MEXICO_KEYWORDS = ["méxico", "mexico", "mexicano"]
SPAIN_KEYWORDS = [
    "espa", "canaria", "peninsular", "madrid", "castilla",
    "andaluc", "catalu", "galicia", "baleares", "valencia"
]


def extract_archive():
    """Extract Common Voice archive if not already extracted."""
    if CV_EXTRACTED_DIR.exists():
        print(f"Archive already extracted at {CV_EXTRACTED_DIR}")
        return

    print(f"Extracting {CV_ARCHIVE}...")
    print("Reading archive contents...")
    with tarfile.open(CV_ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Found {len(members):,} files in archive")
        print("Extracting files (this will take several minutes)...")

        for member in tqdm(members, desc="Extracting", unit="files"):
            tar.extract(member, path=CV_EXTRACTED_DIR.parent)

    print("✓ Extraction complete!")


def is_valid_sample(accent: str, gender: str, age: str) -> tuple:
    """Check if sample meets selection criteria.

    Returns:
        Tuple of (is_valid, accent_category)
        accent_category is one of: "Colombia", "Chile", "Venezuela", "Mexico", "Spain", None
    """
    # Exclude Unknown gender or age
    if not gender or not age or gender.lower() == "unknown" or age.lower() == "unknown":
        return False, None

    if not accent:
        return False, None

    accent_lower = accent.lower()

    # Check Colombia
    for kw in COLOMBIA_KEYWORDS:
        if kw in accent_lower:
            return True, "Colombia"

    # Check Chile
    for kw in CHILE_KEYWORDS:
        if kw in accent_lower:
            return True, "Chile"

    # Check Venezuela
    for kw in VENEZUELA_KEYWORDS:
        if kw in accent_lower:
            return True, "Venezuela"

    # Check Mexico
    for kw in MEXICO_KEYWORDS:
        if kw in accent_lower:
            return True, "Mexico"

    # Check Spain
    for kw in SPAIN_KEYWORDS:
        if kw in accent_lower:
            return True, "Spain"

    return False, None


def parse_validated_tsv() -> Dict[str, List[Dict]]:
    """Parse validated.tsv and group samples by speaker.

    Filters for: Colombia, Chile, Venezuela, Mexico, Spain
    Excludes: Unknown gender or age

    Returns:
        Dict mapping client_id to list of sample metadata dicts
    """
    print(f"Parsing {CV_METADATA_TSV}...")
    speakers = defaultdict(list)
    total_samples = 0
    filtered_samples = 0
    accent_counts = Counter()

    with open(CV_METADATA_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total_samples += 1
            accent = row.get("accents", "").strip()
            gender = row.get("gender", "").strip()
            age = row.get("age", "").strip()

            is_valid, accent_category = is_valid_sample(accent, gender, age)

            if not is_valid:
                continue

            filtered_samples += 1
            accent_counts[accent_category] += 1
            client_id = row["client_id"]
            speakers[client_id].append({
                "path": row["path"],
                "accent": accent,
                "accent_category": accent_category,
                "gender": gender,
                "age": age,
            })

    print(f"Total samples in validated.tsv: {total_samples:,}")
    print(f"Filtered samples (CO/CL/VE/MX/ES, no Unknown): {filtered_samples:,}")
    print(f"Unique speakers: {len(speakers):,}")
    print()
    print("Samples by accent:")
    for accent in ["Colombia", "Chile", "Venezuela", "Mexico", "Spain"]:
        print(f"  {accent:12s}: {accent_counts[accent]:,}")
    return dict(speakers)


def load_audio(audio_path: Path) -> torch.Tensor:
    """Load audio file and resample to 16kHz if needed."""
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform


def extract_cv_embeddings(
    model: EncoderClassifier,
    speakers: Dict[str, List[Dict]]
) -> tuple:
    """Extract one embedding per speaker from Common Voice data.

    Args:
        model: ECAPA-TDNN encoder model
        speakers: Dict mapping client_id to list of samples

    Returns:
        Tuple of (embeddings_array, speaker_metadata_dict)
    """
    embeddings = []
    metadata = {}
    failed_speakers = []

    # Find audio clips directory
    clips_dir = CV_EXTRACTED_DIR / "cv-corpus-24.0-2025-12-05" / "es" / "clips"
    if not clips_dir.exists():
        # Try alternative path
        clips_dir = CV_EXTRACTED_DIR / "es" / "clips"
        if not clips_dir.exists():
            raise FileNotFoundError(
                f"Could not find clips directory. Expected at {clips_dir}"
            )

    print(f"Audio clips directory: {clips_dir}")
    print()

    total_speakers = len(speakers)
    for idx, (client_id, samples) in enumerate(speakers.items(), 1):
        # Select representative sample (longest or first)
        selected_sample = samples[0]
        audio_filename = selected_sample["path"]
        audio_path = clips_dir / audio_filename

        accent_cat = selected_sample.get("accent_category", "unknown")
        if idx % 100 == 0 or idx <= 10:  # Show progress every 100 speakers + first 10
            print(f"[{idx}/{total_speakers}] Processing {client_id}: {accent_cat}, {selected_sample.get('gender', 'unknown')}")

        if not audio_path.exists():
            failed_speakers.append(client_id)
            continue

        try:
            waveform = load_audio(audio_path)

            with torch.no_grad():
                embedding = model.encode_batch(waveform.to(DEVICE))

            embedding = embedding.squeeze().cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)

            embeddings.append(embedding)
            metadata[client_id] = {
                "sample_count": len(samples),
                "accent": selected_sample["accent"],
                "gender": selected_sample["gender"],
                "age": selected_sample["age"],
                "embedding_from": audio_filename,
                "accent_category": accent_cat,
            }

        except Exception as e:
            failed_speakers.append(client_id)
            continue

    embeddings_array = np.array(embeddings, dtype=np.float32)

    print()
    print(f"Successfully processed: {len(metadata):,} speakers")
    print(f"Failed: {len(failed_speakers):,} speakers")

    return embeddings_array, metadata


def main():
    """Extract Common Voice speaker embeddings and save to disk."""
    print("=" * 70)
    print("Common Voice Speaker Embedding Extraction")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()

    # Extract archive
    extract_archive()
    print()

    # Parse validated.tsv
    speakers = parse_validated_tsv()
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ECAPA-TDNN model
    print("Loading ECAPA-TDNN model...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    )
    print(f"Model loaded successfully on {DEVICE}")
    print()

    # Extract embeddings
    embeddings_array, metadata = extract_cv_embeddings(model, speakers)

    print()
    print("=" * 70)
    print("Extraction Complete")
    print("=" * 70)
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print()

    # Save embeddings
    print(f"Saving embeddings to: {EMBEDDINGS_OUTPUT}")
    np.save(EMBEDDINGS_OUTPUT, embeddings_array)

    # Save metadata
    print(f"Saving metadata to: {METADATA_OUTPUT}")
    with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("Done!")
    print()
    print("Output files:")
    print(f"  - {EMBEDDINGS_OUTPUT} ({embeddings_array.nbytes / 1024 / 1024:.2f} MB)")
    print(f"  - {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()
