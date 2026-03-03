"""
Extract speaker embeddings from HABLA bonafide dataset using ECAPA-TDNN.

This script processes all 162 HABLA speakers from the bonafide_dataset_by_speaker
directory, extracts ECAPA-TDNN embeddings from their training samples, and averages
them to produce one representative 512-dimensional embedding per speaker.

Output:
    data/mozilla_speaker_selection/habla_embeddings.npy: (162, 512) float32 array
    data/mozilla_speaker_selection/habla_speaker_ids.json: List of speaker IDs

Usage:
    python -m pipeline.select_mozilla_speakers.01_extract_habla_embeddings
"""
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm


HABLA_DIR = Path("data/bonafide_dataset_by_speaker")
OUTPUT_DIR = Path("data/mozilla_speaker_selection")
EMBEDDINGS_OUTPUT = OUTPUT_DIR / "habla_embeddings.npy"
IDS_OUTPUT = OUTPUT_DIR / "habla_speaker_ids.json"

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(audio_path: Path) -> torch.Tensor:
    """Load audio file and resample to 16kHz if needed.

    Args:
        audio_path: Path to audio file (.wav)

    Returns:
        Audio tensor resampled to 16kHz, shape (1, samples)
    """
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform


def extract_speaker_embedding(
    model: EncoderClassifier,
    audio_files: List[Path],
    max_samples: int = 20
) -> np.ndarray:
    """Extract and average embeddings from multiple audio files.

    Args:
        model: ECAPA-TDNN encoder model
        audio_files: List of audio file paths for one speaker
        max_samples: Maximum number of samples to average (for speed)

    Returns:
        Averaged embedding vector, shape (512,)
    """
    embeddings = []

    # Limit to max_samples for efficiency
    selected_files = audio_files[:max_samples] if len(audio_files) > max_samples else audio_files

    for audio_path in selected_files:
        try:
            waveform = load_audio(audio_path)

            # Extract embedding (SpeechBrain returns shape: (1, 1, 192) or (1, 1, 512) depending on model)
            with torch.no_grad():
                embedding = model.encode_batch(waveform.to(DEVICE))

            # Flatten to 1D
            embedding = embedding.squeeze().cpu().numpy()
            embeddings.append(embedding)

        except Exception as e:
            print(f"  Warning: Failed to process {audio_path.name}: {e}")
            continue

    if not embeddings:
        raise ValueError(f"No valid embeddings extracted from {len(audio_files)} files")

    # Average embeddings
    avg_embedding = np.mean(embeddings, axis=0)

    # L2 normalize
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    return avg_embedding


def main():
    """Extract HABLA speaker embeddings and save to disk."""
    print("=" * 70)
    print("HABLA Speaker Embedding Extraction")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"HABLA directory: {HABLA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ECAPA-TDNN model from SpeechBrain
    print("Loading ECAPA-TDNN model (speechbrain/spkrec-ecapa-voxceleb)...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    )
    print(f"Model loaded successfully on {DEVICE}")
    print()

    # Get all speaker directories
    speaker_dirs = sorted([d for d in HABLA_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(speaker_dirs)} HABLA speakers")
    print()

    speaker_ids = []
    embeddings = []

    # Process each speaker
    for idx, speaker_dir in enumerate(speaker_dirs, 1):
        speaker_id = speaker_dir.name
        speaker_ids.append(speaker_id)

        # Get all training audio files
        train_dir = speaker_dir / "train"
        if not train_dir.exists():
            print(f"Warning: No train directory for {speaker_id}, skipping")
            continue

        audio_files = sorted(train_dir.glob("*.wav"))

        if not audio_files:
            print(f"Warning: No audio files for {speaker_id}, skipping")
            continue

        # Extract and average embeddings
        try:
            print(f"[{idx}/{len(speaker_dirs)}] Processing {speaker_id}: {len(audio_files)} audio files (using up to 20)")
            avg_embedding = extract_speaker_embedding(model, audio_files)
            embeddings.append(avg_embedding)
            print(f"  ✓ Completed {speaker_id}")
        except Exception as e:
            print(f"  ✗ Error processing {speaker_id}: {e}")
            speaker_ids.pop()  # Remove this speaker ID
            continue

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)

    print()
    print("=" * 70)
    print("Extraction Complete")
    print("=" * 70)
    print(f"Successfully processed: {len(speaker_ids)} speakers")
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print()

    # Save embeddings
    print(f"Saving embeddings to: {EMBEDDINGS_OUTPUT}")
    np.save(EMBEDDINGS_OUTPUT, embeddings_array)

    # Save speaker IDs
    print(f"Saving speaker IDs to: {IDS_OUTPUT}")
    with open(IDS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(speaker_ids, f, indent=2)

    print()
    print("Done!")
    print()
    print("Output files:")
    print(f"  - {EMBEDDINGS_OUTPUT} ({embeddings_array.nbytes / 1024:.2f} KB)")
    print(f"  - {IDS_OUTPUT}")


if __name__ == "__main__":
    main()
