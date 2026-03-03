"""
Filter Common Voice speakers by cosine similarity to HABLA speakers.

This script loads HABLA and Common Voice speaker embeddings, computes pairwise
cosine similarity, and filters out CV speakers that are too similar to any HABLA
speaker (threshold: 0.75).

Input:
    data/mozilla_speaker_selection/habla_embeddings.npy
    data/mozilla_speaker_selection/habla_speaker_ids.json
    data/mozilla_speaker_selection/cv_speaker_embeddings.npy
    data/mozilla_speaker_selection/cv_speaker_metadata.json

Output:
    data/mozilla_speaker_selection/filtered_speakers.json: List of client_ids

Usage:
    python -m pipeline.select_mozilla_speakers.03_filter_by_similarity
"""
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


INPUT_DIR = Path("data/mozilla_speaker_selection")
HABLA_EMBEDDINGS = INPUT_DIR / "habla_embeddings.npy"
HABLA_IDS = INPUT_DIR / "habla_speaker_ids.json"
CV_EMBEDDINGS = INPUT_DIR / "cv_speaker_embeddings.npy"
CV_METADATA = INPUT_DIR / "cv_speaker_metadata.json"
FILTERED_OUTPUT = INPUT_DIR / "filtered_speakers.json"

SIMILARITY_THRESHOLD = 0.75


def compute_cosine_similarity(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two embedding matrices.

    Args:
        embeddings_a: Shape (N, D) - N embeddings of dimension D
        embeddings_b: Shape (M, D) - M embeddings of dimension D

    Returns:
        Similarity matrix of shape (N, M) with values in [-1, 1]
    """
    # Normalize embeddings (should already be normalized, but ensure)
    norm_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norm_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Cosine similarity via matrix multiplication
    similarity_matrix = norm_a @ norm_b.T

    return similarity_matrix


def filter_speakers_by_similarity(
    cv_embeddings: np.ndarray,
    habla_embeddings: np.ndarray,
    cv_metadata: dict,
    threshold: float
) -> list:
    """Filter CV speakers by maximum similarity to HABLA speakers.

    Args:
        cv_embeddings: Common Voice embeddings (N, 512)
        habla_embeddings: HABLA embeddings (162, 512)
        cv_metadata: Dict mapping client_id to metadata
        threshold: Maximum allowed cosine similarity

    Returns:
        List of client_ids that pass the filter
    """
    print(f"Computing cosine similarity matrix...")
    print(f"CV embeddings: {cv_embeddings.shape}")
    print(f"HABLA embeddings: {habla_embeddings.shape}")
    print()

    # Compute pairwise similarity: (N_cv, N_habla)
    similarity_matrix = compute_cosine_similarity(cv_embeddings, habla_embeddings)

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print()

    # For each CV speaker, find max similarity to any HABLA speaker
    max_similarities = similarity_matrix.max(axis=1)

    print(f"Max similarity statistics:")
    print(f"  Mean: {max_similarities.mean():.4f}")
    print(f"  Median: {np.median(max_similarities):.4f}")
    print(f"  Std: {max_similarities.std():.4f}")
    print(f"  Min: {max_similarities.min():.4f}")
    print(f"  Max: {max_similarities.max():.4f}")
    print()

    # Filter speakers below threshold
    cv_client_ids = list(cv_metadata.keys())
    filtered_speakers = []

    for i, client_id in enumerate(cv_client_ids):
        if max_similarities[i] < threshold:
            filtered_speakers.append(client_id)

    print(f"Filtering with threshold: {threshold}")
    print(f"  Original CV speakers: {len(cv_client_ids):,}")
    print(f"  Filtered (passed): {len(filtered_speakers):,}")
    print(f"  Rejected (too similar): {len(cv_client_ids) - len(filtered_speakers):,}")
    print(f"  Pass rate: {len(filtered_speakers) / len(cv_client_ids) * 100:.2f}%")

    return filtered_speakers


def main():
    """Filter Common Voice speakers by similarity to HABLA."""
    print("=" * 70)
    print("Filter Common Voice Speakers by Similarity to HABLA")
    print("=" * 70)
    print()

    # Load HABLA embeddings
    print(f"Loading HABLA embeddings from {HABLA_EMBEDDINGS}...")
    habla_embeddings = np.load(HABLA_EMBEDDINGS)

    with open(HABLA_IDS, "r", encoding="utf-8") as f:
        habla_ids = json.load(f)

    print(f"  HABLA speakers: {len(habla_ids)}")
    print(f"  Embedding shape: {habla_embeddings.shape}")
    print()

    # Load CV embeddings
    print(f"Loading CV embeddings from {CV_EMBEDDINGS}...")
    cv_embeddings = np.load(CV_EMBEDDINGS)

    with open(CV_METADATA, "r", encoding="utf-8") as f:
        cv_metadata = json.load(f)

    print(f"  CV speakers: {len(cv_metadata):,}")
    print(f"  Embedding shape: {cv_embeddings.shape}")
    print()

    # Filter by similarity
    filtered_speakers = filter_speakers_by_similarity(
        cv_embeddings,
        habla_embeddings,
        cv_metadata,
        SIMILARITY_THRESHOLD
    )

    print()
    print("=" * 70)
    print("Filtering Complete")
    print("=" * 70)
    print()

    # Save filtered speaker list
    print(f"Saving filtered speaker list to: {FILTERED_OUTPUT}")
    with open(FILTERED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(filtered_speakers, f, indent=2)

    print()
    print("Done!")
    print()
    print(f"Output: {FILTERED_OUTPUT}")
    print(f"Filtered speakers: {len(filtered_speakers):,}")


if __name__ == "__main__":
    main()
