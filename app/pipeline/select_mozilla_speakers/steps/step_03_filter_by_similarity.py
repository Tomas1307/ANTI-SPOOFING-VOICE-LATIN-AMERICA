"""
Step 3: Filter Common Voice speakers by similarity to HABLA speakers.

This step loads embeddings from both datasets and filters out CV speakers
that are too similar to existing HABLA speakers.
"""
import json
from loguru import logger
from pathlib import Path
from typing import List

import numpy as np

from app.pipeline.select_mozilla_speakers.settings import settings


class SimilarityFilter:
    """Filters CV speakers by cosine similarity to HABLA speakers.

    This step ensures speaker independence between CV and HABLA datasets by
    rejecting CV speakers with cosine similarity >= threshold to any HABLA speaker.

    Attributes:
        threshold: Cosine similarity threshold for filtering
        input_dir: Directory containing embedding files
    """

    def __init__(
        self,
        threshold: float | None = None,
        input_dir: Path | None = None
    ) -> None:
        """Initialize the similarity filter.

        Args:
            threshold: Optional override for similarity threshold
            input_dir: Optional override for input directory
        """
        self.threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
        self.input_dir = input_dir or settings.OUTPUT_DIR

    def compute_cosine_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of embeddings.

        Args:
            embeddings_a: First set of embeddings (N, dim)
            embeddings_b: Second set of embeddings (M, dim)

        Returns:
            Similarity matrix (N, M)
        """
        # L2-normalize
        norm_a = embeddings_a / np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        norm_b = embeddings_b / np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        # Cosine similarity = dot product of normalized vectors
        similarity_matrix = norm_a @ norm_b.T

        return similarity_matrix

    def filter_speakers_by_similarity(
        self,
        cv_embeddings: np.ndarray,
        habla_embeddings: np.ndarray,
        cv_client_ids: List[str]
    ) -> List[str]:
        """Filter CV speakers by maximum similarity to HABLA speakers.

        Args:
            cv_embeddings: CV speaker embeddings (N, 192)
            habla_embeddings: HABLA speaker embeddings (M, 192)
            cv_client_ids: List of CV client IDs

        Returns:
            List of client IDs that passed the filter
        """
        logger.info("Computing cosine similarity matrix...")
        logger.info(f"CV embeddings: {cv_embeddings.shape}")
        logger.info(f"HABLA embeddings: {habla_embeddings.shape}")

        # Compute pairwise similarity: (N_cv, N_habla)
        logger.info(f"Computing cosine similarity matrix ({cv_embeddings.shape[0]:,} × {habla_embeddings.shape[0]} = {cv_embeddings.shape[0] * habla_embeddings.shape[0]:,} comparisons)...")
        similarity_matrix = self.compute_cosine_similarity(cv_embeddings, habla_embeddings)
        logger.info("✓ Similarity computation complete")

        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")

        # For each CV speaker, find max similarity to any HABLA speaker
        max_similarities = similarity_matrix.max(axis=1)

        logger.info("Max similarity statistics:")
        logger.info(f"  Mean: {max_similarities.mean():.4f}")
        logger.info(f"  Median: {np.median(max_similarities):.4f}")
        logger.info(f"  Std: {max_similarities.std():.4f}")
        logger.info(f"  Min: {max_similarities.min():.4f}")
        logger.info(f"  Max: {max_similarities.max():.4f}")

        # Filter speakers below threshold
        filtered_speakers = []
        for i, client_id in enumerate(cv_client_ids):
            if max_similarities[i] < self.threshold:
                filtered_speakers.append(client_id)

        logger.info(f"Filtering with threshold: {self.threshold}")
        logger.info(f"  Original CV speakers: {len(cv_client_ids):,}")
        logger.info(f"  Filtered (passed): {len(filtered_speakers):,}")
        logger.info(f"  Rejected (too similar): {len(cv_client_ids) - len(filtered_speakers):,}")
        logger.info(f"  Pass rate: {len(filtered_speakers) / len(cv_client_ids) * 100:.2f}%")

        return filtered_speakers

    def execute(self) -> Path:
        """Execute similarity-based filtering.

        Returns:
            Path to filtered_speakers.json
        """
        logger.info("Step 03 - Similarity Filtering: Starting")

        # Load HABLA embeddings
        habla_embeddings_path = self.input_dir / "habla_embeddings.npy"
        habla_ids_path = self.input_dir / "habla_speaker_ids.json"

        logger.info(f"Loading HABLA embeddings from {habla_embeddings_path}...")
        habla_embeddings = np.load(habla_embeddings_path)

        with open(habla_ids_path, "r", encoding="utf-8") as f:
            habla_ids = json.load(f)

        logger.info(f"  HABLA speakers: {len(habla_ids)}")
        logger.info(f"  Embedding shape: {habla_embeddings.shape}")

        # Load CV embeddings
        cv_embeddings_path = self.input_dir / "cv_speaker_embeddings.npy"
        cv_metadata_path = self.input_dir / "cv_speaker_metadata.json"

        logger.info(f"Loading CV embeddings from {cv_embeddings_path}...")
        cv_embeddings = np.load(cv_embeddings_path)

        with open(cv_metadata_path, "r", encoding="utf-8") as f:
            cv_metadata = json.load(f)

        cv_client_ids = list(cv_metadata.keys())

        logger.info(f"  CV speakers: {len(cv_client_ids)}")
        logger.info(f"  Embedding shape: {cv_embeddings.shape}")

        # Filter by similarity
        filtered_speakers = self.filter_speakers_by_similarity(
            cv_embeddings,
            habla_embeddings,
            cv_client_ids
        )

        # Save results
        output_path = self.input_dir / "filtered_speakers.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_speakers, f, indent=2)

        logger.info(f"Saved filtered speaker list to: {output_path}")
        logger.info(f"Filtered speakers: {len(filtered_speakers):,}")
        logger.info("Step 03 - Similarity Filtering: Complete")

        return output_path
