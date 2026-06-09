"""
Schema for embedding extraction results.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
import numpy.typing as npt


class EmbeddingResult(BaseModel):
    """Result from embedding extraction step.

    Attributes:
        embeddings_path: Path to .npy file containing embeddings
        ids_path: Path to .json file containing speaker/client IDs
        metadata_path: Optional path to metadata JSON file
        embedding_count: Number of embeddings extracted
        embedding_dim: Dimension of each embedding vector
    """

    embeddings_path: Path = Field(
        ...,
        description="Path to .npy file with shape (N, embedding_dim)"
    )
    ids_path: Path = Field(
        ...,
        description="Path to JSON file with list of speaker/client IDs"
    )
    metadata_path: Path | None = Field(
        default=None,
        description="Optional path to metadata JSON file"
    )
    embedding_count: int = Field(
        ...,
        description="Number of embeddings extracted"
    )
    embedding_dim: int = Field(
        ...,
        description="Dimension of each embedding vector (e.g., 192 for ECAPA-TDNN)"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True
