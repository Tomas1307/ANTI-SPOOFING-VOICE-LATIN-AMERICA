"""Result schema for Step 3: Forced alignment."""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class AlignmentResult(BaseModel):
    """Result from forced alignment on bonafide and cloned audio pairs.

    Attributes:
        alignment_path: Path to alignment_metadata.json with word-level
            timestamps for both bonafide and cloned versions.
        total_aligned: Number of utterance pairs successfully aligned.
        failed_alignments: List of sample IDs where alignment failed.
        avg_words_per_utterance: Average word count across aligned utterances.
    """

    alignment_path: Path = Field(
        ...,
        description="Path to alignment_metadata.json.",
    )
    total_aligned: int = Field(
        ...,
        description="Utterance pairs successfully aligned.",
    )
    failed_alignments: List[str] = Field(
        default_factory=list,
        description="Sample IDs where alignment failed.",
    )
    avg_words_per_utterance: float = Field(
        default=0.0,
        description="Average word count per aligned utterance.",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
