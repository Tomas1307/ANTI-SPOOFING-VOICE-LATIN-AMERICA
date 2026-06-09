"""
Result schema for Step 3: Forced Alignment.
"""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class AlignmentResult(BaseModel):
    """Result from forced alignment on bonafide and cloned audio (Step 3).

    Attributes:
        alignment_path: Path to alignment_metadata.json output file.
        total_aligned: Number of utterance pairs successfully aligned.
        failed_alignments: List of sample IDs where alignment failed.
        avg_words_per_utterance: Average word count across aligned utterances.
    """

    alignment_path: Path = Field(
        ...,
        description="Path to alignment_metadata.json output file",
    )
    total_aligned: int = Field(
        ...,
        description="Number of bonafide-cloned utterance pairs successfully aligned",
    )
    failed_alignments: List[str] = Field(
        default_factory=list,
        description="List of sample IDs where alignment failed on either side",
    )
    avg_words_per_utterance: float = Field(
        default=0.0,
        description="Average word count across all aligned utterances",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
