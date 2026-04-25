"""
Schema for clone similarity pre-filter results.
"""
from typing import List
from pydantic import BaseModel, Field


class SimilarityFilterResult(BaseModel):
    """Results from the ECAPA-TDNN clone similarity gate.

    Records which clones passed or failed the minimum cosine similarity
    threshold between the bonafide reference and the TTS-generated clone.

    Attributes:
        total_evaluated: Number of clones evaluated.
        total_passed: Number of clones that passed the threshold.
        total_rejected: Number of clones rejected for low similarity.
        avg_similarity: Mean cosine similarity across all evaluated clones.
        min_similarity_threshold: The threshold used for filtering.
        rejected_keys: Sample keys of rejected clones.
    """

    total_evaluated: int = Field(
        ...,
        description="Number of clones evaluated by the similarity gate",
    )
    total_passed: int = Field(
        ...,
        description="Number of clones that passed the similarity threshold",
    )
    total_rejected: int = Field(
        ...,
        description="Number of clones rejected for low speaker similarity",
    )
    avg_similarity: float = Field(
        ...,
        description="Mean ECAPA-TDNN cosine similarity across all evaluated clones",
    )
    min_similarity_threshold: float = Field(
        ...,
        description="The cosine similarity threshold used for filtering",
    )
    rejected_keys: List[str] = Field(
        default_factory=list,
        description="Sample keys of clones rejected by the gate",
    )
