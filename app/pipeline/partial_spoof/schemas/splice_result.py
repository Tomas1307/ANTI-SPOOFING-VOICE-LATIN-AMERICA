"""Result schema for Step 5: Audio splicing."""
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class SpliceResult(BaseModel):
    """Result from audio splicing of cloned word segments into bonafide audio.

    Attributes:
        metadata_path: Path to splice_metadata.json with per-sample details.
        total_spliced: Number of partially spoofed samples produced.
        failed_splices: List of sample IDs where splicing failed.
        avg_spoof_duration_ratio: Average ratio of spoofed audio duration
            to total audio duration across all samples.
        tier_counts: Number of spliced samples per tier.
    """

    metadata_path: Path = Field(
        ...,
        description="Path to splice_metadata.json.",
    )
    total_spliced: int = Field(
        ...,
        description="Partially spoofed samples produced.",
    )
    failed_splices: List[str] = Field(
        default_factory=list,
        description="Sample IDs where splicing failed.",
    )
    avg_spoof_duration_ratio: float = Field(
        default=0.0,
        description="Average spoofed-duration / total-duration ratio.",
    )
    tier_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Spliced sample count per tier.",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
