"""
Result schema for Step 5: Audio Splicing.
"""
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class SpliceResult(BaseModel):
    """Result from audio splicing operation (Step 5).

    Attributes:
        metadata_path: Path to splice_metadata.json output file.
        total_spliced: Total number of partially spoofed samples created.
        failed_splices: List of sample IDs where splicing failed.
        avg_spoof_duration_ratio: Average spoof duration ratio across all samples.
        tier_counts: Number of spliced samples per tier.
    """

    metadata_path: Path = Field(
        ...,
        description="Path to splice_metadata.json output file",
    )
    total_spliced: int = Field(
        ...,
        description="Total number of partially spoofed samples created",
    )
    failed_splices: List[str] = Field(
        default_factory=list,
        description="List of sample IDs where splicing failed",
    )
    avg_spoof_duration_ratio: float = Field(
        default=0.0,
        description="Average fraction of audio duration that is synthetic",
    )
    tier_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of spliced samples per tier (e.g., {'W1': 100, 'W2': 60, 'W3': 30})",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
