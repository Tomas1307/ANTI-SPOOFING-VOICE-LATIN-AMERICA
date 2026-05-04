"""
Schema for boundary jitter step result (Step 5b).
"""
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field


class BoundaryJitterResult(BaseModel):
    """Result from boundary jitter post-processing of spliced audio (Step 5b).

    Step 5b takes the spliced audio produced by Step 5 and applies random
    structural manipulations (truncate, overlap, bleed) at every internal
    word boundary in each utterance. The goal is to homogenize boundary
    artifacts so the splice boundary does not stand out as the only
    manipulated boundary.

    Attributes:
        jitter_metadata_path: Path to the JSON file with per-utterance
            jitter plans (which boundary received which manipulation).
        total_processed: Number of utterances that received jitter processing.
        total_skipped: Number of utterances skipped (e.g., audio missing).
        total_boundaries_seen: Total internal word boundaries across all
            processed utterances.
        operation_counts: Counts of each manipulation type applied
            (truncate, overlap, bleed, none).
        avg_duration_drift_ms: Average absolute duration drift across all
            utterances (positive = audio grew, negative = shrank). Useful
            for monitoring whether jitter is preserving rhythm.
    """

    jitter_metadata_path: Path = Field(
        ...,
        description="Path to boundary_jitter_metadata.json with per-utterance jitter plans",
    )
    total_processed: int = Field(
        ...,
        description="Number of utterances that received jitter processing",
    )
    total_skipped: int = Field(
        default=0,
        description="Number of utterances skipped (audio file missing, etc.)",
    )
    total_boundaries_seen: int = Field(
        default=0,
        description="Total internal word boundaries across all processed utterances",
    )
    operation_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of each manipulation applied "
                    "(e.g. {'truncate': 120, 'overlap': 130, 'bleed': 110, 'none': 480})",
    )
    avg_duration_drift_ms: float = Field(
        default=0.0,
        description="Mean absolute duration drift across utterances in milliseconds. "
                    "Positive values indicate audio grew on average; negative indicate shrinkage.",
    )
    failed_utterances: List[str] = Field(
        default_factory=list,
        description="Sample IDs that failed jitter processing (for debugging).",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
