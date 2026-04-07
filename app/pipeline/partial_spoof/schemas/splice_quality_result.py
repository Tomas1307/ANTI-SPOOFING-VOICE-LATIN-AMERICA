"""
Result schema for Step 6: Splice Quality Validation.
"""
from pathlib import Path

from pydantic import BaseModel, Field


class SpliceQualityResult(BaseModel):
    """Result from splice quality validation (Step 6).

    Currently a placeholder that computes metrics without rejecting samples.
    Thresholds for rejection and retry logic are TBD.

    Attributes:
        quality_path: Path to splice_quality_metadata.json output file.
        total_validated: Number of spliced samples with computed quality metrics.
        avg_spectral_flux: Average spectral flux at splice boundaries.
        avg_f0_delta: Average F0 (pitch) delta at splice boundaries in Hz.
        avg_energy_delta: Average energy (RMS) delta at splice boundaries.
        retry_count: Number of retries triggered by quality failures (placeholder, always 0).
    """

    quality_path: Path = Field(
        ...,
        description="Path to splice_quality_metadata.json output file",
    )
    total_validated: int = Field(
        ...,
        description="Number of spliced samples with computed quality metrics",
    )
    avg_spectral_flux: float = Field(
        default=0.0,
        description="Average spectral flux at splice boundaries (lower is smoother)",
    )
    avg_f0_delta: float = Field(
        default=0.0,
        description="Average F0 (pitch) delta at splice boundaries in Hz (placeholder, returns 0.0)",
    )
    avg_energy_delta: float = Field(
        default=0.0,
        description="Average energy (RMS) delta at splice boundaries",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries triggered by quality failures (placeholder)",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
