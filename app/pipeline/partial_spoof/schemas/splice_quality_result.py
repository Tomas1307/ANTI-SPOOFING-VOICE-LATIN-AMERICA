"""Result schema for Step 6: Splice quality validation."""
from pathlib import Path

from pydantic import BaseModel, Field


class SpliceQualityResult(BaseModel):
    """Result from splice boundary quality validation.

    This step is currently a placeholder that computes and logs metrics
    without rejecting samples. Future versions will enable retry logic
    based on configurable thresholds.

    Attributes:
        quality_path: Path to splice_quality_metadata.json.
        total_validated: Number of spliced samples evaluated.
        avg_spectral_flux: Average spectral flux at splice boundaries.
        avg_f0_delta: Average fundamental frequency delta in Hz at boundaries.
        avg_energy_delta: Average RMS energy delta at boundaries.
        retry_count: Number of retries triggered (0 when retry is disabled).
    """

    quality_path: Path = Field(
        ...,
        description="Path to splice_quality_metadata.json.",
    )
    total_validated: int = Field(
        ...,
        description="Number of spliced samples evaluated.",
    )
    avg_spectral_flux: float = Field(
        default=0.0,
        description="Average spectral flux at splice boundaries.",
    )
    avg_f0_delta: float = Field(
        default=0.0,
        description="Average F0 delta in Hz at splice boundaries.",
    )
    avg_energy_delta: float = Field(
        default=0.0,
        description="Average RMS energy delta at splice boundaries.",
    )
    retry_count: int = Field(
        default=0,
        description="Total retries triggered across all samples.",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
