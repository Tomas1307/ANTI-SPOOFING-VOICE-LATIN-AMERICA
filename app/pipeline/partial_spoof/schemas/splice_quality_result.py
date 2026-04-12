"""
Result schema for Step 6: Splice Quality Validation.
"""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class SpliceQualityResult(BaseModel):
    """Result from splice quality validation (Step 6).

    Validates spliced audio using Parakeet STT transcription, WER/CER
    against the original bonafide transcript, NISQA MOS quality estimation,
    and ECAPA-TDNN speaker similarity. Rejects samples that fail to preserve
    the original speech content or have zero spoofed words.

    Attributes:
        quality_path: Path to splice_quality_metadata.json output file.
        total_validated: Number of spliced samples that passed validation.
        total_rejected: Number of spliced samples rejected.
        rejected_samples: List of rejected sample details.
        avg_wer: Average WER across passed samples.
        avg_cer: Average CER across passed samples.
        avg_nisqa: Average NISQA MOS across passed samples.
        avg_speaker_similarity: Average ECAPA-TDNN cosine similarity.
        avg_spectral_flux: Average spectral flux at splice boundaries.
        avg_energy_delta: Average energy delta at splice boundaries.
    """

    quality_path: Path = Field(
        ...,
        description="Path to splice_quality_metadata.json output file",
    )
    total_validated: int = Field(
        ...,
        description="Number of spliced samples that passed validation",
    )
    total_rejected: int = Field(
        default=0,
        description="Number of spliced samples rejected",
    )
    rejected_samples: List[dict] = Field(
        default_factory=list,
        description="List of rejected sample details with reasons",
    )
    avg_wer: float = Field(
        default=0.0,
        description="Average WER across passed samples",
    )
    avg_cer: float = Field(
        default=0.0,
        description="Average CER across passed samples",
    )
    avg_nisqa: float = Field(
        default=0.0,
        description="Average NISQA MOS across passed samples",
    )
    avg_speaker_similarity: float = Field(
        default=0.0,
        description="Average ECAPA-TDNN cosine similarity across passed samples",
    )
    avg_spectral_flux: float = Field(
        default=0.0,
        description="Average spectral flux at splice boundaries (lower is smoother)",
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
