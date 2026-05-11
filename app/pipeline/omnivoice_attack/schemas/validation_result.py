"""
Schema for quality validation result from Step 4.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Dict, List


class ValidationResult(BaseModel):
    """Result from quality validation (Step 4).

    Includes silence detection results, Parakeet TDT WER/CER transcription
    accuracy metrics, NISQA speech quality MOS estimation, and ECAPA-TDNN
    speaker similarity scoring.

    Attributes:
        validated_samples_path: Path to JSON file with validated samples only.
        validation_stats: Pass/fail counts and totals.
        rejected_samples: List of rejected sample metadata with reasons.
        avg_wer: Average Word Error Rate across validated samples.
        avg_cer: Average Character Error Rate across validated samples.
        prefix_trim_count: Number of samples that had a spurious prefix trimmed.
        avg_nisqa: Average NISQA MOS score across validated samples.
        avg_speaker_similarity: Average ECAPA-TDNN cosine similarity.
    """

    validated_samples_path: Path = Field(
        ...,
        description="Path to validated_samples.json (only PASS samples)"
    )
    validation_stats: Dict[str, int] = Field(
        ...,
        description="Pass/fail counts (e.g., {'passed': 5, 'rejected': 1, 'total': 6})"
    )
    rejected_samples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of rejected samples with IDs, scores, and rejection reasons"
    )
    avg_wer: float = Field(
        default=0.0,
        description="Average Word Error Rate across validated samples (0.0 = perfect)"
    )
    avg_cer: float = Field(
        default=0.0,
        description="Average Character Error Rate across validated samples (0.0 = perfect)"
    )
    prefix_trim_count: int = Field(
        default=0,
        description="Number of samples where a spurious prefix was detected and trimmed"
    )
    nonverbal_prefix_rejection_count: int = Field(
        default=0,
        description=(
            "Number of samples rejected because pre-speech RMS exceeded the "
            "NONVERBAL_PREFIX_RMS_FLOOR_DB floor (reference voice bleed, breath, "
            "click). These samples are sent to the retry loop for regeneration."
        )
    )
    avg_nisqa: float = Field(
        default=0.0,
        description="Average NISQA MOS score across validated samples (1.0-5.0 scale)"
    )
    avg_speaker_similarity: float = Field(
        default=0.0,
        description="Average ECAPA-TDNN cosine similarity between reference and generated audio"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True
