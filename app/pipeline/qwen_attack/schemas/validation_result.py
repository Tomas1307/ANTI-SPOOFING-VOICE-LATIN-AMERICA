"""
Schema for quality validation result.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Dict, List


class ValidationResult(BaseModel):
    """Result from quality validation (Step 4).

    Includes Qwen-specific artifact detection results alongside
    standard DNSMOS and speaker similarity metrics.

    Attributes:
        validated_samples_path: Path to JSON file with validated samples only.
        validation_stats: Pass/fail counts by metric.
        rejected_samples: List of rejected sample IDs with reasons and scores.
        avg_dnsmos: Average DNSMOS score across validated samples.
        avg_similarity: Average speaker similarity across validated samples.
    """

    validated_samples_path: Path = Field(
        ...,
        description="Path to validated_samples.json (only PASS samples)"
    )
    validation_stats: Dict[str, int] = Field(
        ...,
        description="Pass/fail counts (e.g., {'passed': 5, 'rejected': 1})"
    )
    rejected_samples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of rejected samples with IDs, scores, and rejection reasons"
    )
    avg_dnsmos: float = Field(
        default=0.0,
        description="Average DNSMOS overall score across validated samples"
    )
    avg_similarity: float = Field(
        default=0.0,
        description="Average speaker similarity score across validated samples"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True
