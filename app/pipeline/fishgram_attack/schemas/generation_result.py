"""
Schema for speech generation result.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List


class GenerationResult(BaseModel):
    """Result from synthetic speech generation (Step 4).

    Attributes:
        generated_samples_path: Path to JSON file with generation metadata
        total_generated: Total number of samples successfully generated
        failed_generations: List of failed sample IDs with error messages
        avg_rtf: Average real-time factor across all generations
    """

    generated_samples_path: Path = Field(
        ...,
        description="Path to generation_metadata.json with all generated sample info"
    )
    total_generated: int = Field(
        ...,
        description="Total number of samples successfully generated"
    )
    failed_generations: List[str] = Field(
        default_factory=list,
        description="List of failed sample IDs with error messages"
    )
    avg_rtf: float = Field(
        default=0.0,
        description="Average real-time factor (generation_time / audio_duration)"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True
