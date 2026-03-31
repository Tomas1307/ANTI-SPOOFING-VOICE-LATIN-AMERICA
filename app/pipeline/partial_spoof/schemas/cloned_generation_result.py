"""Result schema for Step 2: Cloned speech generation."""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class ClonedGenerationResult(BaseModel):
    """Result from voice-cloned speech generation.

    Attributes:
        metadata_path: Path to cloned_generation_metadata.json.
        total_generated: Number of cloned utterances successfully generated.
        failed_generations: List of sample identifiers that failed generation.
        avg_rtf: Average real-time factor across all generated samples.
    """

    metadata_path: Path = Field(
        ...,
        description="Path to cloned_generation_metadata.json.",
    )
    total_generated: int = Field(
        ...,
        description="Utterances successfully generated.",
    )
    failed_generations: List[str] = Field(
        default_factory=list,
        description="Sample IDs that failed generation.",
    )
    avg_rtf: float = Field(
        default=0.0,
        description="Average real-time factor (generation_time / audio_duration).",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
