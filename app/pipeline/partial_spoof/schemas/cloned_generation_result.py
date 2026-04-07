"""
Result schema for Step 2: Cloned Speech Generation.
"""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class ClonedGenerationResult(BaseModel):
    """Result from cloned speech generation (Step 2).

    Attributes:
        metadata_path: Path to cloned_generation_metadata.json output file.
        total_generated: Number of utterances successfully cloned.
        failed_generations: List of sample IDs that failed generation.
        avg_rtf: Average real-time factor across all generated samples.
    """

    metadata_path: Path = Field(
        ...,
        description="Path to cloned_generation_metadata.json output file",
    )
    total_generated: int = Field(
        ...,
        description="Number of utterances successfully cloned",
    )
    failed_generations: List[str] = Field(
        default_factory=list,
        description="List of sample IDs that failed generation",
    )
    avg_rtf: float = Field(
        default=0.0,
        description="Average real-time factor (generation_time / audio_duration)",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
