"""
Schema for reference audio preparation result.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict


class ReferenceResult(BaseModel):
    """Result from reference audio preparation (Step 1).

    Attributes:
        reference_metadata_path: Path to JSON file with speaker -> reference mapping.
        reference_count: Number of speakers processed.
        split_breakdown: Counts per split (train/val/test).
        transcribed_count: Number of speakers with non-empty Parakeet ref_text.
    """

    reference_metadata_path: Path = Field(
        ...,
        description="Path to reference_metadata.json with speaker reference audio info"
    )
    reference_count: int = Field(
        ...,
        description="Total number of speakers processed"
    )
    split_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts per split (e.g., {'train': 110, 'val': 26, 'test': 26})"
    )
    transcribed_count: int = Field(
        default=0,
        description="Number of speakers whose reference audio was successfully transcribed"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True
