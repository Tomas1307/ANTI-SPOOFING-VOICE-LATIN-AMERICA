"""
Schema for reference audio preparation result from Step 1.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict


class ReferenceResult(BaseModel):
    """Result from reference audio preparation (Step 1).

    Attributes:
        reference_metadata_path: Path to JSON file with speaker reference mappings.
        reference_count: Number of speakers successfully processed.
        split_breakdown: Sample counts per dataset split (train/val/test).
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

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True
