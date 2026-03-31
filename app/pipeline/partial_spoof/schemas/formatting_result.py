"""Result schema for Step 7: ASVspoof2019 LA output formatting."""
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class FormattingResult(BaseModel):
    """Result from ASVspoof2019 LA output formatting.

    Attributes:
        output_directory: Path to the LA/ directory containing the
            ASVspoof2019 standard structure.
        protocol_files: Mapping of split names to their protocol file paths
            (e.g., {"train": Path(...), "dev": Path(...), "eval": Path(...)}).
        total_samples: Sample counts per split
            (e.g., {"train": 40, "dev": 10, "eval": 10}).
    """

    output_directory: Path = Field(
        ...,
        description="Path to LA/ directory with ASVspoof2019 structure.",
    )
    protocol_files: Dict[str, Path] = Field(
        ...,
        description="Protocol file paths keyed by split name.",
    )
    total_samples: Dict[str, int] = Field(
        ...,
        description="Sample counts per split.",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
