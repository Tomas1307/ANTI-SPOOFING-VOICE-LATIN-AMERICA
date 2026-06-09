"""
Result schema for Step 7: ASVspoof2019 LA Output Formatting.
"""
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class FormattingResult(BaseModel):
    """Result from ASVspoof2019 LA output formatting (Step 7).

    Attributes:
        output_directory: Path to LA/ directory with ASVspoof2019 structure.
        protocol_files: Paths to protocol files for train/dev/eval splits.
        total_samples: Sample counts per split.
    """

    output_directory: Path = Field(
        ...,
        description="Path to LA/ directory with ASVspoof2019 structure",
    )
    protocol_files: Dict[str, Path] = Field(
        ...,
        description="Paths to protocol files (e.g., {'train': Path(...), 'dev': Path(...)})",
    )
    total_samples: Dict[str, int] = Field(
        ...,
        description="Sample counts per split (e.g., {'train': 50, 'dev': 10, 'eval': 20})",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
