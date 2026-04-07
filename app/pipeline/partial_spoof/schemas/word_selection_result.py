"""
Result schema for Step 4: Word Selection.
"""
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class WordSelectionResult(BaseModel):
    """Result from word selection for partial spoofing (Step 4).

    Attributes:
        selection_path: Path to word_selection_metadata.json output file.
        total_selections: Total number of selection plans generated across all tiers.
        tier_counts: Number of selection plans per tier.
    """

    selection_path: Path = Field(
        ...,
        description="Path to word_selection_metadata.json output file",
    )
    total_selections: int = Field(
        ...,
        description="Total number of selection plans generated across all tiers",
    )
    tier_counts: Dict[str, int] = Field(
        ...,
        description="Number of selection plans per tier (e.g., {'W1': 100, 'W2': 60, 'W3': 30})",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
