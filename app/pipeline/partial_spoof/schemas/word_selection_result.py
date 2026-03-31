"""Result schema for Step 4: Word selection."""
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class WordSelectionResult(BaseModel):
    """Result from random word selection for partial spoofing.

    Attributes:
        selection_path: Path to word_selection_metadata.json.
        total_selections: Total number of selection plans generated
            across all utterances and tiers.
        tier_counts: Number of selection plans per tier
            (e.g., {"W1": 100, "W2": 80, "W3": 40}).
    """

    selection_path: Path = Field(
        ...,
        description="Path to word_selection_metadata.json.",
    )
    total_selections: int = Field(
        ...,
        description="Total selection plans across all tiers.",
    )
    tier_counts: Dict[str, int] = Field(
        ...,
        description="Selection plan count per tier.",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
