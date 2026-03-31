"""Result schema for Step 1: Bonafide audio transcription."""
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Result from bonafide audio transcription using Parakeet TDT.

    Attributes:
        transcripts_path: Path to the bonafide_transcripts.json output file.
        total_transcribed: Number of utterances successfully transcribed.
        skipped_short: Number of utterances filtered out for having fewer
            words than the minimum W1 threshold.
        word_count_distribution: Distribution of transcripts by word count
            bucket (e.g., {"4-7": 50, "8-11": 30, "12+": 20}).
    """

    transcripts_path: Path = Field(
        ...,
        description="Path to bonafide_transcripts.json.",
    )
    total_transcribed: int = Field(
        ...,
        description="Number of utterances successfully transcribed.",
    )
    skipped_short: int = Field(
        ...,
        description="Utterances filtered for insufficient word count.",
    )
    word_count_distribution: Dict[str, int] = Field(
        ...,
        description="Transcript count per word-count bucket.",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
