"""
Result schema for Step 1: Bonafide Audio Transcription.
"""
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    """Result from bonafide audio transcription (Step 1).

    Attributes:
        transcripts_path: Path to bonafide_transcripts.json output file.
        total_transcribed: Total number of utterances successfully transcribed.
        skipped_short: Number of utterances skipped due to insufficient word count.
        word_count_distribution: Distribution of word counts across buckets.
    """

    transcripts_path: Path = Field(
        ...,
        description="Path to bonafide_transcripts.json output file",
    )
    total_transcribed: int = Field(
        ...,
        description="Total number of utterances successfully transcribed",
    )
    skipped_short: int = Field(
        ...,
        description="Number of utterances skipped (fewer than MIN_WORDS_W1 words)",
    )
    word_count_distribution: Dict[str, int] = Field(
        ...,
        description="Word count distribution buckets (e.g., '4-7': 50, '8-11': 30, '12+': 20)",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
