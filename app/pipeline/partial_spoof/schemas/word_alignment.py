"""
Schema for a single word-level alignment entry.
"""
from pydantic import BaseModel, Field


class WordAlignment(BaseModel):
    """Timestamp information for a single aligned word.

    Produced by forced alignment on audio with a known transcript. Each
    instance represents one word and its start/end positions in seconds.

    Attributes:
        word: The aligned word text (lowercase, normalized).
        start_seconds: Start time of the word in the audio (seconds).
        end_seconds: End time of the word in the audio (seconds).
        confidence: Alignment confidence score (0.0 to 1.0, engine-dependent).
    """

    word: str = Field(
        ...,
        description="The aligned word text (lowercase, normalized)",
    )
    start_seconds: float = Field(
        ...,
        description="Start time of the word in the audio (seconds)",
    )
    end_seconds: float = Field(
        ...,
        description="End time of the word in the audio (seconds)",
    )
    confidence: float = Field(
        default=0.0,
        description="Alignment confidence score (0.0 to 1.0, engine-dependent)",
    )
