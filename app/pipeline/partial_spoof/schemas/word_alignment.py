"""Schema for a single word-level alignment entry."""
from pydantic import BaseModel, Field


class WordAlignment(BaseModel):
    """Represents the temporal alignment of a single word within an audio file.

    Produced by forced alignment engines (MMS_FA, Whisper, MFA) to map
    each word in a transcript to its start and end time in the audio.

    Attributes:
        word: The aligned word text.
        start_seconds: Start time of the word in seconds.
        end_seconds: End time of the word in seconds.
        confidence: Alignment confidence score (0.0 to 1.0).
    """

    word: str = Field(
        ...,
        description="The aligned word text.",
    )
    start_seconds: float = Field(
        ...,
        description="Start time of the word in seconds.",
    )
    end_seconds: float = Field(
        ...,
        description="End time of the word in seconds.",
    )
    confidence: float = Field(
        default=0.0,
        description="Alignment confidence score (0.0 to 1.0).",
    )
