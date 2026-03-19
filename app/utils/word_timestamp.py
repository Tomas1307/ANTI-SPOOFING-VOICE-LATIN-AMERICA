"""
Pydantic schema for a single word with timing boundaries from ASR output.
"""
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """A single word with its temporal boundaries as returned by Parakeet TDT.

    Attributes:
        word: The transcribed word string.
        start: Word start time in seconds from the beginning of the audio.
        end: Word end time in seconds from the beginning of the audio.
    """

    word: str = Field(..., description="Transcribed word string")
    start: float = Field(..., description="Word start time in seconds")
    end: float = Field(..., description="Word end time in seconds")
