"""
Pydantic schema for per-full-spoof-system description statistics.
"""
from pydantic import BaseModel, Field


class SystemDescriptionStat(BaseModel):
    """Duration and text-length statistics for one full-spoof TTS system.

    Companion to the technical-validation WER/NISQA/SIM table: this
    describes what was generated (volume, length), not how good it is.

    Attributes:
        system: Display name matching the paper's system tables
            (e.g. 'Fish-Speech (FishGram)').
        utterance_count: Number of validated (post-screening) utterances.
        avg_duration_seconds: Mean generated-audio duration, in seconds.
        avg_word_count: Mean word count of the synthesized target text.
        total_hours: Summed generated-audio duration, in hours.
    """

    system: str = Field(..., description="Display name of the TTS system")
    utterance_count: int = Field(..., ge=0)
    avg_duration_seconds: float = Field(..., ge=0.0)
    avg_word_count: float = Field(..., ge=0.0)
    total_hours: float = Field(..., ge=0.0)
