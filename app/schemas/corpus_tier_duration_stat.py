"""
Pydantic schema for per-tier corpus duration and text-length statistics.
"""
from typing import Optional

from pydantic import BaseModel, Field


class TierDurationStat(BaseModel):
    """Duration and text-length statistics for one corpus tier.

    Attributes:
        tier: Tier label (e.g. 'Bonafide', 'Full spoof', 'Partial spoof').
        utterance_count: Number of utterances the statistics were computed over.
        total_hours: Summed audio duration for the tier, in hours.
        avg_duration_seconds: Mean per-utterance audio duration, in seconds.
        avg_word_count: Mean word count of the associated text, if applicable.
            None for tiers where a per-utterance text is not meaningful.
    """

    tier: str = Field(..., description="Tier label")
    utterance_count: int = Field(..., ge=0)
    total_hours: float = Field(..., ge=0.0)
    avg_duration_seconds: float = Field(..., ge=0.0)
    avg_word_count: Optional[float] = Field(default=None, ge=0.0)
