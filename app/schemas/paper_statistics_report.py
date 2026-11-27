"""
Pydantic schema for the paper-statistics report.
"""
from typing import Dict, List

from pydantic import BaseModel, Field


class SystemDurationWordStats(BaseModel):
    """Duration and word-count statistics for one full-spoof system.

    Attributes:
        system: System name as it should appear in the paper table.
        count: Utterances the statistics are computed over.
        mean_duration_s: Mean generated-audio duration, seconds.
        std_duration_s: Standard deviation of duration, seconds.
        mean_words: Mean target-text word count.
        std_words: Standard deviation of target-text word count.
    """

    system: str = Field(..., description="System name for the paper table.")
    count: int = Field(..., description="Utterances the statistics cover.")
    mean_duration_s: float = Field(..., description="Mean duration, seconds.")
    std_duration_s: float = Field(..., description="Std of duration, seconds.")
    mean_words: float = Field(..., description="Mean target-text word count.")
    std_words: float = Field(..., description="Std of target-text word count.")


class TierDurationStats(BaseModel):
    """Duration statistics for one corpus tier.

    Attributes:
        tier: Tier name (bonafide, full_spoof, partial_spoof).
        count: Utterances the statistics are computed over.
        mean_duration_s: Mean duration, seconds.
        std_duration_s: Standard deviation of duration, seconds.
    """

    tier: str = Field(..., description="Tier name.")
    count: int = Field(..., description="Utterances the statistics cover.")
    mean_duration_s: float = Field(..., description="Mean duration, seconds.")
    std_duration_s: float = Field(..., description="Std of duration, seconds.")


class PaperStatisticsReport(BaseModel):
    """All three statistics the main.tex Technical Validation pass needs.

    Attributes:
        accent_utterance_counts: Bonafide utterance count per accent code.
        tier_duration_stats: Duration mean/std per corpus tier.
        system_stats: Duration and word-count mean/std per full-spoof system.
    """

    accent_utterance_counts: Dict[str, int] = Field(
        default_factory=dict, description="Bonafide utterance count per accent."
    )
    tier_duration_stats: List[TierDurationStats] = Field(
        default_factory=list, description="Duration stats per corpus tier."
    )
    system_stats: List[SystemDurationWordStats] = Field(
        default_factory=list, description="Duration/word stats per full-spoof system."
    )
