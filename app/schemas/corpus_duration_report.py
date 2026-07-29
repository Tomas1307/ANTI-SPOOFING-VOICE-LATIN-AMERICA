"""
Pydantic schema for the full corpus duration/description statistics report.
"""
from typing import List

from pydantic import BaseModel, Field

from app.schemas.corpus_tier_duration_stat import TierDurationStat
from app.schemas.fullspoof_system_description_stat import SystemDescriptionStat


class CorpusDurationReport(BaseModel):
    """Corpus-wide duration and text-length statistics.

    Covers the three base tiers (bonafide, full-spoof, partial-spoof).
    Augmentation tiers are not scanned directly: augmentation does not
    change utterance duration (RIR/noise addition and codec re-encoding
    preserve length), so their hours can be derived analytically as
    base-tier hours multiplied by the augmentation factor.

    Attributes:
        per_tier: One TierDurationStat per base corpus tier.
        per_fullspoof_system: One SystemDescriptionStat per of the six
            full-spoof TTS systems.
        base_corpus_total_hours: Sum of total_hours across per_tier.
        notes: Caveats and methodology notes to carry into the paper text.
    """

    per_tier: List[TierDurationStat] = Field(default_factory=list)
    per_fullspoof_system: List[SystemDescriptionStat] = Field(default_factory=list)
    base_corpus_total_hours: float = Field(..., ge=0.0)
    notes: List[str] = Field(default_factory=list)
