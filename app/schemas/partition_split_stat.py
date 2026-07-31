"""
Pydantic schema for per-split statistics of the speaker-disjoint partition.
"""
from pydantic import BaseModel, Field


class PartitionSplitStat(BaseModel):
    """Counts for one split (train/dev/eval) of the speaker-disjoint partition.

    Attributes:
        split: Split name ('train', 'dev', or 'eval').
        speaker_count: Number of speakers assigned to this split.
        bonafide_count: Number of bonafide symlinks created.
        fullspoof_count: Number of full-spoof symlinks created.
        partialspoof_count: Number of partial-spoof symlinks created.
    """

    split: str = Field(..., description="Split name: train, dev, or eval")
    speaker_count: int = Field(..., ge=0)
    bonafide_count: int = Field(..., ge=0)
    fullspoof_count: int = Field(..., ge=0)
    partialspoof_count: int = Field(..., ge=0)
