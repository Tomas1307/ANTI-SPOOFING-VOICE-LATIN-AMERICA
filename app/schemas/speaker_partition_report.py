"""
Pydantic schema for the full speaker-disjoint partition report.
"""
from typing import List

from pydantic import BaseModel, Field

from app.schemas.partition_split_stat import PartitionSplitStat


class SpeakerPartitionReport(BaseModel):
    """Report of a speaker-disjoint train/dev/eval partition build.

    Attributes:
        per_split: One PartitionSplitStat per split.
        total_speakers: Total number of unique speakers partitioned.
        speakers_missing_bonafide: Speaker IDs found in full-spoof or
            partial-spoof sources but with no bonafide directory.
        notes: Caveats and methodology notes.
    """

    per_split: List[PartitionSplitStat] = Field(default_factory=list)
    total_speakers: int = Field(..., ge=0)
    speakers_missing_bonafide: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
