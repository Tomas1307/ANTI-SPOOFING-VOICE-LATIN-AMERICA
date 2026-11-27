"""
Pydantic schema describing one materialised dataset split.
"""
from typing import Dict, List

from pydantic import BaseModel, Field

from app.pipeline.training.schemas.protocol_entry import ProtocolEntry


class DatasetSplit(BaseModel):
    """A split of the corpus, resolved to on-disk clips.

    Attributes:
        name: Split name, one of ``train``, ``dev`` or ``eval``.
        flac_dir: Directory holding the split's FLAC clips.
        entries: Protocol rows retained for this split.
        speaker_count: Distinct speakers present.
        attack_ids: Distinct attack identifiers present, excluding bonafide.
        label_counts: Row counts keyed by ``bonafide`` and ``spoof``.
    """

    name: str = Field(..., description="Split name: train, dev or eval.")
    flac_dir: str = Field(..., description="Directory holding the split's clips.")
    entries: List[ProtocolEntry] = Field(
        default_factory=list, description="Protocol rows retained for this split."
    )
    speaker_count: int = Field(default=0, description="Distinct speakers present.")
    attack_ids: List[str] = Field(
        default_factory=list, description="Distinct attack identifiers present."
    )
    label_counts: Dict[str, int] = Field(
        default_factory=dict, description="Row counts by ground-truth label."
    )
