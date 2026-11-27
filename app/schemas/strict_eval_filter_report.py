"""
Pydantic schema for the strict (sentence-disjoint) eval filter report.
"""
from typing import Dict, List

from pydantic import BaseModel, Field


class StrictEvalFilterReport(BaseModel):
    """Summary of the sentence-disjoint strict-eval filter build.

    Attributes:
        train_sentence_count: Distinct normalized sentences in the training
            reference set.
        per_split_total: Clips inspected per split.
        per_split_strict: Clips marked strict (train-unseen sentence) per split.
        per_split_unresolved: Clips whose source sentence could not be
            resolved per split; these are conservatively marked non-strict.
        mapping_verification: Per-system duration-check result description.
        notes: Methodology caveats.
    """

    train_sentence_count: int = Field(..., ge=0)
    per_split_total: Dict[str, int] = Field(default_factory=dict)
    per_split_strict: Dict[str, int] = Field(default_factory=dict)
    per_split_unresolved: Dict[str, int] = Field(default_factory=dict)
    mapping_verification: Dict[str, str] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
