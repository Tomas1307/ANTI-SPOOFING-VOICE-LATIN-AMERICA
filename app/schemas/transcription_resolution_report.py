"""
Pydantic schema for the ASR-based strict-filter resolution report.
"""
from typing import Dict

from pydantic import BaseModel, Field


class TranscriptionResolutionReport(BaseModel):
    """Summary of resolving unresolved strict-filter clips via ASR.

    Attributes:
        candidates_considered: Unresolved clips selected for resolution.
        resolved: Clips whose source text was identified with a clear margin.
        still_unresolved: Clips left unresolved (no confident candidate).
        newly_strict: Resolved clips marked strict (text unseen in training).
        per_split_resolved: Resolved counts per split.
        acceptance_cer_max: Maximum CER between transcript and winning text.
        margin_min: Minimum CER margin required over the runner-up text.
    """

    candidates_considered: int = Field(..., ge=0)
    resolved: int = Field(..., ge=0)
    still_unresolved: int = Field(..., ge=0)
    newly_strict: int = Field(..., ge=0)
    per_split_resolved: Dict[str, int] = Field(default_factory=dict)
    acceptance_cer_max: float = Field(..., ge=0.0)
    margin_min: float = Field(..., ge=0.0)
