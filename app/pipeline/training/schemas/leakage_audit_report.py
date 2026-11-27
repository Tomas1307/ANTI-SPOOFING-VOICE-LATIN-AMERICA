"""
Pydantic schema for the aggregate corpus leakage audit report.
"""
from typing import Dict, List

from pydantic import BaseModel, Field

from app.pipeline.training.schemas.leakage_check_result import (
    LeakageCheckResult,
)


class LeakageAuditReport(BaseModel):
    """Aggregate verdict of the pre-training corpus audit.

    This report is the reproducible artefact cited by the Technical
    Validation section of the MARSA data descriptor. It is written to disk as
    JSON on every run, passing or failing.

    Attributes:
        corpus_root: Corpus directory that was audited.
        split_sizes: Number of protocol rows per split.
        bonafide_fraction: Bonafide proportion per split.
        checks: Individual invariant results, in execution order.
        passed: True when every fatal check passed.
    """

    corpus_root: str = Field(..., description="Audited corpus directory.")
    split_sizes: Dict[str, int] = Field(
        default_factory=dict, description="Protocol rows per split."
    )
    bonafide_fraction: Dict[str, float] = Field(
        default_factory=dict, description="Bonafide proportion per split."
    )
    checks: List[LeakageCheckResult] = Field(
        default_factory=list, description="Invariant results in execution order."
    )
    passed: bool = Field(default=False, description="True when all fatal checks pass.")
