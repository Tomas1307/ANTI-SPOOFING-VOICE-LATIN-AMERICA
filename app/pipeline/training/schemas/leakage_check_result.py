"""
Pydantic schema for the outcome of a single corpus leakage check.
"""
from typing import List

from pydantic import BaseModel, Field


class LeakageCheckResult(BaseModel):
    """Result of one named invariant asserted against the corpus.

    Attributes:
        name: Short identifier of the check.
        description: What the check proves when it passes.
        passed: Whether the invariant held.
        detail: Human-readable measurement backing the verdict.
        offenders: Bounded sample of violating items, for triage. Empty when
            the check passed.
        fatal: Whether a failure must abort training. Non-fatal checks are
            reported as warnings.
    """

    name: str = Field(..., description="Short identifier of the check.")
    description: str = Field(..., description="What passing proves.")
    passed: bool = Field(..., description="Whether the invariant held.")
    detail: str = Field(default="", description="Measurement backing the verdict.")
    offenders: List[str] = Field(
        default_factory=list, description="Bounded sample of violating items."
    )
    fatal: bool = Field(
        default=True, description="Whether a failure must abort training."
    )
