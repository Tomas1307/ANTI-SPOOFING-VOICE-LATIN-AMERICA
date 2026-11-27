"""
Pydantic schema for the attack identifier normalization report.
"""
from typing import Dict, List

from pydantic import BaseModel, Field


class AttackIdNormalizationReport(BaseModel):
    """Outcome of rewriting an attack identifier across the corpus.

    Attributes:
        source_prefix: Identifier prefix that was replaced.
        target_prefix: Identifier prefix it was replaced with.
        applied: False for a dry run, True when files were rewritten.
        files_scanned: Files inspected.
        files_changed: Files that contained at least one affected row.
        rows_changed: Total rows rewritten.
        per_file: Rows rewritten, keyed by file path.
        before_counts: Row counts per affected identifier before the rewrite.
        after_counts: Row counts per affected identifier after the rewrite.
        residual: Files still containing the source prefix after the rewrite.
            Must be empty for the operation to be considered successful.
        verified: Whether the post-rewrite verification pass succeeded.
    """

    source_prefix: str = Field(..., description="Identifier prefix replaced.")
    target_prefix: str = Field(..., description="Replacement identifier prefix.")
    applied: bool = Field(default=False, description="True when files were rewritten.")
    files_scanned: int = Field(default=0, description="Files inspected.")
    files_changed: int = Field(default=0, description="Files with affected rows.")
    rows_changed: int = Field(default=0, description="Total rows rewritten.")
    per_file: Dict[str, int] = Field(
        default_factory=dict, description="Rows rewritten per file."
    )
    before_counts: Dict[str, int] = Field(
        default_factory=dict, description="Counts per identifier before."
    )
    after_counts: Dict[str, int] = Field(
        default_factory=dict, description="Counts per identifier after."
    )
    residual: List[str] = Field(
        default_factory=list, description="Files still holding the source prefix."
    )
    verified: bool = Field(default=False, description="Verification pass succeeded.")
