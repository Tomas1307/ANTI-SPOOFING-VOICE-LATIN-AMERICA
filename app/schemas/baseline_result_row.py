"""
Pydantic schema for one row of a gathered baseline results table.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BaselineResultRow(BaseModel):
    """One model, one split, one scored run -- the unit a results table joins on.

    Attributes:
        run_name: Training-run directory name, e.g. ``dfarena_zeroshot``.
        detector_backend: Registered backend key, e.g. ``dfarena``, ``lcnn``,
            ``lcnn_fixedcrop``.
        checkpoint: Model identifier actually scored -- a Hugging Face
            repository id for dfarena, or a local .pt path for the LCNN
            family. Identifies which of a backend's several checkpoints
            produced this row, since e.g. ``lcnn`` alone is ambiguous between
            the original and fine-tuned LSTM-sum weights.
        eval_only: Whether this run was zero-shot (no fine-tuning on MARSA)
            or the model was actually trained here.
        split: Corpus split scored, ``dev`` or ``eval``.
        clip_count: Clips scored in the pooled figure.
        eer: Pooled equal error rate, percent.
        strict_clip_count: Clips in the sentence-disjoint strict subset.
        strict_eer: Strict-subset equal error rate, percent. Negative when no
            strict filter was configured for the run.
        per_attack_eer: Equal error rate per attack system, percent.
        per_attack_clips: Spoof clips backing each per-attack rate.
        low_confidence_attacks: Attacks whose per-attack rate rests on too few
            clips to resolve fine differences.
    """

    run_name: str = Field(..., description="Training-run directory name.")
    detector_backend: str = Field(..., description="Registered backend key.")
    checkpoint: Optional[str] = Field(
        default=None, description="Model identifier actually scored."
    )
    eval_only: bool = Field(..., description="Zero-shot vs. trained on MARSA.")
    split: str = Field(..., description="Corpus split scored: dev or eval.")
    clip_count: int = Field(..., description="Clips scored in the pooled figure.")
    eer: float = Field(..., description="Pooled EER, percent.")
    strict_clip_count: int = Field(default=0, description="Clips in strict subset.")
    strict_eer: float = Field(default=-1.0, description="Strict EER, percent.")
    per_attack_eer: Dict[str, float] = Field(
        default_factory=dict, description="EER per attack system, percent."
    )
    per_attack_clips: Dict[str, int] = Field(
        default_factory=dict, description="Spoof clips behind each per-attack rate."
    )
    low_confidence_attacks: List[str] = Field(
        default_factory=list, description="Attacks with too few clips to lean on."
    )
