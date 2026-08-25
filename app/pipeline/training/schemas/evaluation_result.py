"""
Pydantic schema for a scored evaluation pass.
"""
from typing import Dict, List

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Countermeasure performance on a scored split.

    Two equal error rates are reported for dev and eval. The pooled figure
    covers every clip in the split; the strict figure covers only the subset
    whose source sentence is unseen in training, which is the leakage-hardened
    number the data descriptor reports alongside it.

    Attributes:
        split: Split that was scored.
        checkpoint: Checkpoint file used to produce the scores.
        clip_count: Clips scored.
        eer: Pooled equal error rate, as a percentage.
        strict_clip_count: Clips in the sentence-disjoint strict subset.
        strict_eer: Strict-subset equal error rate, as a percentage. Null when
            no strict filter was supplied.
        per_attack_eer: Equal error rate per attack system, each computed
            against the split's full bonafide pool.
        per_attack_clips: Spoof clips backing each per-attack rate. Always
            reported next to the rate, since a rate can only resolve
            differences down to roughly one over this count.
        low_confidence_attacks: Attacks whose clip count falls below the
            reporting threshold. Their rates are still reported in full; the
            flag exists so no reader leans on them.
        score_file: Path of the written per-clip score file.
    """

    split: str = Field(..., description="Split that was scored.")
    checkpoint: str = Field(..., description="Checkpoint used for scoring.")
    clip_count: int = Field(..., ge=0, description="Clips scored.")
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
    score_file: str = Field(default="", description="Path of the score file.")
