"""
Pydantic schema for the outcome of a full training run.
"""
from typing import List, Optional

from pydantic import BaseModel, Field

from app.pipeline.training.schemas.epoch_result import EpochResult
from app.pipeline.training.schemas.evaluation_result import EvaluationResult


class TrainingResult(BaseModel):
    """Everything a completed training run produced.

    Attributes:
        run_name: Identifier of the run, also the output directory name.
        epochs: Per-epoch metrics in execution order.
        best_epoch: Epoch index with the lowest development EER.
        best_dev_eer: Lowest development EER observed, as a percentage.
        best_checkpoint: Path of the checkpoint for the best epoch.
        last_checkpoint: Path of the most recent checkpoint.
        evaluations: Evaluation passes performed after training.
        resumed_from: Checkpoint the run resumed from, if any.
    """

    run_name: str = Field(..., description="Run identifier and output directory.")
    epochs: List[EpochResult] = Field(
        default_factory=list, description="Per-epoch metrics."
    )
    best_epoch: int = Field(default=-1, description="Epoch with lowest dev EER.")
    best_dev_eer: float = Field(default=-1.0, description="Lowest dev EER, percent.")
    best_checkpoint: str = Field(default="", description="Best-epoch checkpoint path.")
    last_checkpoint: str = Field(default="", description="Most recent checkpoint path.")
    evaluations: List[EvaluationResult] = Field(
        default_factory=list, description="Post-training evaluation passes."
    )
    resumed_from: Optional[str] = Field(
        default=None, description="Checkpoint the run resumed from."
    )
