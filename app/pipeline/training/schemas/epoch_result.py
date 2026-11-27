"""
Pydantic schema for the metrics of a single training epoch.
"""
from pydantic import BaseModel, Field


class EpochResult(BaseModel):
    """Metrics recorded at the end of one training epoch.

    Attributes:
        epoch: Zero-based epoch index.
        global_step: Optimiser steps completed at the end of the epoch.
        train_loss: Mean training loss over the epoch.
        dev_loss: Mean development-set loss.
        dev_eer: Development-set equal error rate, as a percentage.
        learning_rate: Learning rate in force at the end of the epoch.
        seconds: Wall-clock duration of the epoch.
        is_best: Whether this epoch produced the lowest development EER so far.
    """

    epoch: int = Field(..., ge=0, description="Zero-based epoch index.")
    global_step: int = Field(..., ge=0, description="Optimiser steps completed.")
    train_loss: float = Field(..., description="Mean training loss.")
    dev_loss: float = Field(..., description="Mean development-set loss.")
    dev_eer: float = Field(..., description="Development EER, percent.")
    learning_rate: float = Field(..., description="Learning rate at epoch end.")
    seconds: float = Field(..., description="Epoch wall-clock duration.")
    is_best: bool = Field(default=False, description="Whether this is the best epoch.")
