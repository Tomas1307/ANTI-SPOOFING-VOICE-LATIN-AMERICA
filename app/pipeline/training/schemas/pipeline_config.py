"""
Pydantic schema for a DF-Arena training run configuration.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class DetectorTrainingConfig(BaseModel):
    """Input configuration for one training run.

    Defaults are supplied by the pipeline settings singleton; the launcher
    script overlays command-line arguments on top. The resulting object is
    serialised into the run directory so a run can be reproduced exactly.

    Attributes:
        run_name: Identifier of the run and name of its output directory.
        corpus_root: Corpus directory containing the ``LA`` tree.
        detector_backend: Registered detector key, for example ``dfarena``.
        model_id: Backbone identifier. None defers to the backend settings.
        freeze_backbone: Whether the self-supervised backbone stays frozen.
        epochs: Number of epochs to train.
        batch_size: Clips per optimiser micro-batch.
        gradient_accumulation: Micro-batches accumulated per optimiser step.
        learning_rate: Peak learning rate for the classifier head.
        backbone_learning_rate: Peak learning rate for unfrozen backbone
            parameters. Ignored when the backbone is frozen.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of total steps spent in linear warm-up.
        crop_seconds: Fixed training crop length in seconds.
        eval_crop_seconds: Fixed scoring crop length in seconds. Zero scores
            the full utterance, one clip at a time.
        num_workers: Data-loader worker processes.
        seed: Master random seed.
        amp_dtype: Mixed-precision dtype, one of ``bf16``, ``fp16`` or ``none``.
        class_weighting: Whether to weight the loss by inverse class frequency.
        checkpoint_every_steps: Optimiser steps between mid-epoch checkpoints.
        keep_last_n_checkpoints: Rolling checkpoints retained on disk.
        resume: Checkpoint to resume from, or ``auto`` to pick up the latest
            checkpoint in the run directory, or None to start fresh.
        eval_splits: Splits scored after training completes.
        strict_filter_csv: Strict sentence-disjoint filter table. When set,
            evaluation reports a strict EER alongside the pooled EER.
        eval_only: Score an existing checkpoint without training it. This is
            how a zero-shot baseline is produced: no optimiser is built and
            the training split is never read.
        eval_checkpoint: Checkpoint to score in eval-only mode. None scores
            the backend as it loads from its published weights.
        skip_audit: Whether to skip the pre-training corpus audit. Intended
            for smoke tests only.
        max_train_items: Cap on training clips, for smoke tests. Zero means no
            cap.
    """

    run_name: str = Field(..., description="Run identifier and output directory.")
    corpus_root: str = Field(..., description="Corpus directory with the LA tree.")
    detector_backend: str = Field(default="dfarena", description="Detector key.")
    model_id: Optional[str] = Field(
        default=None,
        description="Backbone identifier; None defers to the backend settings.",
    )
    freeze_backbone: bool = Field(default=False, description="Freeze the backbone.")

    epochs: int = Field(default=10, ge=1, description="Epochs to train.")
    batch_size: int = Field(default=8, ge=1, description="Clips per micro-batch.")
    gradient_accumulation: int = Field(
        default=4, ge=1, description="Micro-batches per optimiser step."
    )
    learning_rate: float = Field(default=1e-4, gt=0, description="Head learning rate.")
    backbone_learning_rate: float = Field(
        default=1e-6, gt=0, description="Backbone learning rate."
    )
    weight_decay: float = Field(default=1e-4, ge=0, description="AdamW weight decay.")
    warmup_ratio: float = Field(
        default=0.05, ge=0.0, le=0.5, description="Linear warm-up fraction."
    )

    crop_seconds: float = Field(default=4.0, gt=0, description="Training crop length.")
    eval_crop_seconds: float = Field(
        default=0.0, ge=0, description="Scoring crop length; 0 scores full clips."
    )
    num_workers: int = Field(default=8, ge=0, description="Data-loader workers.")
    seed: int = Field(default=42, description="Master random seed.")
    amp_dtype: str = Field(default="bf16", description="Mixed-precision dtype.")
    class_weighting: bool = Field(
        default=True, description="Weight the loss by inverse class frequency."
    )

    checkpoint_every_steps: int = Field(
        default=2000, ge=0, description="Steps between mid-epoch checkpoints."
    )
    keep_last_n_checkpoints: int = Field(
        default=2, ge=1, description="Rolling checkpoints retained on disk."
    )
    resume: Optional[str] = Field(
        default="auto", description="Checkpoint path, 'auto', or None."
    )

    eval_splits: List[str] = Field(
        default_factory=lambda: ["dev", "eval"], description="Splits scored at the end."
    )
    strict_filter_csv: Optional[str] = Field(
        default=None, description="Strict sentence-disjoint filter table."
    )
    eval_only: bool = Field(
        default=False, description="Score a checkpoint without training it."
    )
    eval_checkpoint: Optional[str] = Field(
        default=None, description="Checkpoint to score in eval-only mode."
    )
    skip_audit: bool = Field(default=False, description="Skip the corpus audit.")
    max_train_items: int = Field(
        default=0, ge=0, description="Training clip cap for smoke tests; 0 is no cap."
    )
