"""
Configuration settings for the DF-Arena detector training pipeline.

All pipeline-specific parameters live here as a Pydantic model. Only truly
global settings belong outside this module. Values are defaults: the launcher
script overlays command-line arguments on top of them to build the per-run
DetectorTrainingConfig.
"""
from typing import Dict, List

from pydantic import BaseModel, Field


class MarsaTrainingSettings(BaseModel):
    """Configuration for training anti-spoofing detectors on the MARSA corpus.

    Attributes:
        CORPUS_ROOT: Corpus directory containing the LA tree. The default is
            the 2x uniform augmentation tier.
        PARTITION_ROOT: Speaker-disjoint partition directory, source of the
            strict sentence-disjoint filter table.
        STRICT_FILTER_CSV: Strict filter table joining on source_file.
        RUNS_ROOT: Parent directory for per-run output directories.

        PROTOCOL_FILENAMES: Protocol file name per split, in ASVspoof2019 LA
            naming.
        METADATA_TEMPLATE: Metadata CSV name template per split.
        SPLIT_DIR_TEMPLATE: Split directory name template under LA.

        DETECTOR_BACKEND: Registered detector key used by the factory.
            Backend-specific parameters live in that backend subpackage.
        FREEZE_BACKBONE: Whether the self-supervised backbone stays frozen.

        SAMPLE_RATE: Expected corpus sample rate in Hz.
        CROP_SECONDS: Fixed training crop length in seconds.
        EVAL_CROP_SECONDS: Fixed scoring crop length; zero scores full clips.

        EPOCHS: Default number of epochs.
        BATCH_SIZE: Clips per optimiser micro-batch.
        GRADIENT_ACCUMULATION: Micro-batches accumulated per optimiser step.
        LEARNING_RATE: Peak learning rate for the classifier head.
        BACKBONE_LEARNING_RATE: Peak learning rate for backbone parameters.
        WEIGHT_DECAY: AdamW weight decay.
        WARMUP_RATIO: Fraction of total steps spent in linear warm-up.
        MAX_GRAD_NORM: Gradient-norm clipping threshold.
        AMP_DTYPE: Mixed-precision dtype: bf16, fp16 or none.
        CLASS_WEIGHTING: Weight the loss by inverse class frequency.
        NUM_WORKERS: Data-loader worker processes.
        SEED: Master random seed.

        CHECKPOINT_EVERY_STEPS: Optimiser steps between mid-epoch checkpoints.
        KEEP_LAST_N_CHECKPOINTS: Rolling checkpoints retained on disk. Kept
            deliberately small: a one-billion-parameter checkpoint carrying
            AdamW state is roughly sixteen gigabytes, and ml-server03 runs
            close to full.
        MIN_FREE_DISK_GB: Refuse to start a run with less free disk than this.

        ENFORCE_SINGLE_GPU: Refuse to start when more than one CUDA device is
            visible. ml-server03 is shared; runs must be pinned with
            CUDA_VISIBLE_DEVICES.
        BONAFIDE_FRACTION_TOLERANCE: Allowed absolute deviation of a split
            bonafide fraction from the corpus-wide fraction, before the audit
            flags a class-balance anomaly.
        ORDERING_LEAK_MIN_RUNS_RATIO: Minimum ratio of observed label runs to
            expected label runs along the audio-ID ordering. A corpus whose
            identifiers were assigned class by class collapses this ratio
            toward zero.
        EXPECTED_ATTACK_SYSTEMS: Attack system slugs that must appear in every
            split for leave-one-system-out protocols to be viable.
        MIN_ATTACK_CLIPS_FOR_EER: Spoof clips below which a per-attack rate is
            flagged as low confidence. The rate is still reported in full;
            the flag records that a rate over n clips cannot resolve
            differences finer than about one over n.
    """

    CORPUS_ROOT: str = Field(
        default="data/augmented/augmented_2x",
        description="Corpus directory containing the LA tree.",
    )
    PARTITION_ROOT: str = Field(
        default="data/marsa_speaker_disjoint_partition",
        description="Speaker-disjoint partition directory.",
    )
    STRICT_FILTER_CSV: str = Field(
        default="data/marsa_speaker_disjoint_partition/strict_eval_filter.csv",
        description="Strict sentence-disjoint filter table.",
    )
    RUNS_ROOT: str = Field(
        default="data/training_runs",
        description="Parent directory for per-run output directories.",
    )

    PROTOCOL_FILENAMES: Dict[str, str] = Field(
        default={
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
        },
        description="Protocol file name per split.",
    )
    METADATA_TEMPLATE: str = Field(
        default="MARSA.LA.cm.{split}.metadata.csv",
        description="Metadata CSV name template per split.",
    )
    SPLIT_DIR_TEMPLATE: str = Field(
        default="ASVspoof2019_LA_{split}",
        description="Split directory name template under LA.",
    )

    DETECTOR_BACKEND: str = Field(
        default="dfarena", description="Registered detector key."
    )
    FREEZE_BACKBONE: bool = Field(
        default=False, description="Freeze the self-supervised backbone."
    )

    SAMPLE_RATE: int = Field(default=16000, description="Corpus sample rate in Hz.")
    CROP_SECONDS: float = Field(
        default=4.0, description="Fixed training crop length in seconds."
    )
    EVAL_CROP_SECONDS: float = Field(
        default=0.0, description="Scoring crop length; 0 scores full clips."
    )

    EPOCHS: int = Field(default=10, description="Default number of epochs.")
    BATCH_SIZE: int = Field(default=8, description="Clips per micro-batch.")
    GRADIENT_ACCUMULATION: int = Field(
        default=4, description="Micro-batches per optimiser step."
    )
    LEARNING_RATE: float = Field(
        default=1e-4, description="Peak learning rate for the classifier head."
    )
    BACKBONE_LEARNING_RATE: float = Field(
        default=1e-6, description="Peak learning rate for backbone parameters."
    )
    WEIGHT_DECAY: float = Field(default=1e-4, description="AdamW weight decay.")
    WARMUP_RATIO: float = Field(default=0.05, description="Linear warm-up fraction.")
    MAX_GRAD_NORM: float = Field(
        default=5.0, description="Gradient-norm clipping threshold."
    )
    AMP_DTYPE: str = Field(default="bf16", description="Mixed-precision dtype.")
    CLASS_WEIGHTING: bool = Field(
        default=True, description="Weight the loss by inverse class frequency."
    )
    NUM_WORKERS: int = Field(default=8, description="Data-loader worker processes.")
    SEED: int = Field(default=42, description="Master random seed.")

    CHECKPOINT_EVERY_STEPS: int = Field(
        default=2000, description="Optimiser steps between mid-epoch checkpoints."
    )
    KEEP_LAST_N_CHECKPOINTS: int = Field(
        default=2, description="Rolling checkpoints retained on disk."
    )
    MIN_FREE_DISK_GB: float = Field(
        default=60.0, description="Refuse to start below this free disk margin."
    )

    ENFORCE_SINGLE_GPU: bool = Field(
        default=True, description="Refuse to start with more than one visible GPU."
    )
    BONAFIDE_FRACTION_TOLERANCE: float = Field(
        default=0.03, description="Allowed per-split bonafide fraction deviation."
    )
    ORDERING_LEAK_MIN_RUNS_RATIO: float = Field(
        default=0.80, description="Minimum observed-to-expected label-run ratio."
    )
    EXPECTED_ATTACK_SYSTEMS: List[str] = Field(
        default=[
            "fishgram",
            "qwen",
            "openvoice",
            "chatterbox",
            "outetts",
            "omnivoice",
        ],
        description="Attack systems that must appear in every split.",
    )
    MIN_ATTACK_CLIPS_FOR_EER: int = Field(
        default=30, description="Clips below which a per-attack rate is flagged."
    )


# Module-level singleton
settings = MarsaTrainingSettings()
