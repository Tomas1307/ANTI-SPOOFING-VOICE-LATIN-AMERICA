"""
Configuration settings for OuteTTS Attack Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Tuple


class OuteTTSAttackSettings(BaseModel):
    """Configuration for OuteTTS voice cloning attack pipeline.

    This pipeline generates synthetic Spanish voice cloning attacks using
    OuteTTS 0.6B (Qwen-based LLM with DAC codec) for anti-spoofing dataset
    augmentation. Inference is fully local -- the model creates a speaker
    profile from reference audio, then generates speech autoregressively.

    OuteTTS is notably SLOW (~2-3 min per 10s audio on A40) due to its
    autoregressive LLM backbone. The 8,192 token context window limits
    effective output to ~32s with a reference profile included.

    Attributes:
        VALIDATION_MODE: Toggle validation mode (3 speakers) vs production (all speakers).
        VALIDATION_SPEAKERS: Specific speakers for validation mode.

        BONAFIDE_DIR: Directory containing HABLA bonafide speakers.
        OUTPUT_DIR: Output directory for OuteTTS synthetic samples.
        CV_METADATA_PATH: Path to Mozilla Common Voice validated.tsv.

        SAMPLE_RATE: Target audio processing sample rate (Hz).
        DEVICE: Compute device for model inference.

        REFERENCE_DURATION_TARGET: Target duration for reference audio (seconds).
        SAMPLES_PER_SPEAKER: Number of synthetic samples per speaker.
        RANDOM_SEED: Random seed for reproducible text sampling.

        TEXT_LENGTH_RANGE: Min and max words for text prompts.
        OUTETTS_MODEL_VERSION: OuteTTS model version string for InterfaceHF.
        OUTETTS_MODEL_ID: HuggingFace model ID for OuteTTS weights and tokenizer.
        TOP_K: Top-K sampling parameter for autoregressive decoding.
        TOP_P: Top-P (nucleus) sampling parameter.
        TEMPERATURE: Sampling temperature for speech token generation.
        REPETITION_PENALTY: Penalty for repeated speech tokens.
        MAX_LENGTH: Maximum token length for generation (context window budget).

        PARAKEET_MODEL_ID: Parakeet TDT model identifier for STT transcription.
        WER_MAX_ACCEPTABLE: Hard WER rejection ceiling.
        CER_MAX_ACCEPTABLE: Hard CER rejection ceiling.
        MIN_AUDIO_DURATION: Minimum acceptable audio duration in seconds.
        MAX_AUDIO_DURATION: Maximum acceptable audio duration in seconds.

        OUTETTS_SYSTEM_ID: Attack identifier for protocol files.
        AUDIO_ID_START_TRAIN: Starting ID for train samples.
        AUDIO_ID_START_DEV: Starting ID for dev samples.
        AUDIO_ID_START_EVAL: Starting ID for eval samples.
    """

    # === Validation Mode Toggle ===
    VALIDATION_MODE: bool = Field(
        default=True,
        description="Validation mode: True=3 speakers, False=all 162 speakers"
    )
    VALIDATION_SPEAKERS: List[str] = Field(
        default=["arf_00295", "arf_00610", "arf_01523"],
        description="Speakers for validation mode (3 Argentina Female speakers)"
    )

    # === Directory Paths ===
    BONAFIDE_DIR: Path = Field(
        default=Path("data/bonafide_dataset_by_speaker"),
        description="Directory containing HABLA bonafide speakers (162 total)"
    )
    OUTPUT_DIR: Path = Field(
        default=Path("data/outetts_output"),
        description="Output directory for OuteTTS synthetic samples"
    )
    CV_METADATA_PATH: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05/es/validated.tsv"),
        description="Path to Mozilla Common Voice validated.tsv for text prompts"
    )

    # === Audio Processing ===
    SAMPLE_RATE: int = Field(
        default=16000,
        description="Target audio processing sample rate in Hz (output is resampled to this)"
    )
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Compute device for model inference"
    )

    # === Generation Parameters ===
    REFERENCE_DURATION_TARGET: float = Field(
        default=15.0,
        description=(
            "Target duration for reference audio clips in seconds. "
            "OuteTTS benefits from 10-15s reference for speaker profile extraction."
        )
    )
    SAMPLES_PER_SPEAKER: int = Field(
        default=2,
        description="Texts per speaker (2 for validation, 5 for production)"
    )
    RANDOM_SEED: int = Field(
        default=42,
        description="Random seed for reproducible text sampling"
    )

    # === Text Prompt Configuration ===
    TEXT_SOURCE: str = Field(
        default="mozilla_cv",
        description="Text corpus source (mozilla_cv or custom)"
    )
    TEXT_LENGTH_RANGE: Tuple[int, int] = Field(
        default=(5, 50),
        description="Min and max words for text prompts"
    )

    # === OuteTTS Model Parameters ===
    OUTETTS_MODEL_VERSION: str = Field(
        default="0.3",
        description="OuteTTS model version string passed to InterfaceHF"
    )
    OUTETTS_MODEL_ID: str = Field(
        default="OuteAI/OuteTTS-0.3-500M",
        description="HuggingFace model ID for OuteTTS weights and tokenizer"
    )
    TOP_K: int = Field(
        default=40,
        description="Top-K sampling parameter for autoregressive decoding"
    )
    TOP_P: float = Field(
        default=0.9,
        description="Top-P (nucleus) sampling parameter"
    )
    TEMPERATURE: float = Field(
        default=0.4,
        description="Sampling temperature for speech token generation (lower = more deterministic)"
    )
    REPETITION_PENALTY: float = Field(
        default=1.1,
        description="Penalty for repeated speech tokens (1.0 = no penalty)"
    )
    MAX_LENGTH: int = Field(
        default=4096,
        description=(
            "Maximum token length for generation. OuteTTS context window is 8192 tokens; "
            "4096 leaves room for the speaker profile and prompt overhead."
        )
    )

    # === Parakeet STT Validation ===
    PARAKEET_MODEL_ID: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="NVIDIA Parakeet TDT model ID for transcription-based quality validation (supports Spanish)"
    )
    WER_MAX_ACCEPTABLE: float = Field(
        default=0.15,
        description="Hard WER rejection ceiling; samples above this threshold are rejected"
    )
    CER_MAX_ACCEPTABLE: float = Field(
        default=0.10,
        description="Hard CER rejection ceiling; samples above this threshold are rejected"
    )
    MIN_AUDIO_DURATION: float = Field(
        default=0.5,
        description="Minimum acceptable audio duration in seconds; shorter clips are rejected"
    )
    MAX_AUDIO_DURATION: float = Field(
        default=30.0,
        description="Maximum acceptable audio duration in seconds; longer clips are rejected"
    )

    # === Quality Metrics (Informational, not used for rejection) ===
    NISQA_MIN_ACCEPTABLE: float = Field(
        default=2.5,
        description="Minimum acceptable NISQA MOS score (informational threshold, not used for rejection)"
    )
    SPEAKER_SIM_MIN_ACCEPTABLE: float = Field(
        default=0.7,
        description="Minimum acceptable ECAPA-TDNN cosine similarity (informational, not rejection)"
    )

    # === Split Configuration ===
    TRAIN_SPLIT_NAME: str = Field(default="train", description="Directory name for train split")
    VAL_SPLIT_NAME: str = Field(default="val", description="Directory name for validation split")
    TEST_SPLIT_NAME: str = Field(default="test", description="Directory name for test split")

    # === Output Format Configuration ===
    OUTETTS_SYSTEM_ID: str = Field(
        default="OUTETTS",
        description="Attack identifier for ASVspoof2019 protocol files"
    )
    AUDIO_ID_START_TRAIN: int = Field(
        default=10000000,
        description="Starting ID for train samples (10000000+ range avoids collisions)"
    )
    AUDIO_ID_START_DEV: int = Field(
        default=10000000,
        description="Starting ID for dev samples"
    )
    AUDIO_ID_START_EVAL: int = Field(
        default=10000000,
        description="Starting ID for eval samples"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False


# Module-level singleton
settings = OuteTTSAttackSettings()
