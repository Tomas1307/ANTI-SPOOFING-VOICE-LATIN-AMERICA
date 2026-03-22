"""
Configuration settings for FishGram Attack Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple


class FishGramAttackSettings(BaseModel):
    """Configuration for FishGram voice cloning attack pipeline.

    This pipeline generates synthetic Spanish voice cloning attacks using Fish Speech
    (4B parameters) for anti-spoofing dataset augmentation.

    Attributes:
        VALIDATION_MODE: Toggle validation mode (3 speakers) vs production (all speakers)
        VALIDATION_SPEAKERS: Specific speakers for validation mode

        BONAFIDE_DIR: Directory containing HABLA bonafide speakers
        OUTPUT_DIR: Output directory for FishGram synthetic samples
        FISHGRAM_MODEL_PATH: Path to Fish Speech model checkpoint
        CV_METADATA_PATH: Path to Mozilla Common Voice validated.tsv

        SAMPLE_RATE: Target audio sample rate (Hz)
        DEVICE: Compute device for model inference
        DTYPE: Model precision (bfloat16/float16)

        REFERENCE_DURATION_TARGET: Target duration for reference audio (seconds)
        SAMPLES_PER_SPEAKER: Number of synthetic samples per speaker
        RANDOM_SEED: Random seed for reproducible text sampling

        TEXT_LENGTH_RANGE: Min and max words for text prompts
        PARAKEET_MODEL_ID: Parakeet TDT model identifier for STT transcription
        WER_MAX_ACCEPTABLE: Hard WER rejection ceiling (samples above this are rejected)
        CER_MAX_ACCEPTABLE: Hard CER rejection ceiling (samples above this are rejected)
        MIN_AUDIO_DURATION: Minimum acceptable audio duration in seconds
        MAX_AUDIO_DURATION: Maximum acceptable audio duration in seconds

        FISHGRAM_SYSTEM_ID: Attack identifier for protocol files
        AUDIO_ID_START_TRAIN: Starting ID for train samples
        AUDIO_ID_START_DEV: Starting ID for dev samples
        AUDIO_ID_START_EVAL: Starting ID for eval samples
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
        default=Path("data/fishgram_output"),
        description="Output directory for FishGram synthetic samples"
    )
    FISHGRAM_MODEL_PATH: Path = Field(
        default=Path("pretrained_models/fish-speech-4b"),
        description="Path to Fish Speech model checkpoint"
    )
    CV_METADATA_PATH: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05/es/validated.tsv"),
        description="Path to Mozilla Common Voice validated.tsv for text prompts"
    )

    # === Fish Speech API Server ===
    FISH_SPEECH_API_URL: str = Field(
        default="http://localhost:8080",
        description="URL of the Fish Speech HTTP API server running on ml-server03"
    )
    FISH_SPEECH_FORMAT: str = Field(
        default="wav",
        description="Audio format for Fish Speech output (wav, mp3, pcm)"
    )
    FISH_SPEECH_TOP_P: float = Field(
        default=0.8,
        description="Top-p sampling for Fish Speech generation (0.1-1.0)"
    )
    FISH_SPEECH_TEMPERATURE: float = Field(
        default=0.8,
        description="Temperature for Fish Speech generation (0.1-1.0)"
    )
    FISH_SPEECH_REPETITION_PENALTY: float = Field(
        default=1.1,
        description="Repetition penalty for Fish Speech (0.9-2.0)"
    )

    # === Audio Processing ===
    SAMPLE_RATE: int = Field(
        default=16000,
        description="Target audio sample rate in Hz"
    )
    DEVICE: str = Field(
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        description="Compute device for model inference"
    )
    DTYPE: str = Field(
        default="bfloat16",
        description="Model precision (bfloat16 for A40 Ampere, float16 for older GPUs)"
    )

    # === Generation Parameters ===
    REFERENCE_DURATION_TARGET: float = Field(
        default=15.0,
        description="Target duration for reference audio clips (seconds)"
    )
    REFERENCE_DURATION_RANGE: Tuple[float, float] = Field(
        default=(10.0, 30.0),
        description="Acceptable range for reference audio duration (min, max)"
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
        default=(5, 100),
        description="Min and max words for text prompts"
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
    TRAIN_SPLIT_NAME: str = Field(
        default="train",
        description="Directory name for train split"
    )
    VAL_SPLIT_NAME: str = Field(
        default="val",
        description="Directory name for validation split"
    )
    TEST_SPLIT_NAME: str = Field(
        default="test",
        description="Directory name for test split"
    )

    # === Output Format Configuration ===
    FISHGRAM_SYSTEM_ID: str = Field(
        default="FISHGRAM",
        description="Attack identifier for ASVspoof2019 protocol files"
    )
    AUDIO_ID_START_TRAIN: int = Field(
        default=9000000,
        description="Starting ID for train samples (9000000-9999999 range avoids collisions)"
    )
    AUDIO_ID_START_DEV: int = Field(
        default=9000000,
        description="Starting ID for dev samples"
    )
    AUDIO_ID_START_EVAL: int = Field(
        default=9000000,
        description="Starting ID for eval samples"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False  # Allow runtime modification for toggling validation/production mode


# Module-level singleton
settings = FishGramAttackSettings()
