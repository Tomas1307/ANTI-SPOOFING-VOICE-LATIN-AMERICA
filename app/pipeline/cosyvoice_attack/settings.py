"""
Configuration settings for CosyVoice Attack Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Tuple


class CosyVoiceAttackSettings(BaseModel):
    """Configuration for CosyVoice voice cloning attack pipeline.

    This pipeline generates synthetic Spanish voice cloning attacks using
    CosyVoice 3.0 (Alibaba FunAudioLLM, Conditional Flow Matching architecture,
    trained on 1 million hours) for anti-spoofing dataset augmentation.
    Inference is fully local via zero-shot voice cloning mode.

    CosyVoice is not pip-installable. The repo is cloned from GitHub and
    imported by adding COSYVOICE_REPO_PATH and its Matcha-TTS submodule
    to sys.path at runtime.

    Attributes:
        VALIDATION_MODE: Toggle validation mode (3 speakers) vs production (all speakers).
        VALIDATION_SPEAKERS: Specific speakers for validation mode.

        BONAFIDE_DIR: Directory containing HABLA bonafide speakers.
        OUTPUT_DIR: Output directory for CosyVoice synthetic samples.
        CV_METADATA_PATH: Path to Mozilla Common Voice validated.tsv.

        COSYVOICE_REPO_PATH: Filesystem path where the CosyVoice GitHub repo is cloned.
        COSYVOICE_MODEL_DIR: Model directory relative to COSYVOICE_REPO_PATH.
        COSYVOICE_LOAD_JIT: Whether to load JIT-compiled model components.
        COSYVOICE_LOAD_TRT: Whether to load TensorRT-optimized model components.

        SAMPLE_RATE: Target audio processing sample rate (Hz).
        DEVICE: Compute device for model inference.

        REFERENCE_DURATION_TARGET: Target duration for reference audio (seconds).
        SAMPLES_PER_SPEAKER: Number of synthetic samples per speaker.
        RANDOM_SEED: Random seed for reproducible text sampling.

        TEXT_LENGTH_RANGE: Min and max words for text prompts.

        WHISPER_MODEL_SIZE: Whisper model size for reference audio transcription.
        WHISPER_COMPUTE_TYPE: Whisper compute precision for CTranslate2 backend.

        PARAKEET_MODEL_ID: Parakeet TDT model identifier for STT transcription.
        WER_MAX_ACCEPTABLE: Hard WER rejection ceiling.
        CER_MAX_ACCEPTABLE: Hard CER rejection ceiling.
        MIN_AUDIO_DURATION: Minimum acceptable audio duration in seconds.
        MAX_AUDIO_DURATION: Maximum acceptable audio duration in seconds.

        COSYVOICE_SYSTEM_ID: Attack identifier for protocol files.
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
        default=Path("data/cosyvoice_output"),
        description="Output directory for CosyVoice synthetic samples"
    )
    CV_METADATA_PATH: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05/es/validated.tsv"),
        description="Path to Mozilla Common Voice validated.tsv for text prompts"
    )

    # === CosyVoice Repo and Model Configuration ===
    COSYVOICE_REPO_PATH: Path = Field(
        default=Path.home() / "CosyVoice",
        description=(
            "Filesystem path where the CosyVoice GitHub repo is cloned. "
            "This repo is added to sys.path at import time because CosyVoice "
            "is not pip-installable."
        )
    )
    COSYVOICE_MODEL_DIR: str = Field(
        default="pretrained_models/CosyVoice2-0.5B",
        description="Model directory relative to COSYVOICE_REPO_PATH"
    )
    COSYVOICE_LOAD_JIT: bool = Field(
        default=False,
        description="Whether to load JIT-compiled model components (requires pre-compilation)"
    )
    COSYVOICE_LOAD_TRT: bool = Field(
        default=False,
        description="Whether to load TensorRT-optimized model components (requires TRT build)"
    )

    # === Whisper STT Configuration (for reference transcription) ===
    WHISPER_MODEL_SIZE: str = Field(
        default="large-v3",
        description="Whisper model size for reference audio transcription"
    )
    WHISPER_COMPUTE_TYPE: str = Field(
        default="float16",
        description="Whisper compute precision (float16, int8_float16, int8)"
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
        description="Target duration for reference audio clips in seconds."
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
    COSYVOICE_SYSTEM_ID: str = Field(
        default="COSYVOICE",
        description="Attack identifier for ASVspoof2019 protocol files"
    )
    AUDIO_ID_START_TRAIN: int = Field(
        default=11000000,
        description="Starting ID for train samples (11000000-11999999 range avoids collisions)"
    )
    AUDIO_ID_START_DEV: int = Field(
        default=11000000,
        description="Starting ID for dev samples"
    )
    AUDIO_ID_START_EVAL: int = Field(
        default=11000000,
        description="Starting ID for eval samples"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False


# Module-level singleton
settings = CosyVoiceAttackSettings()
