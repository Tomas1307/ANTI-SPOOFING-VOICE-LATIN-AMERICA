"""
Configuration settings for Qwen Attack Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Tuple


class QwenAttackSettings(BaseModel):
    """Configuration for Qwen3-TTS voice cloning attack pipeline.

    This pipeline generates synthetic Spanish voice cloning attacks using
    Qwen3-TTS (1.7B parameters) for anti-spoofing dataset augmentation.
    Secondary attack providing codec architecture diversity alongside FishGram.

    Attributes:
        VALIDATION_MODE: Toggle validation mode (3 speakers) vs production (all speakers).
        VALIDATION_SPEAKERS: Specific speakers for validation mode.

        BONAFIDE_DIR: Directory containing HABLA bonafide speakers.
        OUTPUT_DIR: Output directory for Qwen synthetic samples.
        CV_METADATA_PATH: Path to Mozilla Common Voice validated.tsv.

        QWEN_MODEL_ID: HuggingFace model identifier for Qwen3-TTS.
        QWEN_LANGUAGE: Language tag for speech generation.
        QWEN_ATTN_IMPLEMENTATION: Attention mechanism implementation.

        SAMPLE_RATE: Target audio sample rate (Hz).
        DEVICE: Compute device for model inference.
        DTYPE: Model precision (bfloat16/float16).

        REFERENCE_DURATION_TARGET: Target duration for reference audio (seconds).
        SAMPLES_PER_SPEAKER: Number of synthetic samples per speaker.
        RANDOM_SEED: Random seed for reproducible text sampling.

        DNSMOS_THRESHOLD_OVRL: Minimum DNSMOS overall score for validation.
        SPEAKER_SIM_THRESHOLD: Minimum speaker similarity for validation.

        QWEN_SYSTEM_ID: Attack identifier for protocol files.
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
        default=Path("data/qwen_output"),
        description="Output directory for Qwen synthetic samples"
    )
    CV_METADATA_PATH: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05/es/validated.tsv"),
        description="Path to Mozilla Common Voice validated.tsv for text prompts"
    )

    # === Qwen3-TTS Model Configuration ===
    QWEN_MODEL_ID: str = Field(
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        description="HuggingFace model ID for Qwen3-TTS (must be 1.7B, 0.6B has dimension bugs)"
    )
    QWEN_LANGUAGE: str = Field(
        default="Spanish",
        description="Language tag for generate_voice_clone (Spanish, English, Chinese, etc.)"
    )
    QWEN_ATTN_IMPLEMENTATION: str = Field(
        default="sdpa",
        description="Attention implementation (sdpa default, flash_attention_2 if flash-attn installed)"
    )
    X_VECTOR_ONLY_MODE: bool = Field(
        default=False,
        description="False=full voice cloning with ref_text (higher quality), True=embedding only"
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
    MAX_NEW_TOKENS: int = Field(
        default=2048,
        description="Maximum new tokens for Qwen3-TTS generation"
    )

    # === Qwen Sampling Parameters ===
    TOP_K: int = Field(
        default=50,
        description="Top-k sampling for main talker"
    )
    TOP_P: float = Field(
        default=1.0,
        description="Top-p (nucleus) sampling for main talker"
    )
    TEMPERATURE: float = Field(
        default=0.9,
        description="Temperature for main talker generation"
    )
    REPETITION_PENALTY: float = Field(
        default=1.05,
        description="Repetition penalty for generation"
    )

    # === Qwen Subtalker Parameters ===
    SUBTALKER_TOP_K: int = Field(
        default=50,
        description="Top-k sampling for subtalker (secondary codebook decoder)"
    )
    SUBTALKER_TOP_P: float = Field(
        default=1.0,
        description="Top-p sampling for subtalker"
    )
    SUBTALKER_TEMPERATURE: float = Field(
        default=0.9,
        description="Temperature for subtalker generation"
    )

    # === Text Prompt Configuration ===
    TEXT_SOURCE: str = Field(
        default="mozilla_cv",
        description="Text corpus source (mozilla_cv or custom)"
    )
    TEXT_LENGTH_RANGE: Tuple[int, int] = Field(
        default=(5, 40),
        description="Min and max words for text prompts (40 max to prevent Qwen truncation)"
    )
    MAX_TEXT_LENGTH_WORDS: int = Field(
        default=40,
        description="Hard ceiling on text length to prevent Qwen silent truncation artifacts"
    )

    # === Parakeet STT Validation ===
    PARAKEET_MODEL_ID: str = Field(
        default="nvidia/parakeet-tdt-1.1b",
        description="HuggingFace model ID for NVIDIA Parakeet TDT ASR"
    )
    WER_THRESHOLD: float = Field(
        default=0.0,
        description="Target Word Error Rate (0.0 = perfect match). Logged but not used for rejection."
    )
    CER_THRESHOLD: float = Field(
        default=0.0,
        description="Target Character Error Rate (0.0 = perfect match). Logged but not used for rejection."
    )
    WER_MAX_ACCEPTABLE: float = Field(
        default=0.15,
        description="Hard rejection ceiling for WER after prefix trimming (0.15 = 15% of words wrong)"
    )
    CER_MAX_ACCEPTABLE: float = Field(
        default=0.10,
        description="Hard rejection ceiling for CER after prefix trimming (0.10 = 10% of chars wrong)"
    )

    # === Artifact Detection Thresholds ===
    MIN_AUDIO_DURATION: float = Field(
        default=0.5,
        description="Minimum acceptable audio duration in seconds (reject silent/garbled outputs)"
    )
    MAX_AUDIO_DURATION: float = Field(
        default=30.0,
        description="Maximum acceptable audio duration in seconds"
    )
    LOW_ENERGY_THRESHOLD: float = Field(
        default=0.001,
        description="RMS threshold below which audio is considered near-silent"
    )
    MIN_WORDS_PER_SECOND: float = Field(
        default=1.5,
        description="Minimum expected speaking rate for truncation detection"
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
    QWEN_SYSTEM_ID: str = Field(
        default="QWEN3TTS",
        description="Attack identifier for ASVspoof2019 protocol files"
    )
    AUDIO_ID_START_TRAIN: int = Field(
        default=8000000,
        description="Starting ID for train samples (8000000 range avoids FishGram 9000000 collisions)"
    )
    AUDIO_ID_START_DEV: int = Field(
        default=8000000,
        description="Starting ID for dev samples"
    )
    AUDIO_ID_START_EVAL: int = Field(
        default=8000000,
        description="Starting ID for eval samples"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False  # Allow runtime modification for toggling validation/production mode


# Module-level singleton
settings = QwenAttackSettings()
