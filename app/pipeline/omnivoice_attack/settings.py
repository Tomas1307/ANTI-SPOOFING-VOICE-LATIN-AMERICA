"""
Configuration settings for OmniVoice Attack Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Tuple


class OmniVoiceAttackSettings(BaseModel):
    """Configuration for OmniVoice voice cloning attack pipeline.

    OmniVoice (k2-fsa) is a massively multilingual zero-shot TTS model based
    on a diffusion language model architecture. It supports 646 languages
    including Spanish (27,559 hours of training data, one of the largest
    represented languages). This pipeline generates synthetic Spanish voice
    cloning attacks for anti-spoofing dataset augmentation.

    Attributes:
        VALIDATION_MODE: Toggle validation mode (3 speakers) vs production (all speakers).
        VALIDATION_SPEAKERS: Specific speakers for validation mode.

        BONAFIDE_DIR: Directory containing HABLA bonafide speakers.
        OUTPUT_DIR: Output directory for OmniVoice synthetic samples.
        OMNIVOICE_MODEL_ID: HuggingFace model identifier for OmniVoice.
        CV_METADATA_PATH: Path to Mozilla Common Voice validated.tsv.

        SAMPLE_RATE: Target audio processing sample rate (Hz). All loads/writes
            outside Step 3 are normalized to this rate via librosa.
        OMNIVOICE_NATIVE_SAMPLE_RATE: Native output sample rate of OmniVoice (24 kHz).
            Step 3 writes raw output at this rate; downstream steps resample on load.
        DEVICE: Compute device for model inference.
        DTYPE: Model precision (float16 recommended by OmniVoice docs).

        REFERENCE_DURATION_TARGET: Target duration for reference audio (seconds).
            OmniVoice docs recommend 3-10s; longer audio degrades cloning quality.
        SAMPLES_PER_SPEAKER: Number of synthetic samples per speaker.
        MATCH_BONAFIDE_COUNT: When True, match bonafide_count per speaker.
        MAX_GENERATION_RETRIES: Max retry rounds for regenerating rejected samples.
        RANDOM_SEED: Random seed for reproducible text sampling.

        TEXT_SOURCE: Text corpus source identifier.
        TEXT_LENGTH_RANGE: Min and max words for text prompts.

        OMNIVOICE_NUM_STEP: Diffusion steps (16 for faster, 32 for higher quality).
        OMNIVOICE_SPEED: Speed factor for generation (1.0 = normal).

        PARAKEET_MODEL_ID: NVIDIA Parakeet TDT model ID for STT (ref_text + validation).
        WER_MAX_ACCEPTABLE: Hard WER rejection ceiling (samples above this are rejected).
        CER_MAX_ACCEPTABLE: Hard CER rejection ceiling (samples above this are rejected).
        MIN_AUDIO_DURATION: Minimum acceptable audio duration in seconds.
        MAX_AUDIO_DURATION: Maximum acceptable audio duration in seconds.

        NISQA_MIN_ACCEPTABLE: Informational NISQA MOS threshold (not used for rejection).
        SPEAKER_SIM_MIN_ACCEPTABLE: Informational ECAPA-TDNN similarity threshold.

        TRAIN_SPLIT_NAME: Directory name for train split.
        VAL_SPLIT_NAME: Directory name for validation split.
        TEST_SPLIT_NAME: Directory name for test split.

        OMNIVOICE_SYSTEM_ID: Attack identifier for protocol files.
        AUDIO_ID_START_TRAIN: Starting ID for train samples.
        AUDIO_ID_START_DEV: Starting ID for dev samples.
        AUDIO_ID_START_EVAL: Starting ID for eval samples.
    """

    VALIDATION_MODE: bool = Field(
        default=True,
        description="Validation mode: True=3 speakers, False=all speakers"
    )
    VALIDATION_SPEAKERS: List[str] = Field(
        default=["arf_00295", "arf_00610", "arf_01523"],
        description="Speakers for validation mode (3 Argentina Female speakers, matches FishGram)"
    )

    BONAFIDE_DIR: Path = Field(
        default=Path("data/bonafide_dataset_by_speaker_v2"),
        description="Directory containing HABLA v2 bonafide speakers (1,567 total)"
    )
    OUTPUT_DIR: Path = Field(
        default=Path("data/omnivoice_output"),
        description="Output directory for OmniVoice synthetic samples"
    )
    OMNIVOICE_MODEL_ID: str = Field(
        default="k2-fsa/OmniVoice",
        description="HuggingFace model identifier for OmniVoice (k2-fsa repository)"
    )
    CV_METADATA_PATH: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05/es/validated.tsv"),
        description="Path to Mozilla Common Voice validated.tsv for text prompts"
    )

    SAMPLE_RATE: int = Field(
        default=16000,
        description="Target audio processing sample rate in Hz (output FLAC and Parakeet input)"
    )
    OMNIVOICE_NATIVE_SAMPLE_RATE: int = Field(
        default=24000,
        description="Native sample rate of OmniVoice output before resampling"
    )
    DEVICE: str = Field(
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        description="Compute device for model inference"
    )
    DTYPE: str = Field(
        default="float16",
        description="Model precision (float16 recommended by OmniVoice docs for NVIDIA GPUs)"
    )

    REFERENCE_DURATION_TARGET: float = Field(
        default=10.0,
        description="Target duration for reference audio in seconds (OmniVoice tip: 3-10s, longer hurts quality)"
    )
    REFERENCE_DURATION_RANGE: Tuple[float, float] = Field(
        default=(3.0, 10.0),
        description="Acceptable range for reference audio duration (min, max)"
    )
    SAMPLES_PER_SPEAKER: int = Field(
        default=2,
        description="Texts per speaker (fallback when MATCH_BONAFIDE_COUNT is False)"
    )
    MATCH_BONAFIDE_COUNT: bool = Field(
        default=True,
        description="When True, generate as many samples as bonafide files per speaker"
    )
    MAX_GENERATION_RETRIES: int = Field(
        default=5,
        description="Max retry rounds for regenerating rejected samples after Step 4 validation"
    )
    RANDOM_SEED: int = Field(
        default=42,
        description="Random seed for reproducible text sampling and generation"
    )

    TEXT_SOURCE: str = Field(
        default="mozilla_cv",
        description="Text corpus source (mozilla_cv or custom)"
    )
    TEXT_LENGTH_RANGE: Tuple[int, int] = Field(
        default=(5, 100),
        description="Min and max words for text prompts"
    )

    OMNIVOICE_NUM_STEP: int = Field(
        default=32,
        description="Diffusion sampling steps (16 for faster, 32 for higher quality)"
    )
    OMNIVOICE_SPEED: float = Field(
        default=1.0,
        description="Speed factor for OmniVoice generation (>1.0 faster, <1.0 slower)"
    )
    OMNIVOICE_LANGUAGE: str = Field(
        default="es",
        description="Language code passed to OmniVoice generate(). 'es' for Spanish per k2-fsa/OmniVoice languages.md"
    )

    PARAKEET_MODEL_ID: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="NVIDIA Parakeet TDT model ID for STT (ref_text + Step 4 validation)"
    )
    WER_MAX_ACCEPTABLE: float = Field(
        default=0.15,
        description="Hard WER rejection ceiling; samples above this are rejected"
    )
    CER_MAX_ACCEPTABLE: float = Field(
        default=0.10,
        description="Hard CER rejection ceiling; samples above this are rejected"
    )
    MIN_AUDIO_DURATION: float = Field(
        default=0.5,
        description="Minimum acceptable audio duration in seconds"
    )
    MAX_AUDIO_DURATION: float = Field(
        default=30.0,
        description="Maximum acceptable audio duration in seconds"
    )

    NISQA_MIN_ACCEPTABLE: float = Field(
        default=2.5,
        description="Minimum acceptable NISQA MOS score (informational, not used for rejection)"
    )
    SPEAKER_SIM_MIN_ACCEPTABLE: float = Field(
        default=0.7,
        description="Minimum acceptable ECAPA-TDNN cosine similarity (informational only)"
    )

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

    OMNIVOICE_SYSTEM_ID: str = Field(
        default="OMNIVOICE",
        description="Attack identifier for ASVspoof2019 protocol files"
    )
    AUDIO_ID_START_TRAIN: int = Field(
        default=15000000,
        description="Starting ID for train samples (15000000+ avoids collision with partial_spoof main W1/W2/W3 at 12-14M)"
    )
    AUDIO_ID_START_DEV: int = Field(
        default=15000000,
        description="Starting ID for dev samples"
    )
    AUDIO_ID_START_EVAL: int = Field(
        default=15000000,
        description="Starting ID for eval samples"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False


settings = OmniVoiceAttackSettings()
