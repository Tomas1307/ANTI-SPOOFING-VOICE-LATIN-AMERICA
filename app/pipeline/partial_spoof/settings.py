"""Pipeline-scoped configuration for the Partial Spoof pipeline.

All tunable parameters for generating partially spoofed audio from HABLA
bonafide utterances. Global settings (CUDA device, project paths) remain
in app/config.py; only partial-spoof-specific parameters belong here.
"""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class PartialSpoofSettings(BaseModel):
    """Configuration for the Partial Spoof pipeline.

    Attributes:
        VALIDATION_MODE: When True, process only VALIDATION_SPEAKERS.
        VALIDATION_SPEAKERS: Speaker IDs for quick validation runs.
        ATTACK_SYSTEM: Voice cloning system to use as the attack strategy.
        BONAFIDE_DIR: Root directory containing HABLA bonafide speaker folders.
        OUTPUT_DIR: Base output directory for pipeline artifacts.
        SAMPLE_RATE: Target sample rate in Hz for all audio processing.
        DEVICE: PyTorch device string for GPU/CPU selection.
        PARAKEET_MODEL_ID: HuggingFace model ID for Parakeet TDT ASR.
        MIN_WORDS_W1: Minimum word count in transcript for W1 tier eligibility.
        MIN_WORDS_W2: Minimum word count in transcript for W2 tier eligibility.
        MIN_WORDS_W3: Minimum word count in transcript for W3 tier eligibility.
        RANDOM_SEED: Seed for reproducible random word selection.
        REQUIRE_NON_ADJACENT: Enforce non-adjacent word selection constraint.
        ALIGNMENT_ENGINE: Forced alignment backend identifier.
        CROSSFADE_MS: Duration of crossfade window at splice boundaries in ms.
        MAX_SILENCE_STEAL_MS: Maximum silence to steal from adjacent pauses in ms.
        MAX_DURATION_STRETCH_RATIO: Maximum allowed time-compression ratio for
            cloned segments that exceed original duration.
        ENABLE_SPLICE_RETRY: Enable re-generation on splice quality failure.
        MAX_SPLICE_RETRIES: Maximum retry attempts per splice when enabled.
        REFERENCE_DURATION_TARGET: Target duration in seconds for speaker
            reference audio clips used by voice cloning systems.
        AUDIO_ID_START_W1: Starting audio ID counter for W1 tier samples.
        AUDIO_ID_START_W2: Starting audio ID counter for W2 tier samples.
        AUDIO_ID_START_W3: Starting audio ID counter for W3 tier samples.
        ENABLED_TIERS: List of active tier identifiers to generate.
    """

    # === Validation Mode ===
    VALIDATION_MODE: bool = Field(
        default=True,
        description="When True, process only VALIDATION_SPEAKERS for quick testing.",
    )
    VALIDATION_SPEAKERS: List[str] = Field(
        default=["arf_00295", "arf_00610", "arf_01523"],
        description="Speaker IDs used in validation mode.",
    )

    # === Attack System ===
    ATTACK_SYSTEM: str = Field(
        default="fishgram",
        description="Voice cloning system: fishgram, qwen, cosyvoice, outetts, chatterbox, openvoice.",
    )

    # === Directory Paths ===
    BONAFIDE_DIR: Path = Field(
        default=Path("data/bonafide_dataset_by_speaker"),
        description="Root directory with HABLA bonafide speaker subfolders.",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("data/partial_spoof_output"),
        description="Base output directory for pipeline artifacts.",
    )
    SAMPLE_RATE: int = Field(
        default=16000,
        description="Target sample rate in Hz for all audio.",
    )
    DEVICE: str = Field(
        default="cuda:0",
        description="PyTorch device for model inference.",
    )

    # === Transcription (Step 1) ===
    PARAKEET_MODEL_ID: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="HuggingFace model ID for Parakeet TDT ASR.",
    )

    # === Tier Word Count Minimums (Step 4) ===
    MIN_WORDS_W1: int = Field(
        default=4,
        description="Minimum transcript word count for W1 tier (1 word replaced).",
    )
    MIN_WORDS_W2: int = Field(
        default=8,
        description="Minimum transcript word count for W2 tier (2 words replaced).",
    )
    MIN_WORDS_W3: int = Field(
        default=12,
        description="Minimum transcript word count for W3 tier (3 words replaced).",
    )

    # === Word Selection (Step 4) ===
    RANDOM_SEED: int = Field(
        default=42,
        description="Random seed for reproducible word selection.",
    )
    REQUIRE_NON_ADJACENT: bool = Field(
        default=True,
        description="Enforce that selected word indices differ by at least 2.",
    )

    # === Forced Alignment (Step 3) ===
    ALIGNMENT_ENGINE: str = Field(
        default="torchaudio_mms",
        description="Forced alignment backend: torchaudio_mms, whisper_timestamps, mfa.",
    )

    # === Splicing (Step 5) ===
    CROSSFADE_MS: float = Field(
        default=5.0,
        description="Crossfade window duration at splice boundaries in milliseconds.",
    )
    MAX_SILENCE_STEAL_MS: float = Field(
        default=50.0,
        description="Maximum silence to absorb from adjacent pauses in milliseconds.",
    )
    MAX_DURATION_STRETCH_RATIO: float = Field(
        default=1.1,
        description="Maximum time-compression ratio (1.1 = 10% compression allowed).",
    )

    # === Splice Quality Validation (Step 6) ===
    ENABLE_SPLICE_RETRY: bool = Field(
        default=False,
        description="Re-generate cloned speech on splice quality failure.",
    )
    MAX_SPLICE_RETRIES: int = Field(
        default=3,
        description="Maximum retry attempts per splice when retry is enabled.",
    )

    # === Reference Audio (Step 2) ===
    REFERENCE_DURATION_TARGET: float = Field(
        default=15.0,
        description="Target duration in seconds for speaker reference clips.",
    )

    # === Output Format (Step 7) ===
    AUDIO_ID_START_W1: int = Field(
        default=12000000,
        description="Starting audio ID counter for W1 tier.",
    )
    AUDIO_ID_START_W2: int = Field(
        default=13000000,
        description="Starting audio ID counter for W2 tier.",
    )
    AUDIO_ID_START_W3: int = Field(
        default=14000000,
        description="Starting audio ID counter for W3 tier.",
    )
    ENABLED_TIERS: List[str] = Field(
        default=["W1", "W2", "W3"],
        description="Active tier identifiers to generate.",
    )


# Module-level singleton
settings = PartialSpoofSettings()
