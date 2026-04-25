"""
Configuration settings for Partial Spoof Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List


class PartialSpoofSettings(BaseModel):
    """Configuration for partial spoof word-level splice attack pipeline.

    This pipeline creates partially spoofed Latin American Spanish audio by
    replacing individual words in bonafide HABLA utterances with voice-cloned
    versions extracted from full-sentence clones. Supports 6 attack systems
    via the Strategy pattern and 3 word replacement tiers (W1, W2, W3).

    Attributes:
        VALIDATION_MODE: Toggle validation mode (3 speakers) vs production (all speakers).
        VALIDATION_SPEAKERS: Specific speakers for validation mode.

        ATTACK_SYSTEM: Voice cloning system to use (overridden by pipeline config).
        BONAFIDE_DIR: Directory containing HABLA bonafide speakers.
        OUTPUT_DIR: Output directory for partial spoof samples.

        SAMPLE_RATE: Target audio processing sample rate (Hz).
        DEVICE: Compute device for model inference.

        PARAKEET_MODEL_ID: Parakeet TDT model ID for transcription and alignment.

        MIN_WORDS_W1: Minimum sentence length for W1 tier (1 word replaced).
        MIN_WORDS_W2: Minimum sentence length for W2 tier (2 words replaced).
        MIN_WORDS_W3: Minimum sentence length for W3 tier (3 words replaced).

        RANDOM_SEED: Random seed for reproducible word selection.
        REQUIRE_NON_ADJACENT: Enforce non-adjacent word selection constraint.

        ALIGNMENT_ENGINE: Forced alignment backend selection.

        CROSSFADE_MIN_MS: Minimum crossfade overlap drawn per splice (ms).
        CROSSFADE_MAX_MS: Maximum crossfade overlap drawn per splice (ms).
        MAX_SILENCE_STEAL_MS: Maximum silence to steal from adjacent pauses.
        MAX_DURATION_STRETCH_RATIO: Maximum time compression ratio for duration mismatch.

        ENABLE_SPLICE_RETRY: Enable retry with different seed on splice quality failure.
        MAX_SPLICE_RETRIES: Maximum retry attempts per splice.

        REFERENCE_DURATION_TARGET: Target duration for reference audio in seconds.

        AUDIO_ID_START_W1: Starting audio ID for W1 tier samples.
        AUDIO_ID_START_W2: Starting audio ID for W2 tier samples.
        AUDIO_ID_START_W3: Starting audio ID for W3 tier samples.
        ENABLED_TIERS: List of tiers to generate.
    """

    # === Validation Mode Toggle ===
    VALIDATION_MODE: bool = Field(
        default=True,
        description="Validation mode: True=3 speakers, False=all 162 speakers",
    )
    VALIDATION_SPEAKERS: List[str] = Field(
        default=["arf_00295", "arf_00610", "arf_01523", "arm_00412", "arm_00780"],
        description="Speakers for validation mode (3 Argentina Female + 2 Argentina Male)",
    )

    # === Attack System Selection ===
    ATTACK_SYSTEM: str = Field(
        default="qwen",
        description="Voice cloning system: fishgram, qwen, cosyvoice, outetts, chatterbox, openvoice",
    )

    # === Directory Paths ===
    BONAFIDE_DIR: Path = Field(
        default=Path("data/bonafide_dataset_by_speaker_v2"),
        description="Directory containing HABLA v2 bonafide speakers (1,567 total)",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("data/partial_spoof_output"),
        description="Output directory for partial spoof samples (templated at runtime)",
    )

    # === Audio Processing ===
    SAMPLE_RATE: int = Field(
        default=16000,
        description="Target audio processing sample rate in Hz",
    )
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Compute device for model inference",
    )

    # === Transcription (Step 1) ===
    PARAKEET_MODEL_ID: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="NVIDIA Parakeet TDT model ID for transcription (supports Spanish, 3.45 pct WER FLEURS)",
    )

    # === Tier Word Count Minimums (Step 4) ===
    MIN_WORDS_W1: int = Field(
        default=4,
        description="Minimum sentence word count for W1 tier (1 word replaced, max 25 pct spoof)",
    )
    MIN_WORDS_W2: int = Field(
        default=8,
        description="Minimum sentence word count for W2 tier (2 words replaced, max 25 pct spoof)",
    )
    MIN_WORDS_W3: int = Field(
        default=12,
        description="Minimum sentence word count for W3 tier (3 words replaced, max 25 pct spoof)",
    )

    # === Word Selection (Step 4) ===
    RANDOM_SEED: int = Field(
        default=42,
        description="Random seed for reproducible word selection across tiers",
    )
    REQUIRE_NON_ADJACENT: bool = Field(
        default=True,
        description="Enforce non-adjacent word indices (differ by >= 2) to prevent contiguous spoof blocks",
    )

    # === Forced Alignment (Step 3) ===
    ALIGNMENT_ENGINE: str = Field(
        default="torchaudio_mms",
        description="Forced alignment backend: torchaudio_mms (default, GPU, Spanish support)",
    )

    # === Clone Quality Gate (between Steps 2 and 3) ===
    ENABLE_CLONE_SIMILARITY_GATE: bool = Field(
        default=True,
        description="Enable ECAPA-TDNN cosine similarity pre-filter between Steps 2 and 3. "
                    "Rejects clones below MIN_CLONE_SIMILARITY before wasting compute on "
                    "alignment and splicing.",
    )
    MIN_CLONE_SIMILARITY: float = Field(
        default=0.60,
        description="Minimum ECAPA-TDNN cosine similarity between bonafide and clone. "
                    "Clones below this are rejected before alignment. "
                    "Production averages: FishGram=0.602, Qwen=0.720, OpenVoice=0.394.",
    )

    # === Valley Score Word Selection (Step 4) ===
    VALLEY_SCORE_WINDOW_MS: float = Field(
        default=100.0,
        description="Window size in ms (+/- around each boundary) for energy valley analysis.",
    )
    VALLEY_SCORE_FRAME_MS: float = Field(
        default=5.0,
        description="Frame size in ms for RMS energy computation within the valley window.",
    )
    VALLEY_SCORE_THRESHOLD: float = Field(
        default=0.65,
        description="Maximum acceptable valley score. Words with combined score above this "
                    "are ineligible for replacement. Lower = stricter (deeper valleys required).",
    )
    MIN_WORD_DURATION_MS: float = Field(
        default=200.0,
        description="Minimum word duration in ms to be eligible for replacement. "
                    "Very short words (e.g. 'el', 'y') produce meaningless splices.",
    )
    MAX_STRETCH_RATIO: float = Field(
        default=1.25,
        description="Maximum duration ratio (cloned/bonafide or inverse) for word eligibility. "
                    "Words requiring time-stretch outside [1/ratio, ratio] are skipped.",
    )

    # === Splicing Parameters (Step 5) ===
    CROSSFADE_MIN_MS: float = Field(
        default=30.0,
        description="Minimum crossfade overlap drawn per splice boundary (ms). "
                    "Actual overlap is drawn from Uniform[CROSSFADE_MIN_MS, CROSSFADE_MAX_MS] "
                    "and then clamped to the available inter-word gap so the cloned word "
                    "content is never truncated.",
    )
    CROSSFADE_MAX_MS: float = Field(
        default=80.0,
        description="Maximum crossfade overlap drawn per splice boundary (ms). "
                    "For CUT_PASTE method this value is ignored (overlap = 0).",
    )
    MAX_SILENCE_STEAL_MS: float = Field(
        default=50.0,
        description="Maximum milliseconds to steal from adjacent silence for duration mismatch",
    )
    MAX_DURATION_STRETCH_RATIO: float = Field(
        default=1.1,
        description="Maximum time compression ratio (1.1 = 10 pct compression allowed)",
    )

    # === Splice Quality Validation (Step 6) ===
    ENABLE_SPLICE_RETRY: bool = Field(
        default=False,
        description="Enable retry with different TTS seed when splice quality fails (placeholder)",
    )
    MAX_SPLICE_RETRIES: int = Field(
        default=3,
        description="Maximum retry attempts per splice before accepting with warning",
    )

    # === Reference Audio (Step 2) ===
    MAX_SAMPLES: int = Field(
        default=0,
        description="Maximum total samples to process (0=unlimited). Useful for quick tests.",
    )

    REFERENCE_DURATION_TARGET: float = Field(
        default=15.0,
        description="Target duration for reference audio clips in seconds",
    )

    # === Fish Speech API (used when ATTACK_SYSTEM=fishgram) ===
    FISH_SPEECH_API_URL: str = Field(
        default="http://localhost:8080",
        description="Fish Speech HTTP API URL for FishGram strategy",
    )

    # === Split Configuration ===
    TRAIN_SPLIT_NAME: str = Field(default="train", description="Directory name for train split")
    VAL_SPLIT_NAME: str = Field(default="val", description="Directory name for validation split")
    TEST_SPLIT_NAME: str = Field(default="test", description="Directory name for test split")

    # === Output Format Configuration (Step 7) ===
    AUDIO_ID_START_W1: int = Field(
        default=12000000,
        description="Starting audio ID for W1 tier (12000000-12999999 range)",
    )
    AUDIO_ID_START_W2: int = Field(
        default=13000000,
        description="Starting audio ID for W2 tier (13000000-13999999 range)",
    )
    AUDIO_ID_START_W3: int = Field(
        default=14000000,
        description="Starting audio ID for W3 tier (14000000-14999999 range)",
    )
    ENABLED_TIERS: List[str] = Field(
        default=["W1", "W2", "W3"],
        description="List of word replacement tiers to generate",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False


# Module-level singleton
settings = PartialSpoofSettings()
