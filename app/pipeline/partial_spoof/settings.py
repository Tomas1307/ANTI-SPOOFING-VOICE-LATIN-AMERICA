"""
Configuration settings for Partial Spoof Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple


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
        default=["arf_00295", "clm_00610", "cof_00610", "mxm_00001", "pef_00610"],
        description="Speakers for validation mode (5 countries, mixed gender: Argentina F, Chile M, Colombia F, Mexico M, Peru F)",
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

    # === Bonafide File Partition (Step 1) ===
    BONAFIDE_FILE_PARTITION: str = Field(
        default="not_jittered",
        description="Per-speaker bonafide file partition: 'not_jittered' or 'jittered'. "
                    "Files are shuffled deterministically per speaker (seeded) and split 50/50; "
                    "'not_jittered' takes the first half, 'jittered' takes the second half. "
                    "Ensures the boundary jitter dataset uses sentences disjoint from the "
                    "non-jittered partial spoof dataset.",
    )
    BONAFIDE_PARTITION_SEED: int = Field(
        default=42,
        description="Seed for the per-speaker bonafide file partition shuffle. "
                    "Combined with a deterministic speaker hash so the same speaker always "
                    "yields the same not_jittered/jittered split across runs.",
    )

    # === Manifest-Driven Attack Dispatch ===
    ATTACK_WEIGHTS: Dict[str, float] = Field(
        default={
            "omnivoice": 0.40,
            "qwen": 0.20,
            "fishgram": 0.10,
            "openvoice": 0.10,
            "chatterbox": 0.10,
            "outetts": 0.10,
        },
        description="Probabilistic weights for per-file attack assignment in the "
                    "dispatch manifest. Must sum to 1.0. Applied per-speaker via "
                    "multinomial draw so each speaker's files are distributed "
                    "across attacks in proportion to these weights. Corpus-wide "
                    "marginal converges to these weights under independence.",
    )
    ATTACK_ASSIGNMENT_SEED: int = Field(
        default=42,
        description="Base seed for the per-speaker attack-assignment RNG. "
                    "Combined with sha256(speaker_id) so the same speaker always "
                    "yields the same attack assignment across manifest regenerations. "
                    "Independent of BONAFIDE_PARTITION_SEED so the two stages do not "
                    "couple.",
    )
    MANIFEST_PATH: Path = Field(
        default=Path("data/manifests/partial_spoof_plan.csv"),
        description="Path to the pre-flight dispatch manifest CSV. One row per "
                    "eligible bonafide file with assigned (attack, partition, planned_tiers).",
    )
    MANIFEST_SUMMARY_PATH: Path = Field(
        default=Path("data/manifests/partial_spoof_plan_summary.json"),
        description="Path to the manifest summary JSON sidecar (corpus marginals, "
                    "speaker coverage, target vs actual attack weights).",
    )
    MANIFEST_SLICE_ATTACK: Optional[str] = Field(
        default=None,
        description="Runtime override: restrict the pipeline to the manifest slice "
                    "for this attack only. Set by the facade config; None means "
                    "use settings.ATTACK_SYSTEM as the slice key.",
    )
    MANIFEST_SLICE_PARTITION: Optional[str] = Field(
        default=None,
        description="Runtime override: restrict the pipeline to the manifest slice "
                    "for this partition only. None means use "
                    "settings.BONAFIDE_FILE_PARTITION as the slice key.",
    )

    # === Checkpointing and Retry Budget ===
    ENABLE_CHECKPOINT_RESUME: bool = Field(
        default=True,
        description="When True, Steps 2/5/5b read OUTPUT_DIR/.checkpoint.json on "
                    "startup and skip sample_keys already marked complete. When False, "
                    "the pipeline re-runs all samples from scratch (and overwrites WAVs).",
    )
    MAX_GENERATION_RETRIES: int = Field(
        default=3,
        description="Maximum number of Step 2 regeneration attempts per sample for "
                    "RECOVERABLE errors only (CUDA OOM, model exception, NaN audio, "
                    "zero-byte output). Quality failures are NEVER retried at this "
                    "layer under the keep-bad-stuff principle.",
    )
    MIN_CLONE_DURATION_S: float = Field(
        default=0.5,
        description="Minimum acceptable cloned audio duration in seconds. Clones "
                    "shorter than this are treated as Step 2 generation failures "
                    "(empty/truncated output from diffusion TTS such as OmniVoice) "
                    "and trigger the recoverable-retry loop with a bumped seed. "
                    "0.5 s is a conservative floor -- even a single-syllable "
                    "Spanish word lasts ~150-300 ms, so anything under 500 ms is "
                    "almost certainly degenerate.",
    )
    ENABLE_STEP_6_REJECTION: bool = Field(
        default=False,
        description="When False (default), Step 6 computes WER/CER/NISQA/ECAPA/boundary "
                    "metrics for every sample but does NOT reject any. Every spliced WAV "
                    "lands in the corpus with its quality_flag label. When True (legacy "
                    "behaviour), low-quality samples are filtered out before Step 7.",
    )

    # === Boundary Jitter (Step 5b) ===
    ENABLE_BOUNDARY_JITTER: bool = Field(
        default=False,
        description="Enable Step 5b boundary jitter. When True, after splicing each "
                    "internal word boundary in the utterance is independently subjected to "
                    "a random structural manipulation (truncate, overlap, or bleed) with "
                    "probability JITTER_PROBABILITY. Spoof boundaries receive the same "
                    "treatment so the splice does not stand out as the only manipulated boundary. "
                    "Targets the 'find the noisy boundary' detector shortcut documented in "
                    "Negroni et al. (2024) and Muller (2024).",
    )
    JITTER_PROBABILITY: float = Field(
        default=0.5,
        description="Probability of applying a manipulation per internal boundary. "
                    "0.5 means each boundary independently has a 50 pct chance of being "
                    "left natural and a 50 pct chance of receiving a uniformly chosen "
                    "manipulation (truncate, overlap, or bleed).",
    )
    JITTER_TRUNCATE_RANGE_MS: Tuple[int, int] = Field(
        default=(10, 40),
        description="Uniform random range (ms) for truncate magnitude. "
                    "Lower bound covers Spanish VOT minimum (~4 ms); upper bound stays below "
                    "Spanish syllable nucleus duration (~50-90 ms) to preserve intelligibility.",
    )
    JITTER_OVERLAP_RANGE_MS: Tuple[int, int] = Field(
        default=(30, 80),
        description="Uniform random range (ms) for overlap magnitude. "
                    "Matches the LlamaPartialSpoof crossfade range (Luong et al. 2024) and the "
                    "main partial_spoof CROSSFADE_MIN_MS/MAX_MS so jittered boundaries are "
                    "indistinguishable from splice boundaries in temporal scale.",
    )
    JITTER_BLEED_RANGE_MS: Tuple[int, int] = Field(
        default=(20, 60),
        description="Uniform random range (ms) for bleed magnitude (cross-word fragment insert). "
                    "Lower bound covers Spanish VOT plus consonant transition; upper bound covers "
                    "consonant onset plus vowel attack without inserting a complete second phoneme.",
    )
    JITTER_OVERLAP_FADE: str = Field(
        default="hanning",
        description="Fade shape applied during overlap manipulation. "
                    "'hanning' applies a half-Hanning window to each side of the overlap region "
                    "(matches OLA-Hanning literature); 'linear' applies a linear crossfade.",
    )
    JITTER_SEED: int = Field(
        default=42,
        description="Random seed for boundary jitter decisions and magnitude draws. "
                    "Combined with a sample-key hash so the same utterance always receives "
                    "the same jitter plan across runs.",
    )
    AUDIO_ID_START_W1_JITTER: int = Field(
        default=16000000,
        description="Starting audio ID for W1 tier under boundary jitter "
                    "(16000000-16999999, disjoint from non-jittered W1 at 12M).",
    )
    AUDIO_ID_START_W2_JITTER: int = Field(
        default=17000000,
        description="Starting audio ID for W2 tier under boundary jitter "
                    "(17000000-17999999, disjoint from non-jittered W2 at 13M).",
    )
    AUDIO_ID_START_W3_JITTER: int = Field(
        default=18000000,
        description="Starting audio ID for W3 tier under boundary jitter "
                    "(18000000-18999999, disjoint from non-jittered W3 at 14M).",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False


# Module-level singleton
settings = PartialSpoofSettings()
