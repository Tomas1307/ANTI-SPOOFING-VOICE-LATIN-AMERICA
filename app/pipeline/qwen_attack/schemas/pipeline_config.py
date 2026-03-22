"""
Pipeline configuration schema for Qwen attack generation.
"""
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


class QwenPipelineConfig(BaseModel):
    """Configuration for running the Qwen attack pipeline.

    This schema defines the runtime configuration that can be passed to the pipeline.
    Most defaults are defined in settings.py and can be overridden here.

    Attributes:
        run_step_1: Whether to run Step 1 (prepare reference audio with STT transcription).
        run_step_2: Whether to run Step 2 (prepare text prompts).
        run_step_3: Whether to run Step 3 (generate synthetic speech via Qwen3-TTS).
        run_step_4: Whether to run Step 4 (validate quality with artifact detection).
        run_step_5: Whether to run Step 5 (format output to ASVspoof2019 LA).
        samples_per_speaker_override: Optional override for samples per speaker.
        random_seed_override: Optional override for random seed.
        output_dir_override: Optional override for output directory.
        device_override: Optional override for compute device.
    """

    run_step_1: bool = Field(
        default=True,
        description="Run Step 1: Prepare reference audio with STT transcription"
    )
    run_step_2: bool = Field(
        default=True,
        description="Run Step 2: Prepare text prompts"
    )
    run_step_3: bool = Field(
        default=True,
        description="Run Step 3: Generate synthetic speech via Qwen3-TTS"
    )
    run_step_4: bool = Field(
        default=True,
        description="Run Step 4: Validate quality with artifact detection"
    )
    run_step_5: bool = Field(
        default=True,
        description="Run Step 5: Format output to ASVspoof2019 LA"
    )

    samples_per_speaker_override: Optional[int] = Field(
        default=None,
        description="Override samples per speaker (default: from settings)"
    )
    random_seed_override: Optional[int] = Field(
        default=None,
        description="Override random seed (default: from settings)"
    )
    output_dir_override: Optional[Path] = Field(
        default=None,
        description="Override output directory (default: from settings)"
    )
    device_override: Optional[str] = Field(
        default=None,
        description="Override compute device (default: from settings)"
    )
    skip_existing_step_3: bool = Field(
        default=False,
        description="Skip already-generated WAV files in Step 3 (for resume after interruption)"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
