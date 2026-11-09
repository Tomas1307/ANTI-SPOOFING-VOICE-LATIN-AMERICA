"""
Configuration settings for the standalone voice-cloning demo script.

This is a thin, pipeline-scoped settings module (not part of any attack
pipeline) for the `clone_voice.py` CLI: given a reference speaker file
under ``REFERENCES_DIR`` and arbitrary Spanish text, it reuses the
existing OpenVoice ``Cloner`` (app/pipeline/openvoice_attack/utils/cloner.py)
to synthesise speech in that speaker's voice.

Global application settings belong elsewhere; this module only holds the
parameters specific to this demo script.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field


class VoiceDemoSettings(BaseModel):
    """Configuration for the voice-cloning demo CLI.

    Attributes:
        REFERENCES_DIR: Directory containing one reference audio file per
            speaker (e.g. ``ruben.mp4``, ``mery.m4a``). The speaker name
            passed on the command line is matched case-insensitively
            against the file stem.
        OUTPUT_DIR: Directory where synthesised WAV files are written.
        SAMPLE_RATE: Output sample rate in Hz.
        DEVICE: Compute device for model inference, forwarded to the
            underlying OpenVoice Cloner.
    """

    REFERENCES_DIR: Path = Field(
        default=Path("tesis/references"),
        description="Directory holding one reference audio file per speaker",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("data/voice_demo_output"),
        description="Output directory for synthesised WAV files",
    )
    SAMPLE_RATE: int = Field(
        default=16000,
        description="Target output sample rate in Hz",
    )
    DEVICE: str = Field(
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        description="Compute device for model inference",
    )
    class Config:
        """Pydantic model configuration."""

        frozen = False


# Module-level singleton
settings = VoiceDemoSettings()
