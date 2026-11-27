"""
Configuration settings for the standalone voice-cloning demo script.

This is a thin, pipeline-scoped settings module (not part of any attack
pipeline) for the `clone_voice.py` CLI: given a reference speaker file
under ``REFERENCES_DIR`` and arbitrary Spanish text, it reuses the
existing OmniVoice ``Cloner`` (app/pipeline/omnivoice_attack/utils/cloner.py)
to synthesise speech in that speaker's voice.

Global application settings belong elsewhere; this module only holds the
parameters specific to this demo script.
"""
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
        REFERENCE_WAV_SAMPLE_RATE: Sample rate used when transcoding the
            raw reference container (mp4/m4a) to WAV before it reaches
            the cloner.
    """

    REFERENCES_DIR: Path = Field(
        default=Path("tesis/references"),
        description="Directory holding one reference audio file per speaker",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("data/voice_demo_output"),
        description="Output directory for synthesised WAV files",
    )
    REFERENCE_WAV_SAMPLE_RATE: int = Field(
        default=24000,
        description="Sample rate for the transcoded mono WAV reference file",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False


# Module-level singleton
settings = VoiceDemoSettings()
