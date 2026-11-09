"""
Pydantic schema for a single voice-cloning demo request.
"""
from pathlib import Path

from pydantic import BaseModel, Field


class CloneRequest(BaseModel):
    """A single resolved request to clone one speaker's voice onto text.

    Attributes:
        speaker_name: Human-readable speaker identifier as passed on the
            command line (e.g. ``"ruben"``, ``"mery"``).
        reference_audio_path: Resolved path to the speaker's original
            reference audio file under ``REFERENCES_DIR``.
        text: Spanish text to synthesise in the speaker's voice.
        output_path: Destination WAV path for the synthesised audio.
    """

    speaker_name: str = Field(..., description="Speaker identifier matched from REFERENCES_DIR")
    reference_audio_path: Path = Field(..., description="Resolved reference audio file path")
    text: str = Field(..., min_length=1, description="Spanish text to synthesise")
    output_path: Path = Field(..., description="Destination WAV path for the synthesised audio")
