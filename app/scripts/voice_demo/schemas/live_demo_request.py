"""
Pydantic schema for the live partial-spoof demo request.
"""
from pathlib import Path

from pydantic import BaseModel, Field


class LiveDemoRequest(BaseModel):
    """A single request to build the crude and mitigated splice demos.

    Attributes:
        host_audio_path: Path to the single ~10 s recording that serves as
            both the OmniVoice reference clip and the bonafide host
            utterance to be attacked.
        target_phrase: The exact words to remove from the host utterance,
            in order (e.g. ``["trescientos", "mil"]``), matched against
            Parakeet's word-level transcript after accent/punctuation
            normalisation.
        replacement_phrase: The words that replace ``target_phrase`` in the
            synthesised sentence, in the same order (e.g.
            ``["un", "millon"]``).
        output_dir: Destination directory for ``crudo.wav``, ``mitigado.wav``,
            and the intermediate reference/cloned audio.
    """

    host_audio_path: Path = Field(..., description="Path to the single host/reference recording")
    target_phrase: list[str] = Field(..., min_length=1, description="Words to remove from the host")
    replacement_phrase: list[str] = Field(
        ..., min_length=1, description="Words that replace target_phrase in the clone"
    )
    output_dir: Path = Field(..., description="Destination directory for the demo outputs")
