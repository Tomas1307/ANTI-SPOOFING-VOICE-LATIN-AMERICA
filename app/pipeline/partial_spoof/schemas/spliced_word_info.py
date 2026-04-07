"""
Schema for metadata about a single spliced (replaced) word.
"""
from pydantic import BaseModel, Field


class SplicedWordInfo(BaseModel):
    """Metadata for a single word that was replaced in a partial spoof.

    Captures the original bonafide word position and the cloned word position
    used for replacement, along with duration information for analysis.

    Attributes:
        word: The word text that was replaced.
        word_index: Zero-based index of the word in the transcript.
        bonafide_start_s: Start time of the word in the bonafide audio (seconds).
        bonafide_end_s: End time of the word in the bonafide audio (seconds).
        cloned_start_s: Start time of the word in the cloned audio (seconds).
        cloned_end_s: End time of the word in the cloned audio (seconds).
        duration_ratio: Ratio of cloned duration to bonafide duration.
        crossfade_ms: Crossfade duration applied at splice boundaries.
    """

    word: str = Field(
        ...,
        description="The word text that was replaced",
    )
    word_index: int = Field(
        ...,
        description="Zero-based index of the word in the transcript",
    )
    bonafide_start_s: float = Field(
        ...,
        description="Start time of the word in the bonafide audio (seconds)",
    )
    bonafide_end_s: float = Field(
        ...,
        description="End time of the word in the bonafide audio (seconds)",
    )
    cloned_start_s: float = Field(
        ...,
        description="Start time of the word in the cloned audio (seconds)",
    )
    cloned_end_s: float = Field(
        ...,
        description="End time of the word in the cloned audio (seconds)",
    )
    duration_ratio: float = Field(
        ...,
        description="Ratio of cloned word duration to bonafide word duration",
    )
    crossfade_ms: float = Field(
        ...,
        description="Crossfade duration applied at splice boundaries (milliseconds)",
    )
