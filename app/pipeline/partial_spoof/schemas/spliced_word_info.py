"""Schema for metadata about a single spliced word within a partial spoof sample."""
from pydantic import BaseModel, Field


class SplicedWordInfo(BaseModel):
    """Metadata for one word that was replaced in a partial spoof sample.

    Records the original bonafide and cloned alignment boundaries used
    for extraction and splicing, along with the duration ratio and
    crossfade parameters applied at the splice boundaries.

    Attributes:
        word: The word text that was replaced.
        word_index: Zero-based index of the word within the transcript.
        bonafide_start_s: Start time of the word in the bonafide audio.
        bonafide_end_s: End time of the word in the bonafide audio.
        cloned_start_s: Start time of the word in the cloned audio.
        cloned_end_s: End time of the word in the cloned audio.
        duration_ratio: Ratio of cloned word duration to bonafide word duration.
        crossfade_ms: Crossfade duration applied at splice boundaries in ms.
    """

    word: str = Field(
        ...,
        description="The replaced word text.",
    )
    word_index: int = Field(
        ...,
        description="Zero-based word index in the transcript.",
    )
    bonafide_start_s: float = Field(
        ...,
        description="Start time in bonafide audio (seconds).",
    )
    bonafide_end_s: float = Field(
        ...,
        description="End time in bonafide audio (seconds).",
    )
    cloned_start_s: float = Field(
        ...,
        description="Start time in cloned audio (seconds).",
    )
    cloned_end_s: float = Field(
        ...,
        description="End time in cloned audio (seconds).",
    )
    duration_ratio: float = Field(
        ...,
        description="cloned_duration / bonafide_duration.",
    )
    crossfade_ms: float = Field(
        ...,
        description="Crossfade applied at splice boundaries (ms).",
    )
