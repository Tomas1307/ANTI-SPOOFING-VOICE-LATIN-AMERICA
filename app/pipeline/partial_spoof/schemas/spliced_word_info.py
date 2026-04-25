"""
Schema for metadata about a single spliced (replaced) word.
"""
from typing import Optional
from pydantic import BaseModel, Field


class SplicedWordInfo(BaseModel):
    """Metadata for a single word that was replaced in a partial spoof.

    Captures the original bonafide word position and the cloned word position
    used for replacement, along with splice technique and duration information.

    Attributes:
        word: The word text that was replaced.
        word_index: Zero-based index of the word in the transcript.
        bonafide_start_s: Start time of the word in the bonafide audio (seconds).
        bonafide_end_s: End time of the word in the bonafide audio (seconds).
        cloned_start_s: Start time of the word in the cloned audio (seconds).
        cloned_end_s: End time of the word in the cloned audio (seconds).
        duration_ratio: Ratio of cloned duration to bonafide duration.
        crossfade_ms: Desired crossfade overlap drawn for this splice (ms).
        effective_crossfade_ms: Actual crossfade applied after gap clamping (ms).
            May be less than crossfade_ms when inter-word gap is smaller than desired.
        splice_method: Fade-curve technique used (e.g. 'ola_hanning', 'cosine').
        margin_before_ms: Margin captured before the word start from inter-word gap (ms).
        margin_after_ms: Margin captured after the word end from inter-word gap (ms).
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
        description="Desired crossfade overlap drawn for this splice (milliseconds)",
    )
    effective_crossfade_ms: Optional[float] = Field(
        default=None,
        description="Actual crossfade applied after gap clamping (ms). "
                    "Less than crossfade_ms when inter-word gap is narrower.",
    )
    splice_method: Optional[str] = Field(
        default=None,
        description="Fade-curve technique applied at this splice boundary. "
                    "One of: cut_paste, ola_hanning, linear, cosine, half_sine, "
                    "logarithmic, parabola.",
    )
    margin_before_ms: Optional[float] = Field(
        default=None,
        description="Silence margin captured before the word start (ms). "
                    "Limited by gap to previous cloned word.",
    )
    margin_after_ms: Optional[float] = Field(
        default=None,
        description="Silence margin captured after the word end (ms). "
                    "Limited by gap to next cloned word.",
    )
