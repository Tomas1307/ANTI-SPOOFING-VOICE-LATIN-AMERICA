"""
Schema for valley-score word boundary analysis results.
"""
from pydantic import BaseModel, Field


class ValleyScore(BaseModel):
    """Energy valley score for a single word boundary pair.

    Measures how cleanly a word can be cut from the cloned audio by analyzing
    the depth of energy valleys at its start and end boundaries. A deep valley
    (low score) indicates a natural pause where the cut will be inaudible.

    Attributes:
        word_index: Zero-based index of the word in the transcript.
        word: The word text.
        left_score: Valley score at the word's start boundary.
            Ratio of min RMS to avg RMS in a +/-window around the boundary.
            Range [0, 1]. Lower = deeper valley = cleaner cut.
        right_score: Valley score at the word's end boundary.
        combined_score: Worst of left and right scores (max). Both boundaries
            must be clean for the word to be spliceable.
        duration_ms: Duration of the word in the cloned audio (ms).
        stretch_ratio: Ratio of cloned word duration to bonafide word duration.
            Values near 1.0 mean minimal time-stretching is needed.
        eligible: Whether the word passes all selection filters (score, duration,
            stretch ratio).
    """

    word_index: int = Field(
        ...,
        description="Zero-based index of the word in the transcript",
    )
    word: str = Field(
        ...,
        description="The word text",
    )
    left_score: float = Field(
        ...,
        description="Valley score at word start boundary (0=perfect valley, 1=no valley)",
    )
    right_score: float = Field(
        ...,
        description="Valley score at word end boundary (0=perfect valley, 1=no valley)",
    )
    combined_score: float = Field(
        ...,
        description="Worst boundary score: max(left_score, right_score)",
    )
    duration_ms: float = Field(
        ...,
        description="Duration of the word in the cloned audio (ms)",
    )
    stretch_ratio: float = Field(
        default=1.0,
        description="Cloned word duration / bonafide word duration",
    )
    eligible: bool = Field(
        default=True,
        description="Whether the word passes all selection filters",
    )
