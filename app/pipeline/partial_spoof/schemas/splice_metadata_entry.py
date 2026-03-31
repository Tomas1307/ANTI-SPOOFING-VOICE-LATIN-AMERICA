"""Schema for complete metadata of a single partial spoof sample."""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from app.pipeline.partial_spoof.schemas.spliced_word_info import SplicedWordInfo


class SpliceMetadataEntry(BaseModel):
    """Full metadata record for one partially spoofed audio sample.

    Tracks the source bonafide utterance, the attack system used, which
    words were replaced, and both word-count and duration-based spoof ratios.

    Attributes:
        sample_id: Unique identifier for this partial spoof sample.
        speaker_id: HABLA speaker identifier (e.g., 'arf_00295').
        split: Dataset split ('train', 'val', 'test').
        tier: Word replacement tier ('W1', 'W2', 'W3').
        attack_system: Name of the voice cloning system used.
        bonafide_audio_path: Path to the source bonafide audio file.
        cloned_audio_path: Path to the full cloned audio file.
        spliced_audio_path: Path to the output partially spoofed audio.
        transcript: Full transcript of the utterance.
        total_words: Total word count in the transcript.
        spoofed_words: List of metadata entries for each replaced word.
        spoof_word_ratio: Fraction of words that are spoofed (N/total).
        spoof_duration_ratio: Fraction of audio duration that is synthetic.
        total_duration_s: Total duration of the output audio in seconds.
    """

    sample_id: str = Field(
        ...,
        description="Unique sample identifier.",
    )
    speaker_id: str = Field(
        ...,
        description="HABLA speaker identifier.",
    )
    split: str = Field(
        ...,
        description="Dataset split (train, val, test).",
    )
    tier: str = Field(
        ...,
        description="Replacement tier (W1, W2, W3).",
    )
    attack_system: str = Field(
        ...,
        description="Voice cloning system name.",
    )
    bonafide_audio_path: Path = Field(
        ...,
        description="Path to source bonafide audio.",
    )
    cloned_audio_path: Path = Field(
        ...,
        description="Path to full cloned audio.",
    )
    spliced_audio_path: Path = Field(
        ...,
        description="Path to output partial spoof audio.",
    )
    transcript: str = Field(
        ...,
        description="Full utterance transcript.",
    )
    total_words: int = Field(
        ...,
        description="Total words in transcript.",
    )
    spoofed_words: List[SplicedWordInfo] = Field(
        ...,
        description="Metadata for each replaced word.",
    )
    spoof_word_ratio: float = Field(
        ...,
        description="Spoofed words / total words.",
    )
    spoof_duration_ratio: float = Field(
        ...,
        description="Spoofed audio duration / total duration.",
    )
    total_duration_s: float = Field(
        ...,
        description="Output audio duration in seconds.",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
