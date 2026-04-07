"""
Schema for metadata of a single partially spoofed utterance.
"""
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from app.pipeline.partial_spoof.schemas.spliced_word_info import SplicedWordInfo


class SpliceMetadataEntry(BaseModel):
    """Complete metadata for one partially spoofed audio sample.

    Each entry represents a bonafide utterance with N words replaced by
    voice-cloned versions, where N is determined by the tier (W1=1, W2=2, W3=3).

    Attributes:
        sample_id: Unique identifier for this partial spoof sample.
        speaker_id: HABLA speaker identifier (e.g., arf_00295).
        split: Dataset split (train, val, test).
        tier: Word replacement tier (W1, W2, W3).
        attack_system: Voice cloning system used (e.g., FISHGRAM).
        bonafide_audio_path: Path to the original bonafide audio file.
        cloned_audio_path: Path to the full-sentence cloned audio file.
        spliced_audio_path: Path to the output partially spoofed audio file.
        transcript: Full text transcript of the utterance.
        total_words: Total number of words in the transcript.
        spoofed_words: List of metadata for each replaced word.
        spoof_word_ratio: Fraction of words replaced (N / total_words).
        spoof_duration_ratio: Fraction of audio duration that is synthetic (derived).
        total_duration_s: Total duration of the output audio in seconds.
    """

    sample_id: str = Field(
        ...,
        description="Unique identifier for this partial spoof sample",
    )
    speaker_id: str = Field(
        ...,
        description="HABLA speaker identifier (e.g., arf_00295)",
    )
    split: str = Field(
        ...,
        description="Dataset split (train, val, test)",
    )
    tier: str = Field(
        ...,
        description="Word replacement tier (W1, W2, W3)",
    )
    attack_system: str = Field(
        ...,
        description="Voice cloning system used (e.g., FISHGRAM)",
    )
    bonafide_audio_path: Path = Field(
        ...,
        description="Path to the original bonafide audio file",
    )
    cloned_audio_path: Path = Field(
        ...,
        description="Path to the full-sentence cloned audio file",
    )
    spliced_audio_path: Path = Field(
        ...,
        description="Path to the output partially spoofed audio file",
    )
    transcript: str = Field(
        ...,
        description="Full text transcript of the utterance",
    )
    total_words: int = Field(
        ...,
        description="Total number of words in the transcript",
    )
    spoofed_words: List[SplicedWordInfo] = Field(
        ...,
        description="List of metadata for each replaced word",
    )
    spoof_word_ratio: float = Field(
        ...,
        description="Fraction of words replaced (N / total_words)",
    )
    spoof_duration_ratio: float = Field(
        ...,
        description="Fraction of audio duration that is synthetic (derived)",
    )
    total_duration_s: float = Field(
        ...,
        description="Total duration of the output audio in seconds",
    )

    class Config:
        """Pydantic model configuration."""

        frozen = False
        arbitrary_types_allowed = True
