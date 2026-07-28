"""
Pydantic schema for a single high-error Parakeet-vs-CV transcript pair.
"""
from pydantic import BaseModel, Field


class TranscriptOutlier(BaseModel):
    """One CV-origin utterance where Parakeet diverges most from ground truth.

    Kept for manual spot-checking: a high WER can indicate either a genuine
    Parakeet transcription error or noisy Common Voice ground truth (crowd
    read text that does not exactly match the printed sentence).

    Attributes:
        sample_key: Sample key as constructed by SampleKeyBuilder, unique
            per bonafide file.
        speaker_id: Corpus speaker identifier (e.g. 'mx_00612').
        accent: Common Voice accent label for this speaker.
        wer: Word Error Rate for this utterance.
        cer: Character Error Rate for this utterance.
        cv_sentence: Original Common Voice ground-truth sentence.
        parakeet_transcript: Parakeet TDT transcript of the same audio.
    """

    sample_key: str = Field(..., description="Unique sample key")
    speaker_id: str = Field(..., description="Corpus speaker identifier")
    accent: str = Field(..., description="Common Voice accent label")
    wer: float = Field(..., ge=0.0, description="Word Error Rate")
    cer: float = Field(..., ge=0.0, description="Character Error Rate")
    cv_sentence: str = Field(..., description="Original Common Voice sentence")
    parakeet_transcript: str = Field(..., description="Parakeet TDT transcript")
