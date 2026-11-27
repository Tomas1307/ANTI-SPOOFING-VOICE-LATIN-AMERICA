"""
Pydantic schema for per-accent WER/CER aggregation.
"""
from pydantic import BaseModel, Field


class AccentValidationStat(BaseModel):
    """Aggregated WER/CER statistics for one Common Voice accent group.

    Attributes:
        accent: Accent label as recorded in cv_speaker_mapping.json
            (e.g. 'Mexico', 'Colombia').
        sample_count: Number of matched CV-origin utterances for this accent.
        mean_wer: Mean Word Error Rate of Parakeet transcripts against the
            original Common Voice sentence, across this accent's samples.
        mean_cer: Mean Character Error Rate, same comparison.
    """

    accent: str = Field(..., description="Common Voice accent label")
    sample_count: int = Field(..., ge=0, description="Matched utterance count")
    mean_wer: float = Field(..., ge=0.0, description="Mean WER for this accent")
    mean_cer: float = Field(..., ge=0.0, description="Mean CER for this accent")
