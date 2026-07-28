"""
Pydantic schema for the Parakeet-vs-Common-Voice transcript validation report.
"""
from typing import List

from pydantic import BaseModel, Field

from app.schemas.cv_parakeet_accent_stat import AccentValidationStat
from app.schemas.cv_parakeet_transcript_outlier import TranscriptOutlier


class CVParakeetValidationReport(BaseModel):
    """Full report comparing Parakeet TDT transcripts against Common Voice
    ground-truth sentences for the CV-origin subset of the bonafide corpus.

    This does not validate full-spoof target text (which is sourced directly
    from Common Voice sentences, not from Parakeet). It validates the
    transcription/word-alignment stage that partial-spoof depends on, for the
    subset of bonafide utterances where an independent ground truth exists.

    Attributes:
        total_cv_samples: Rows read from the CV selection manifest
            (selected_15340.tsv).
        matched_samples: Rows for which a cached Parakeet transcript was
            found under the constructed sample_key.
        missing_samples: Rows with no cached Parakeet transcript, typically
            because the utterance fell below MIN_WORDS_W1 and was skipped
            during bonafide transcription.
        overall_wer: Mean Word Error Rate across all matched samples.
        overall_cer: Mean Character Error Rate across all matched samples.
        per_accent: WER/CER broken down by Common Voice accent label.
        worst_outliers: The highest-WER matched samples, for manual review.
    """

    total_cv_samples: int = Field(..., ge=0)
    matched_samples: int = Field(..., ge=0)
    missing_samples: int = Field(..., ge=0)
    overall_wer: float = Field(..., ge=0.0)
    overall_cer: float = Field(..., ge=0.0)
    per_accent: List[AccentValidationStat] = Field(default_factory=list)
    worst_outliers: List[TranscriptOutlier] = Field(default_factory=list)
