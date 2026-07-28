"""
Pydantic schemas shared across scripts and pipelines (not scoped to a
single attack pipeline).
"""
from app.schemas.cv_parakeet_accent_stat import AccentValidationStat
from app.schemas.cv_parakeet_transcript_outlier import TranscriptOutlier
from app.schemas.cv_parakeet_validation_report import CVParakeetValidationReport

__all__ = [
    "AccentValidationStat",
    "TranscriptOutlier",
    "CVParakeetValidationReport",
]
