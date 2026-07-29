"""
Pydantic schemas shared across scripts and pipelines (not scoped to a
single attack pipeline).
"""
from app.schemas.corpus_duration_report import CorpusDurationReport
from app.schemas.corpus_tier_duration_stat import TierDurationStat
from app.schemas.cv_parakeet_accent_stat import AccentValidationStat
from app.schemas.cv_parakeet_transcript_outlier import TranscriptOutlier
from app.schemas.cv_parakeet_validation_report import CVParakeetValidationReport
from app.schemas.fullspoof_system_description_stat import SystemDescriptionStat

__all__ = [
    "AccentValidationStat",
    "TranscriptOutlier",
    "CVParakeetValidationReport",
    "CorpusDurationReport",
    "TierDurationStat",
    "SystemDescriptionStat",
]
