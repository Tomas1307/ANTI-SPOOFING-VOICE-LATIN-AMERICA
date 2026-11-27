"""
Pydantic schema for the partial-spoof per-attack-system quality report.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class SystemQualityStats(BaseModel):
    """Quality-metric statistics for one attack system's partial-spoof output.

    Attributes:
        system: Attack system label (for example OmniVoice, Qwen3-TTS).
        count: Spliced samples the statistics are computed over.
        mean_wer: Mean word-error rate against the target transcript.
        std_wer: Standard deviation of word-error rate.
        mean_cer: Mean character-error rate against the target transcript.
        std_cer: Standard deviation of character-error rate.
        mean_nisqa: Mean NISQA no-reference perceptual-quality score.
        std_nisqa: Standard deviation of the NISQA score.
        mean_sim: Mean ECAPA-TDNN cosine speaker similarity of the
            spliced result to the target speaker.
        std_sim: Standard deviation of speaker similarity.
    """

    system: str = Field(..., description="Attack system label.")
    count: int = Field(..., description="Spliced samples for this system.")
    mean_wer: Optional[float] = Field(None, description="Mean WER.")
    std_wer: Optional[float] = Field(None, description="Std of WER.")
    mean_cer: Optional[float] = Field(None, description="Mean CER.")
    std_cer: Optional[float] = Field(None, description="Std of CER.")
    mean_nisqa: Optional[float] = Field(None, description="Mean NISQA score.")
    std_nisqa: Optional[float] = Field(None, description="Std of NISQA score.")
    mean_sim: Optional[float] = Field(
        None, description="Mean ECAPA-TDNN speaker similarity."
    )
    std_sim: Optional[float] = Field(
        None, description="Std of ECAPA-TDNN speaker similarity."
    )


class PartialSpoofSystemQualityReport(BaseModel):
    """Per-attack-system partial-spoof quality statistics.

    Attributes:
        system_stats: One entry per attack system plus a final ALL row
            covering the full partial-spoof tier.
        unmatched_samples: Rows in ``corpus_samples.csv`` skipped because
            the attack label was unrecognised or every metric field was
            missing or non-numeric.
    """

    system_stats: List[SystemQualityStats] = Field(
        default_factory=list, description="Per-system and overall statistics."
    )
    unmatched_samples: int = Field(
        0, description="Sample rows skipped due to missing/invalid fields."
    )
