"""
Pydantic schema for the partial-spoof per-tier quality statistics report.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class TierQualityStats(BaseModel):
    """Quality-metric statistics for one partial-spoof replacement tier.

    Attributes:
        tier: Tier label (W1, W2, W3, or ALL for the corpus-wide row).
        count: Spliced samples the WER/CER/NISQA/SIM statistics are
            computed over.
        mean_wer: Mean word-error rate against the target transcript.
        std_wer: Standard deviation of word-error rate.
        mean_cer: Mean character-error rate against the target transcript.
        std_cer: Standard deviation of character-error rate.
        mean_nisqa: Mean NISQA no-reference perceptual-quality score.
        std_nisqa: Standard deviation of the NISQA score.
        mean_sim: Mean ECAPA-TDNN cosine speaker similarity of the
            spliced result to the target speaker.
        std_sim: Standard deviation of speaker similarity.
        boundary_word_count: Spoofed-word boundary rows the margin and
            crossfade statistics are computed over. Distinct from
            ``count`` because a sample can carry more than one spoofed
            word (W2, W3).
        mean_margin_before_ms: Mean available silence margin before the
            splice seam, from ``spoofed_words.csv``.
        std_margin_before_ms: Standard deviation of that margin.
        mean_margin_after_ms: Mean available silence margin after the
            splice seam.
        std_margin_after_ms: Standard deviation of that margin.
        mean_crossfade_ms: Mean effective crossfade duration actually
            applied at the seam, bounded by the available silence.
        std_crossfade_ms: Standard deviation of the effective crossfade
            duration.
    """

    tier: str = Field(..., description="Tier label (W1, W2, W3, or ALL).")
    count: int = Field(..., description="Spliced samples in this tier.")
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
    boundary_word_count: int = Field(
        0, description="Spoofed-word rows the boundary statistics cover."
    )
    mean_margin_before_ms: Optional[float] = Field(
        None, description="Mean available silence margin before the seam."
    )
    std_margin_before_ms: Optional[float] = Field(
        None, description="Std of the pre-seam silence margin."
    )
    mean_margin_after_ms: Optional[float] = Field(
        None, description="Mean available silence margin after the seam."
    )
    std_margin_after_ms: Optional[float] = Field(
        None, description="Std of the post-seam silence margin."
    )
    mean_crossfade_ms: Optional[float] = Field(
        None, description="Mean effective crossfade duration applied."
    )
    std_crossfade_ms: Optional[float] = Field(
        None, description="Std of the effective crossfade duration."
    )


class PartialSpoofQualityReport(BaseModel):
    """Per-tier and corpus-wide partial-spoof quality statistics.

    Attributes:
        tier_stats: One entry per tier (W1, W2, W3) plus a final ALL row
            covering the full partial-spoof tier.
        unmatched_samples: Rows in ``corpus_samples.csv`` skipped because
            a WER/CER/NISQA/SIM field was missing or non-numeric.
        unmatched_boundary_words: Rows in ``corpus_spoofed_words.csv``
            skipped for the same reason.
    """

    tier_stats: List[TierQualityStats] = Field(
        default_factory=list, description="Per-tier and overall statistics."
    )
    unmatched_samples: int = Field(
        0, description="Sample rows skipped due to missing/invalid fields."
    )
    unmatched_boundary_words: int = Field(
        0, description="Boundary-word rows skipped due to missing/invalid fields."
    )
