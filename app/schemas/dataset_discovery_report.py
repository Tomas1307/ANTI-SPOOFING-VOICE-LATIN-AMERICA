"""
Pydantic schema for the file-discovery reconciliation report emitted by
DatasetLoader for each split.
"""
from typing import Dict

from pydantic import BaseModel, Field


class DatasetDiscoveryReport(BaseModel):
    """Reconciliation of every file seen on disk for one split.

    This report exists so that files present on disk but not ingested by the
    pipeline can never be dropped silently. ``audio_file_count`` plus the sum
    of ``skipped_by_extension`` must equal the total number of regular files
    found under the split directory; any non-empty ``skipped_by_extension``
    means audio may be missing from the released corpus.

    Attributes:
        split: Split name ('train', 'dev', or 'eval').
        structure: Directory layout detected ('split-first' or 'speaker-first').
        speaker_count: Number of speaker directories traversed.
        audio_file_count: Number of files accepted as audio and ingested.
        bonafide_count: Files classified as bonafide.
        spoof_count: Files classified as spoof.
        unknown_prefix_count: Files whose name matched neither the 'bonafide_'
            nor the 'spoof_' prefix convention.
        skipped_by_extension: Count of ignored files keyed by lowercase
            extension (for example '.mp3' or '.txt').
    """

    split: str = Field(..., description="Split name: train, dev, or eval")
    structure: str = Field(..., description="split-first or speaker-first")
    speaker_count: int = Field(..., ge=0)
    audio_file_count: int = Field(..., ge=0)
    bonafide_count: int = Field(..., ge=0)
    spoof_count: int = Field(..., ge=0)
    unknown_prefix_count: int = Field(default=0, ge=0)
    skipped_by_extension: Dict[str, int] = Field(default_factory=dict)

    @property
    def skipped_total(self) -> int:
        """Total number of non-audio files ignored during discovery.

        Returns:
            Sum of all per-extension skip counts.
        """
        return sum(self.skipped_by_extension.values())
