"""
Pydantic schema for a single MARSA protocol row.
"""
from pydantic import BaseModel, Field


class ProtocolEntry(BaseModel):
    """One utterance as declared by the MARSA protocol and metadata CSV.

    The protocol file carries the five ASVspoof2019 LA fields; the companion
    metadata CSV carries the augmentation label and the source utterance
    basename. This model is the joined view of both.

    Attributes:
        audio_id: Clip identifier, for example ``LA_T_0000001``.
        speaker_id: HABLA speaker identifier, for example ``arf_00295``.
        key: Ground-truth label, either ``bonafide`` or ``spoof``.
        attack_id: Generating system slug for spoof clips, ``-`` for bonafide.
        aug_id: Augmentation label, ``-`` for clean copies. Stacked
            augmentations are joined with ``|``.
        source_file: Basename of the source utterance in the speaker-disjoint
            partition. Joins a clip to the strict sentence-disjoint filter.
    """

    audio_id: str = Field(..., description="Clip identifier.")
    speaker_id: str = Field(..., description="HABLA speaker identifier.")
    key: str = Field(..., description="Ground-truth label: bonafide or spoof.")
    attack_id: str = Field(..., description="Attack system slug, or '-'.")
    aug_id: str = Field(default="-", description="Augmentation label, or '-'.")
    source_file: str = Field(default="", description="Source utterance basename.")

    @property
    def label(self) -> int:
        """Return the integer training label.

        Returns:
            1 for bonafide, 0 for spoof. This follows the ASVspoof convention
            in which the countermeasure score is a bonafide likelihood, so a
            higher score means more genuine.
        """
        return 1 if self.key == "bonafide" else 0
