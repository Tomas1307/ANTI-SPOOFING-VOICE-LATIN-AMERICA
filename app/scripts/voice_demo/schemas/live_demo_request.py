"""
Pydantic schema for the live partial-spoof demo request.
"""
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class LiveDemoRequest(BaseModel):
    """A single request to build the original/spoof/spoof+jitter splice demo.

    Attributes:
        host_audio_path: Path to the single ~10 s recording that serves as
            both the OmniVoice reference clip and the bonafide host
            utterance to be attacked.
        target_phrases: One word-group per swap, each the exact words to
            remove from the host utterance, in order (e.g.
            ``["300", "mil"]``), matched against Parakeet's word-level
            transcript after accent/punctuation normalisation. Multiple
            entries produce a multi-word splice (W2-style), one per group.
        replacement_phrases: The words that replace each entry of
            ``target_phrases``, in the same order and at the same index
            (e.g. ``["cinco", "millones", "de"]``).
        output_dir: Destination directory for ``original.wav``,
            ``spoof_sin_jitter.wav``, ``spoof_con_jitter.wav``, and the
            intermediate reference/cloned audio.
        tries: Number of independent OmniVoice renders of the replacement
            sentence to generate. OmniVoice exposes no deterministic
            sampling seed, so repeated calls with the same text produce
            different prosody; each candidate is scored by how much
            silence flanks every replacement phrase and the best one is
            used for the final splice. 1 disables the search.
    """

    host_audio_path: Path = Field(..., description="Path to the single host/reference recording")
    target_phrases: list[list[str]] = Field(
        ..., min_length=1, description="One word-group per swap, to remove from the host"
    )
    replacement_phrases: list[list[str]] = Field(
        ..., min_length=1, description="One word-group per swap, replacing the matching target_phrases entry"
    )
    output_dir: Path = Field(..., description="Destination directory for the demo outputs")
    tries: int = Field(default=1, ge=1, description="Number of OmniVoice renders to try, keeping the best")

    @model_validator(mode="after")
    def check_phrase_counts_match(self) -> "LiveDemoRequest":
        """Ensure target_phrases and replacement_phrases are parallel lists.

        Returns:
            The validated model.

        Raises:
            ValueError: If the two lists have different lengths.
        """
        if len(self.target_phrases) != len(self.replacement_phrases):
            raise ValueError(
                f"target_phrases has {len(self.target_phrases)} entries but "
                f"replacement_phrases has {len(self.replacement_phrases)}; they must match 1:1."
            )
        return self
