"""
Tier eligibility computer for partial spoof manifest planning.

Given a Parakeet TDT word count and the MIN_WORDS_W1/W2/W3 thresholds
from settings, returns the list of tiers a bonafide file is eligible
to produce. Used by both the pre-flight ManifestGenerator (to assign
planned_tiers per row) and by Step 4 (to filter selections when the
file's effective word count drifts from the manifest's cached value).
"""
from typing import List


class TierEligibilityComputer:
    """Determine which W1/W2/W3 tiers a bonafide file is eligible for.

    Eligibility is purely a function of the bonafide word count vs. the
    minimum word-count thresholds defined per tier. The opportunistic
    yield model means we never pad shorter sentences up; we just record
    which tiers each file can support and let yield fall out of the
    HABLA v2 utterance-length distribution.

    Attributes:
        min_words_w1: Minimum word count for W1 tier eligibility.
        min_words_w2: Minimum word count for W2 tier eligibility.
        min_words_w3: Minimum word count for W3 tier eligibility.
    """

    def __init__(
        self,
        min_words_w1: int,
        min_words_w2: int,
        min_words_w3: int,
    ) -> None:
        """Initialise with the three tier word-count thresholds.

        Args:
            min_words_w1: Minimum words for W1 (1 word replaced).
            min_words_w2: Minimum words for W2 (2 words replaced).
            min_words_w3: Minimum words for W3 (3 words replaced).

        Raises:
            ValueError: If thresholds are not strictly increasing.
        """
        if not (min_words_w1 <= min_words_w2 <= min_words_w3):
            raise ValueError(
                f"Tier thresholds must be non-decreasing: "
                f"W1={min_words_w1}, W2={min_words_w2}, W3={min_words_w3}"
            )
        self.min_words_w1 = min_words_w1
        self.min_words_w2 = min_words_w2
        self.min_words_w3 = min_words_w3

    def compute(self, word_count: int) -> List[str]:
        """Return the tier list a file with this word count is eligible for.

        Args:
            word_count: Number of words in the Parakeet TDT transcription.

        Returns:
            Subset of ['W1', 'W2', 'W3']. Empty list if the file is too
            short for any tier (should be excluded from the manifest).
        """
        tiers: List[str] = []
        if word_count >= self.min_words_w1:
            tiers.append("W1")
        if word_count >= self.min_words_w2:
            tiers.append("W2")
        if word_count >= self.min_words_w3:
            tiers.append("W3")
        return tiers

    def is_eligible(self, word_count: int) -> bool:
        """Quick predicate: is this file eligible for any tier?

        Args:
            word_count: Number of words in the transcription.

        Returns:
            True if the file qualifies for at least W1, else False.
        """
        return word_count >= self.min_words_w1
