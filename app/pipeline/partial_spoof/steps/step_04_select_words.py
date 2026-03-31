"""
Step 4: Select Words to Replace

For each aligned utterance, determines eligible tiers based on word count
and randomly selects N word indices per tier. Generates up to 3 selection
plans per utterance (W1=1 word, W2=2 words, W3=3 words).

Word selection uses a seeded RNG for reproducibility and enforces a
non-adjacency constraint so replaced words are distributed across
the utterance rather than forming a contiguous spoofed block.
"""
import json
import random
from pathlib import Path
from typing import Dict, List

from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.word_selection_result import WordSelectionResult


TIER_CONFIG = {
    "W1": {"count": 1, "min_words_setting": "MIN_WORDS_W1"},
    "W2": {"count": 2, "min_words_setting": "MIN_WORDS_W2"},
    "W3": {"count": 3, "min_words_setting": "MIN_WORDS_W3"},
}


class WordSelector:
    """Selects words to replace for partial spoofing at each tier level.

    For each aligned utterance, determines which tiers (W1, W2, W3) are
    eligible based on word count minimums, then selects random non-adjacent
    word indices for each eligible tier.

    Attributes:
        output_dir: Directory for pipeline artifacts.
        random_seed: Base seed for reproducible selection.
        require_non_adjacent: Enforce non-adjacency constraint.
        enabled_tiers: List of tier identifiers to generate.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        random_seed: int | None = None,
        require_non_adjacent: bool | None = None,
        enabled_tiers: List[str] | None = None,
    ) -> None:
        """Initialize word selector.

        Args:
            output_dir: Output directory (default: from settings).
            random_seed: Base random seed (default: from settings).
            require_non_adjacent: Non-adjacency constraint (default: from settings).
            enabled_tiers: Tiers to generate (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.random_seed = random_seed if random_seed is not None else settings.RANDOM_SEED
        self.require_non_adjacent = (
            require_non_adjacent
            if require_non_adjacent is not None
            else settings.REQUIRE_NON_ADJACENT
        )
        self.enabled_tiers = enabled_tiers or settings.ENABLED_TIERS

    def execute(self) -> WordSelectionResult:
        """Generate word selection plans for all aligned utterances.

        Returns:
            WordSelectionResult with selection statistics per tier.
        """
        logger.info("Step 4: Selecting words to replace for partial spoofing...")

        alignment_path = self.output_dir / "alignment_metadata.json"
        with open(alignment_path, "r", encoding="utf-8") as f:
            alignment_data = json.load(f)

        selections = {}
        tier_counts: Dict[str, int] = {tier: 0 for tier in self.enabled_tiers}
        total_selections = 0

        for sample_key, entry in tqdm(alignment_data.items(), desc="Selecting words"):
            word_count = entry["word_count"]
            bonafide_words = entry["bonafide_words"]

            sample_selections = []

            for tier in self.enabled_tiers:
                tier_cfg = TIER_CONFIG[tier]
                min_words = getattr(settings, tier_cfg["min_words_setting"])
                n_replace = tier_cfg["count"]

                if word_count < min_words:
                    continue

                per_sample_seed = self.random_seed + hash(sample_key + tier) % (2**31)
                rng = random.Random(per_sample_seed)

                selected_indices = self._select_non_adjacent(
                    n_select=n_replace,
                    total_words=len(bonafide_words),
                    rng=rng,
                )

                if selected_indices is None:
                    logger.warning(
                        f"Could not select {n_replace} non-adjacent words "
                        f"from {len(bonafide_words)} words for {sample_key} tier {tier}."
                    )
                    continue

                selected_words = [
                    {
                        "word_index": idx,
                        "word": bonafide_words[idx]["word"],
                    }
                    for idx in selected_indices
                ]

                sample_selections.append({
                    "tier": tier,
                    "n_replaced": n_replace,
                    "selected_words": selected_words,
                    "selected_indices": selected_indices,
                })

                tier_counts[tier] += 1
                total_selections += 1

            if sample_selections:
                selections[sample_key] = {
                    "speaker_id": entry["speaker_id"],
                    "split": entry["split"],
                    "transcript": entry["transcript"],
                    "word_count": word_count,
                    "selections": sample_selections,
                }

        selection_path = self.output_dir / "word_selection_metadata.json"
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(selections, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Step 4 complete: {total_selections} selection plans. "
            f"Tier breakdown: {tier_counts}"
        )

        return WordSelectionResult(
            selection_path=selection_path,
            total_selections=total_selections,
            tier_counts=tier_counts,
        )

    def _select_non_adjacent(
        self,
        n_select: int,
        total_words: int,
        rng: random.Random,
    ) -> List[int] | None:
        """Select n_select non-adjacent word indices from [0, total_words).

        Non-adjacent means selected indices must differ by at least 2.
        Uses rejection sampling with a limited number of attempts.

        Args:
            n_select: Number of indices to select.
            total_words: Total available word indices.
            rng: Seeded Random instance.

        Returns:
            Sorted list of selected indices, or None if selection is impossible.
        """
        if not self.require_non_adjacent:
            candidates = list(range(total_words))
            return sorted(rng.sample(candidates, min(n_select, len(candidates))))

        if total_words < (2 * n_select - 1):
            return None

        max_attempts = 100
        for _ in range(max_attempts):
            candidates = sorted(rng.sample(range(total_words), n_select))
            if all(candidates[i + 1] - candidates[i] >= 2 for i in range(len(candidates) - 1)):
                return candidates

        return None
