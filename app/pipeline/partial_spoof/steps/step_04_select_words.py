"""
Step 4: Select Words to Replace (Valley-Score Based)

For each aligned utterance, scores every word by the depth of energy
valleys at its boundaries in the cloned audio. Words with the deepest
valleys (cleanest cut points) are selected first, subject to the
non-adjacency constraint and tier word count requirements.

Words are filtered by:
- Valley score <= VALLEY_SCORE_THRESHOLD (boundaries must have clear dips)
- Duration >= MIN_WORD_DURATION_MS (very short words are meaningless to replace)
- Stretch ratio within MAX_STRETCH_RATIO (avoid excessive time-stretching)
"""
import json
import random
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.word_selection_result import WordSelectionResult
from app.pipeline.partial_spoof.utils.valley_scorer import ValleyScorer


TIER_CONFIG = {
    "W1": {"count": 1, "min_words_setting": "MIN_WORDS_W1"},
    "W2": {"count": 2, "min_words_setting": "MIN_WORDS_W2"},
    "W3": {"count": 3, "min_words_setting": "MIN_WORDS_W3"},
}


class WordSelector:
    """Selects words to replace using energy valley scores from cloned audio.

    For each aligned utterance, loads the cloned audio, computes valley
    scores at every word boundary, then selects the best-scoring non-adjacent
    words for each tier. Words with poor boundaries (score above threshold),
    short duration, or extreme stretch ratios are filtered out.

    Attributes:
        output_dir: Directory for pipeline artifacts.
        random_seed: Base seed for reproducible selection.
        require_non_adjacent: Enforce non-adjacency constraint.
        enabled_tiers: List of tier identifiers to generate.
        valley_scorer: Scorer instance for boundary energy analysis.
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
        self.valley_scorer = ValleyScorer(
            sample_rate=settings.SAMPLE_RATE,
            window_ms=settings.VALLEY_SCORE_WINDOW_MS,
            frame_ms=settings.VALLEY_SCORE_FRAME_MS,
        )

    def execute(self) -> WordSelectionResult:
        """Generate word selection plans for all aligned utterances.

        Returns:
            WordSelectionResult with selection statistics per tier.
        """
        logger.info("Step 4: Selecting words by valley score...")

        alignment_path = self.output_dir / "alignment_metadata.json"
        with open(alignment_path, "r", encoding="utf-8") as f:
            alignment_data = json.load(f)

        selections = {}
        tier_counts: Dict[str, int] = {tier: 0 for tier in self.enabled_tiers}
        total_selections = 0
        skipped_no_eligible = 0

        for sample_key, entry in tqdm(alignment_data.items(), desc="Selecting words"):
            word_count = entry["word_count"]
            bonafide_words = entry["bonafide_words"]
            cloned_words = entry["cloned_words"]

            cloned_audio = self._load_audio(entry["cloned_audio_path"])
            if cloned_audio is None:
                continue

            valley_scores = self.valley_scorer.score_words(
                cloned_audio=cloned_audio,
                cloned_words=cloned_words,
                bonafide_words=bonafide_words,
                min_duration_ms=settings.MIN_WORD_DURATION_MS,
                max_stretch_ratio=settings.MAX_STRETCH_RATIO,
            )

            eligible = [
                s for s in valley_scores
                if s.eligible and s.combined_score <= settings.VALLEY_SCORE_THRESHOLD
            ]

            if not eligible:
                skipped_no_eligible += 1
                logger.debug(
                    f"No eligible words for {sample_key} "
                    f"(best score: {valley_scores[0].combined_score:.3f})"
                )
                continue

            sample_selections = []

            for tier in self.enabled_tiers:
                tier_cfg = TIER_CONFIG[tier]
                min_words = getattr(settings, tier_cfg["min_words_setting"])
                n_replace = tier_cfg["count"]

                if word_count < min_words:
                    continue

                per_sample_seed = self.random_seed + hash(sample_key + tier) % (2**31)
                rng = random.Random(per_sample_seed)

                selected_indices = self._select_by_valley_score(
                    eligible_words=eligible,
                    n_select=n_replace,
                    rng=rng,
                )

                if selected_indices is None:
                    logger.debug(
                        f"Could not select {n_replace} non-adjacent eligible words "
                        f"from {len(eligible)} candidates for {sample_key} tier {tier}."
                    )
                    continue

                score_lookup = {s.word_index: s for s in valley_scores}
                selected_words = [
                    {
                        "word_index": idx,
                        "word": bonafide_words[idx]["word"],
                        "valley_score": score_lookup[idx].combined_score,
                        "duration_ms": score_lookup[idx].duration_ms,
                        "stretch_ratio": score_lookup[idx].stretch_ratio,
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
            f"Tier breakdown: {tier_counts}. "
            f"Skipped (no eligible words): {skipped_no_eligible}"
        )

        return WordSelectionResult(
            selection_path=selection_path,
            total_selections=total_selections,
            tier_counts=tier_counts,
        )

    def _select_by_valley_score(
        self,
        eligible_words: list,
        n_select: int,
        rng: random.Random,
    ) -> List[int] | None:
        """Select n_select non-adjacent words prioritized by valley score.

        First attempts a greedy pass: iterate through eligible words (sorted
        best-first by score) and pick the first n_select that satisfy the
        non-adjacency constraint. If the greedy pass fails (e.g. best words
        are adjacent), retries with randomized shuffles of the top candidates.

        Args:
            eligible_words: List of ValleyScore objects, pre-sorted by
                combined_score ascending (best first).
            n_select: Number of words to select.
            rng: Seeded Random instance for tie-breaking.

        Returns:
            Sorted list of selected word indices, or None if impossible.
        """
        if len(eligible_words) < n_select:
            return None

        selected = self._greedy_select(eligible_words, n_select)
        if selected is not None:
            return selected

        top_pool = eligible_words[:min(len(eligible_words), n_select * 4)]
        for _ in range(50):
            shuffled = list(top_pool)
            rng.shuffle(shuffled)
            selected = self._greedy_select(shuffled, n_select)
            if selected is not None:
                return selected

        return None

    def _greedy_select(
        self,
        candidates: list,
        n_select: int,
    ) -> List[int] | None:
        """Greedily pick n_select non-adjacent indices from candidates.

        Args:
            candidates: ValleyScore objects in priority order.
            n_select: Number to select.

        Returns:
            Sorted list of selected indices, or None if not enough found.
        """
        selected = []
        for vs in candidates:
            idx = vs.word_index
            if self.require_non_adjacent:
                if any(abs(idx - s) < 2 for s in selected):
                    continue
            selected.append(idx)
            if len(selected) == n_select:
                return sorted(selected)
        return None

    def _load_audio(self, audio_path: str) -> np.ndarray | None:
        """Load audio file, handling path resolution.

        Args:
            audio_path: Relative path from the pipeline output.

        Returns:
            Audio array at SAMPLE_RATE, or None if file not found.
        """
        full_path = Path(audio_path)
        if not full_path.is_absolute():
            full_path = self.output_dir.parent.parent / audio_path

        if not full_path.exists():
            logger.warning(f"Audio file not found: {full_path}")
            return None

        audio, _ = librosa.load(str(full_path), sr=settings.SAMPLE_RATE, mono=True)
        return audio
