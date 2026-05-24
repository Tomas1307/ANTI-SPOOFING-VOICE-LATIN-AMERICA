"""
Step 5: Splice Audio with Retry

For each word selection plan, extracts the selected word segments from
the cloned audio and splices them into the bonafide audio at the aligned
positions. If the splice produces fewer words than the tier requires
(due to missing words in the clone), retries with different word selections
up to MAX_SPLICE_RETRIES times. Rejected samples (tier word count not
met) are kept in splice_rejected.json so the regeneration loop can
schedule them for Step 2 retry.

Resumable via the optional CheckpointManager: every successful splice
commit calls checkpoint.mark_spliced(splice_key) before metadata is
flushed, so a killed run loses at most the in-flight splice.
"""
import json
import random
from pathlib import Path
from typing import Optional

import librosa
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.splice_result import SpliceResult
from app.pipeline.partial_spoof.utils.checkpoint_manager import CheckpointManager
from app.pipeline.partial_spoof.utils.splice_engine import splice_words

MAX_SPLICE_RETRIES = 5

TIER_WORD_COUNT = {"W1": 1, "W2": 2, "W3": 3}


class AudioSplicer:
    """Splices cloned word segments into bonafide audio with retry logic.

    For each selection plan (utterance + tier), attempts to splice the
    requested number of words. If the splice engine cannot find all
    target words in the cloned audio, re-selects different words and
    retries up to MAX_SPLICE_RETRIES times. Samples that cannot meet
    the tier word count after all retries are rejected with metadata
    explaining why.

    Attributes:
        output_dir: Directory for pipeline artifacts.
        attack_system_name: Uppercase name of the attack system for file naming.
        checkpoint: Optional CheckpointManager for per-splice resume.
    """

    def __init__(
        self,
        attack_system_name: str,
        output_dir: Path | None = None,
        checkpoint: Optional[CheckpointManager] = None,
    ) -> None:
        """Initialize audio splicer.

        Args:
            attack_system_name: Uppercase attack system name (e.g., 'FISHGRAM').
            output_dir: Output directory (default: from settings).
            checkpoint: Optional CheckpointManager. When provided, Step 5
                marks every successful spliced WAV for resume and skips
                splice_keys already marked complete on subsequent runs.
        """
        self.attack_system_name = attack_system_name
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.checkpoint = checkpoint

    def execute(self) -> SpliceResult:
        """Splice cloned word segments into bonafide audio for all selections.

        Returns:
            SpliceResult with splicing statistics.
        """
        logger.info("Step 5: Splicing cloned word segments into bonafide audio...")

        spliced_dir = self.output_dir / "spliced"
        spliced_dir.mkdir(parents=True, exist_ok=True)

        selection_path = self.output_dir / "word_selection_metadata.json"
        with open(selection_path, "r", encoding="utf-8") as f:
            selections = json.load(f)

        alignment_path = self.output_dir / "alignment_metadata.json"
        with open(alignment_path, "r", encoding="utf-8") as f:
            alignment_data = json.load(f)

        # Accumulate splice_metadata across regeneration rounds: load prior
        # round's successes from disk so they are not overwritten when this
        # round only processes a regen subset. Without this, the final-round
        # write trashes earlier-round splices and Step 6 sees zero samples
        # even though spliced WAVs sit on disk.
        metadata_path = self.output_dir / "splice_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    splice_metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    f"splice_metadata.json unreadable ({exc}); starting fresh"
                )
                splice_metadata = {}
        else:
            splice_metadata = {}

        # rejected_metadata stays fresh per round: the facade reads
        # splice_rejected.json to decide which samples need the next regen,
        # so accumulating would falsely re-flag already-resolved samples.
        rejected_metadata = {}
        failed_splices = []
        tier_counts = {}
        total_spoof_ratio = 0.0
        total_spliced = 0
        total_retries = 0

        for sample_key, selection_entry in tqdm(selections.items(), desc="Splicing audio"):
            if sample_key not in alignment_data:
                failed_splices.append(sample_key)
                continue

            alignment = alignment_data[sample_key]
            bonafide_path = Path(alignment["bonafide_audio_path"])
            cloned_path = Path(alignment["cloned_audio_path"])

            if not bonafide_path.exists() or not cloned_path.exists():
                logger.warning(f"Audio files missing for {sample_key}")
                failed_splices.append(sample_key)
                continue

            try:
                bonafide_audio, _ = librosa.load(
                    str(bonafide_path), sr=settings.SAMPLE_RATE, mono=True
                )
                cloned_audio, _ = librosa.load(
                    str(cloned_path), sr=settings.SAMPLE_RATE, mono=True
                )
            except Exception as exc:
                logger.error(f"Failed to load audio for {sample_key}: {exc}")
                failed_splices.append(sample_key)
                continue

            for sel in selection_entry["selections"]:
                tier = sel["tier"]
                expected_count = TIER_WORD_COUNT.get(tier, 1)
                splice_key = f"{sample_key}_{tier}"
                output_filename = (
                    f"{self.attack_system_name}_PSW{tier[1]}_{sample_key}.wav"
                )
                output_path = spliced_dir / output_filename

                if (
                    self.checkpoint is not None
                    and self.checkpoint.is_spliced(splice_key)
                    and output_path.exists()
                ):
                    logger.debug(f"Splice already committed (resume): {splice_key}")
                    continue

                best_result = None
                best_details = []
                retry_history = []

                confirmed_indices = []
                remaining_needed = expected_count
                tried_indices = set()

                for attempt in range(MAX_SPLICE_RETRIES + 1):
                    if attempt == 0:
                        candidate_indices = sel["selected_indices"]
                    else:
                        new_picks = self._pick_replacements(
                            total_words=len(alignment["bonafide_words"]),
                            confirmed=confirmed_indices,
                            n_new=remaining_needed,
                            exclude=tried_indices,
                            seed=settings.RANDOM_SEED + hash(splice_key) + attempt,
                            allowed_indices=sel.get("eligible_indices"),
                        )
                        if new_picks is None:
                            retry_history.append({
                                "attempt": attempt,
                                "reason": "No more candidate words available",
                            })
                            break
                        candidate_indices = sorted(confirmed_indices + new_picks)

                    tried_indices.update(candidate_indices)

                    # Mask the sign bit so NumPy's SeedSequence never
                    # sees a negative integer. Python's hash() returns
                    # signed 64-bit ints, so adding it to RANDOM_SEED
                    # routinely produced negative seeds that crashed
                    # default_rng with "expected non-negative integer".
                    raw_seed = settings.RANDOM_SEED + hash(splice_key)
                    safe_seed = raw_seed & ((1 << 63) - 1)

                    try:
                        spliced_audio, splice_details = splice_words(
                            bonafide_audio=bonafide_audio,
                            cloned_audio=cloned_audio,
                            bonafide_words=alignment["bonafide_words"],
                            cloned_words=alignment["cloned_words"],
                            selected_indices=candidate_indices,
                            sample_rate=settings.SAMPLE_RATE,
                            crossfade_min_ms=settings.CROSSFADE_MIN_MS,
                            crossfade_max_ms=settings.CROSSFADE_MAX_MS,
                            max_silence_steal_ms=settings.MAX_SILENCE_STEAL_MS,
                            max_stretch_ratio=settings.MAX_STRETCH_RATIO,
                            splice_seed=safe_seed,
                            valley_search_ms=settings.VALLEY_SEARCH_MS,
                        )
                    except Exception as exc:
                        retry_history.append({
                            "attempt": attempt,
                            "indices": candidate_indices,
                            "reason": f"Splice error: {exc}",
                        })
                        continue

                    if len(splice_details) >= expected_count:
                        best_result = spliced_audio
                        best_details = splice_details
                        if attempt > 0:
                            total_retries += attempt
                        break

                    succeeded = [d["word_index"] for d in splice_details]
                    failed_words = [
                        alignment["bonafide_words"][i]["word"]
                        for i in candidate_indices
                        if i not in succeeded
                    ]

                    confirmed_indices = succeeded
                    remaining_needed = expected_count - len(confirmed_indices)

                    retry_history.append({
                        "attempt": attempt,
                        "indices": candidate_indices,
                        "spliced": len(splice_details),
                        "expected": expected_count,
                        "confirmed": confirmed_indices,
                        "failed_words": failed_words,
                    })

                    if len(splice_details) > len(best_details):
                        best_result = spliced_audio
                        best_details = splice_details

                    if remaining_needed <= 0:
                        break

                if len(best_details) >= expected_count and best_result is not None:
                    sf.write(str(output_path), best_result, settings.SAMPLE_RATE)
                    if self.checkpoint is not None:
                        self.checkpoint.mark_spliced(splice_key)

                    total_duration = len(best_result) / settings.SAMPLE_RATE
                    spoofed_duration = sum(
                        d["bonafide_end_s"] - d["bonafide_start_s"]
                        for d in best_details
                    )
                    spoof_ratio = spoofed_duration / total_duration if total_duration > 0 else 0.0

                    splice_metadata[splice_key] = {
                        "sample_id": splice_key,
                        "speaker_id": selection_entry["speaker_id"],
                        "split": selection_entry["split"],
                        "tier": tier,
                        "attack_system": self.attack_system_name,
                        "bonafide_audio_path": str(bonafide_path),
                        "cloned_audio_path": str(cloned_path),
                        "spliced_audio_path": str(output_path),
                        "transcript": selection_entry["transcript"],
                        "total_words": selection_entry["word_count"],
                        "spoofed_words": best_details,
                        "spoof_word_ratio": len(best_details) / selection_entry["word_count"],
                        "spoof_duration_ratio": spoof_ratio,
                        "total_duration_s": total_duration,
                        "retries_needed": len(retry_history),
                    }

                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    total_spoof_ratio += spoof_ratio
                    total_spliced += 1
                else:
                    rejected_metadata[splice_key] = {
                        "sample_id": splice_key,
                        "speaker_id": selection_entry["speaker_id"],
                        "tier": tier,
                        "expected_words": expected_count,
                        "best_achieved": len(best_details),
                        "retry_history": retry_history,
                        "rejection_reason": (
                            f"Could not splice {expected_count} words after "
                            f"{MAX_SPLICE_RETRIES + 1} attempts (best: {len(best_details)})"
                        ),
                    }
                    logger.debug(
                        f"Rejected {splice_key}: needed {expected_count} words, "
                        f"best was {len(best_details)} after {len(retry_history)} attempts"
                    )

        metadata_path = self.output_dir / "splice_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(splice_metadata, f, ensure_ascii=False, indent=2)

        rejected_path = self.output_dir / "splice_rejected.json"
        with open(rejected_path, "w", encoding="utf-8") as f:
            json.dump(rejected_metadata, f, ensure_ascii=False, indent=2)

        avg_ratio = total_spoof_ratio / total_spliced if total_spliced > 0 else 0.0

        logger.info(
            f"Step 5 complete: {total_spliced} spliced, "
            f"{len(rejected_metadata)} rejected, "
            f"{len(failed_splices)} failed. Tiers: {tier_counts}. "
            f"Avg spoof duration ratio: {avg_ratio:.3f}. "
            f"Total retries: {total_retries}"
        )

        return SpliceResult(
            metadata_path=metadata_path,
            total_spliced=total_spliced,
            failed_splices=failed_splices + list(rejected_metadata.keys()),
            avg_spoof_duration_ratio=avg_ratio,
            tier_counts=tier_counts,
        )

    def _pick_replacements(
        self,
        total_words: int,
        confirmed: list,
        n_new: int,
        exclude: set,
        seed: int,
        allowed_indices: list | None = None,
    ) -> list | None:
        """Pick replacement indices for failed words, keeping confirmed ones.

        Selects n_new new indices that are non-adjacent to each other
        and to the already confirmed indices. Excludes previously tried
        indices to avoid repeating failures. Constrained to allowed_indices
        (valley-score eligible words from Step 4) when provided.

        Args:
            total_words: Total number of words in the utterance.
            confirmed: Indices already confirmed as spliceable.
            n_new: Number of new indices needed.
            exclude: Set of indices already tried (to avoid repeats).
            seed: Random seed for selection.
            allowed_indices: Optional whitelist of eligible word indices from
                Step 4 valley scoring. When provided, only these indices are
                considered as candidates (prevents retrying words with bad
                stretch ratios or valley scores that Step 4 already rejected).

        Returns:
            List of new indices to try, or None if no valid picks remain.
        """
        rng = random.Random(seed)

        blocked = set(exclude)
        for idx in confirmed:
            blocked.add(idx - 1)
            blocked.add(idx)
            blocked.add(idx + 1)

        candidate_pool = allowed_indices if allowed_indices is not None else list(range(total_words))
        available = [i for i in candidate_pool if i not in blocked]

        if len(available) < n_new:
            return None

        for _ in range(100):
            picks = sorted(rng.sample(available, min(n_new, len(available))))

            all_indices = sorted(confirmed + picks)
            is_valid = True
            for i in range(len(all_indices) - 1):
                if all_indices[i + 1] - all_indices[i] < 2:
                    is_valid = False
                    break

            if is_valid:
                return picks

        return None
