"""
Core word-level audio splicing engine (duration-preserving).

Replaces selected word segments in bonafide audio with corresponding segments
from cloned audio. The cloned word is time-stretched to fit the exact bonafide
word duration, then overwritten in place. Total audio length never changes —
speech rhythm and inter-word gaps are perfectly preserved.

Each splice boundary independently draws a random technique and overlap
duration from a seeded generator, producing a heterogeneous dataset of splice
artifacts.
"""
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple

from app.pipeline.partial_spoof.utils.crossfade import (
    _compute_fade_curves,
    draw_splice_method,
    find_nearest_valley,
    find_nearest_zero_crossing,
    normalize_energy,
)
from app.pipeline.partial_spoof.utils.energy_refiner import (
    refine_word_boundary_by_energy,
)
from app.pipeline.partial_spoof.utils.splice_method import SpliceMethod


def _normalize_word(word: str) -> str:
    """Normalize a word for text-based matching.

    Strips punctuation, lowercases, and removes diacritical marks
    (accents) to handle differences like 'hotel.' vs 'hotel',
    'Hay' vs 'hay', or 'Como' vs 'Cómo'.

    Args:
        word: Raw word string from ASR timestamps.

    Returns:
        Normalized lowercase word without punctuation or accents.
    """
    import unicodedata

    stripped = word.lower().strip(".,;:!?()[]{}\"'¿¡")
    nfkd = unicodedata.normalize("NFKD", stripped)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _build_cloned_word_map(cloned_words: List[Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
    """Build a lookup map from normalized word text to (index, entry) pairs.

    Groups by normalized word text so that duplicate words (e.g. two
    occurrences of 'de') can each be consumed once in order.
    Stores the original index for gap calculation.

    Args:
        cloned_words: Word-level timestamps from cloned audio.

    Returns:
        Dict mapping normalized word text to a list of (index, timestamp_dict) tuples.
    """
    word_map: Dict[str, List[Tuple[int, Dict]]] = {}
    for i, cw in enumerate(cloned_words):
        key = _normalize_word(cw["word"])
        if key not in word_map:
            word_map[key] = []
        word_map[key].append((i, cw))
    return word_map


def _time_stretch(segment: np.ndarray, target_length: int) -> np.ndarray:
    """Time-stretch audio segment to exactly target_length samples.

    Uses linear interpolation for resampling. Suitable for moderate
    stretch ratios (0.75x-1.25x). For extreme ratios, a phase vocoder
    would preserve quality better, but those cases are filtered out
    at selection time.

    Args:
        segment: Audio segment (1-D float32).
        target_length: Desired number of output samples.

    Returns:
        Resampled segment of exactly target_length samples.
    """
    if target_length <= 0 or len(segment) == 0:
        return segment
    if len(segment) == target_length:
        return segment.copy()

    indices = np.linspace(0, len(segment) - 1, target_length)
    return np.interp(indices, np.arange(len(segment)), segment).astype(np.float32)


def splice_words(
    bonafide_audio: np.ndarray,
    cloned_audio: np.ndarray,
    bonafide_words: List[Dict],
    cloned_words: List[Dict],
    selected_indices: List[int],
    sample_rate: int,
    crossfade_min_ms: float,
    crossfade_max_ms: float,
    max_silence_steal_ms: float,
    max_stretch_ratio: float,
    splice_seed: int = 42,
    valley_search_ms: float = 0.0,
    energy_refine_radius_s: float = 0.0,
    energy_refine_silence_rms: float = 0.015,
) -> Tuple[np.ndarray, List[Dict]]:
    """Replace selected words in bonafide audio with cloned word segments.

    Duration-preserving approach: the cloned word is time-stretched to fit
    the exact bonafide word slot, then overwritten in place. The total audio
    length never changes — speech rhythm and gaps are preserved.

    The crossfade happens INSIDE the slot boundaries: the first cf samples
    blend from bonafide into cloned, and the last cf samples blend from
    cloned back into bonafide.

    For each selected word, independently draws:
    - A splice technique (SpliceMethod) sampled from SPLICE_METHOD_WEIGHTS
    - An overlap duration in [crossfade_min_ms, crossfade_max_ms] ms

    Per-word RNG is seeded from (splice_seed, word_index) for reproducibility.

    Word dicts must have keys: 'word' (str), 'start' (float s), 'end' (float s).

    Args:
        bonafide_audio: Full bonafide waveform (1-D float32 array).
        cloned_audio: Full cloned waveform (1-D float32 array).
        bonafide_words: Word-level timestamps for bonafide audio.
        cloned_words: Word-level timestamps for cloned audio.
        selected_indices: Zero-based indices of words to replace.
        sample_rate: Audio sample rate in Hz.
        crossfade_min_ms: Minimum crossfade overlap drawn per splice (ms).
        crossfade_max_ms: Maximum crossfade overlap drawn per splice (ms).
        max_silence_steal_ms: Unused (kept for API compatibility).
        max_stretch_ratio: Maximum acceptable stretch ratio. Words requiring
            stretch outside [1/ratio, ratio] are skipped.
        splice_seed: Base seed for per-word RNG. Seeded as (splice_seed, idx).
        valley_search_ms: Half-width (ms) of the search window used to
            snap each bonafide slot boundary to the nearest energy
            valley. Without this, Parakeet's word boundaries often clip
            inside the acoustic word, so parts of the bonafide word
            survive outside the splice slot and bleed through the
            crossfade -- the listener hears both the cloned word and
            the original word. Setting this to 30-60 ms reliably
            relocates each cut onto a silent inter-word gap; the
            crossfade then mixes cloned speech against bonafide
            silence and only the cloned signal is audible. Set to 0.0
            to disable snapping and preserve the legacy behaviour.
        energy_refine_radius_s: Search radius (seconds) for refining
            each Parakeet word boundary by acoustic energy BEFORE the
            valley snap runs. Parakeet TDT routinely drifts 100-300 ms
            on phrase-merged words ("la casa"); when the drift exceeds
            the valley-snap window, the splice slot lands in the wrong
            region entirely and the bonafide word survives outside it.
            Energy refinement locates the actual word by detecting the
            speech segment closest to Parakeet's centre within +/-
            radius seconds. Set to 0.0 to disable.
        energy_refine_silence_rms: RMS threshold below which audio is
            considered silence in the energy refinement. Tune per
            dataset; ~0.015 works for HABLA at 16 kHz mono.

    Returns:
        Tuple of (spliced_audio, splice_details) where splice_details is a
        list of dicts with per-word splice metadata including splice_method.
    """
    cloned_map = _build_cloned_word_map(cloned_words)

    result = bonafide_audio.copy()
    splice_details = []

    for idx in sorted(selected_indices):
        if idx >= len(bonafide_words):
            logger.warning(
                f"Word index {idx} out of range for bonafide "
                f"({len(bonafide_words)} words). Skipping."
            )
            continue

        bw = bonafide_words[idx]
        target_key = _normalize_word(bw["word"])

        if target_key not in cloned_map or len(cloned_map[target_key]) == 0:
            logger.warning(
                f"Word '{bw['word']}' (index {idx}) not found in cloned audio. Skipping."
            )
            continue

        cw_idx, cw = cloned_map[target_key].pop(0)

        parakeet_start_s = float(bw["start"])
        parakeet_end_s = float(bw["end"])
        parakeet_cloned_start_s = float(cw["start"])
        parakeet_cloned_end_s = float(cw["end"])

        # Energy refinement runs on BOTH sides. Parakeet drifts on
        # natural fast speech (bonafide) and occasionally on TTS
        # output (cloned). Refining only one side leaves a duration
        # mismatch -- the refined bonafide slot can be 150 ms while
        # the cloned source is still 240 ms per Parakeet, blowing
        # past the stretch envelope. Applying the same refinement to
        # the cloned word keeps source and destination acoustically
        # comparable so the resulting splice respects the stretch
        # ratio constraint.
        if energy_refine_radius_s > 0.0:
            refined_start_s, refined_end_s = refine_word_boundary_by_energy(
                bonafide_audio,
                parakeet_start_s,
                parakeet_end_s,
                sample_rate,
                search_radius_s=energy_refine_radius_s,
                silence_threshold_rms=energy_refine_silence_rms,
            )
            refined_cloned_start_s, refined_cloned_end_s = refine_word_boundary_by_energy(
                cloned_audio,
                parakeet_cloned_start_s,
                parakeet_cloned_end_s,
                sample_rate,
                search_radius_s=energy_refine_radius_s,
                silence_threshold_rms=energy_refine_silence_rms,
            )
        else:
            refined_start_s = parakeet_start_s
            refined_end_s = parakeet_end_s
            refined_cloned_start_s = parakeet_cloned_start_s
            refined_cloned_end_s = parakeet_cloned_end_s

        b_start_raw = _clamp(int(refined_start_s * sample_rate), 0, len(result))
        b_end_raw = _clamp(int(refined_end_s * sample_rate), b_start_raw, len(result))
        c_start = _clamp(int(refined_cloned_start_s * sample_rate), 0, len(cloned_audio))
        c_end = _clamp(int(refined_cloned_end_s * sample_rate), c_start, len(cloned_audio))

        cl_raw_len = c_end - c_start

        b_start = b_start_raw
        b_end = b_end_raw
        if valley_search_ms > 0.0 and cl_raw_len > 0:
            # Asymmetric snap: the start boundary may only move EARLIER
            # (further into the inter-word silence preceding the word),
            # the end boundary only LATER. A symmetric search would
            # frequently move a boundary inward and shrink the slot,
            # leaving part of the bonafide word outside the replaced
            # region -- exactly the failure mode we saw on FishGram
            # 'casa' (snap +50 ms inward left the bonafide onset
            # audible). Outward-only guarantees the slot covers the
            # full acoustic word; the crossfade then falls inside
            # silence on both sides and the bonafide signal is gone.
            candidate_start = find_nearest_valley(
                bonafide_audio,
                b_start_raw,
                sample_rate,
                search_ms=valley_search_ms,
                direction="earlier",
            )
            candidate_end = find_nearest_valley(
                bonafide_audio,
                b_end_raw,
                sample_rate,
                search_ms=valley_search_ms,
                direction="later",
            )
            candidate_start = _clamp(candidate_start, 0, len(result))
            candidate_end = _clamp(candidate_end, candidate_start, len(result))

            # Only adopt the snapped boundaries if they keep the
            # required stretch ratio inside the configured envelope.
            # Outward-only expansion enlarges the slot, which can
            # push the required stretch below 1/max_stretch_ratio
            # (cloned must be aggressively stretched to fill). When
            # that happens, falling back to the raw Parakeet
            # boundaries is the lesser evil: the ghost is back but
            # the splice itself succeeds rather than getting rejected.
            candidate_slot_len = candidate_end - candidate_start
            if candidate_slot_len > 0:
                candidate_stretch = cl_raw_len / candidate_slot_len
                if (
                    1.0 / max_stretch_ratio
                    <= candidate_stretch
                    <= max_stretch_ratio
                ):
                    b_start = candidate_start
                    b_end = candidate_end

        slot_len = b_end - b_start

        if slot_len == 0 or cl_raw_len == 0:
            logger.warning(f"Zero-length segment for word index {idx}, skipping.")
            continue

        stretch_ratio = cl_raw_len / slot_len
        if stretch_ratio > max_stretch_ratio or stretch_ratio < (1.0 / max_stretch_ratio):
            logger.warning(
                f"Word '{bw['word']}' (idx={idx}): stretch ratio {stretch_ratio:.2f} "
                f"outside [{1/max_stretch_ratio:.2f}, {max_stretch_ratio:.2f}]. Skipping."
            )
            continue

        cloned_word = cloned_audio[c_start:c_end].copy()
        fitted = _time_stretch(cloned_word, slot_len)

        bonafide_slot = result[b_start:b_end].copy()
        fitted = normalize_energy(fitted, bonafide_slot)

        # NumPy's SeedSequence rejects negative integers ("expected
        # non-negative integer"). Python's built-in hash() can return
        # any signed 64-bit int, so call sites that derive the seed
        # from hash(splice_key) routinely produce negative values --
        # this used to crash entire splice attempts and surface as
        # ``splice_rejected.json`` entries with that exact message.
        # Mask the sign bit to get a stable non-negative seed without
        # losing any of the per-key entropy.
        seed_safe = (splice_seed & ((1 << 63) - 1)) ^ (idx & 0xFFFF)
        word_rng = np.random.default_rng([seed_safe, idx])
        method = draw_splice_method(word_rng)
        overlap_ms = float(word_rng.uniform(crossfade_min_ms, crossfade_max_ms))

        if method is SpliceMethod.CUT_PASTE:
            result[b_start:b_end] = fitted
            effective_cf = 0
        else:
            cf = int(overlap_ms * sample_rate / 1000)
            effective_cf = min(cf, slot_len // 4)

            if effective_cf > 0:
                t = np.linspace(0.0, 1.0, effective_cf, dtype=np.float32)
                fade_in, fade_out = _compute_fade_curves(t, method)

                fitted[:effective_cf] = (
                    bonafide_slot[:effective_cf] * fade_out
                    + fitted[:effective_cf] * fade_in
                )

                fade_in_end, fade_out_end = _compute_fade_curves(t, method)
                fitted[-effective_cf:] = (
                    fitted[-effective_cf:] * fade_out_end
                    + bonafide_slot[-effective_cf:] * fade_in_end
                )

            result[b_start:b_end] = fitted

        splice_details.append({
            "word_index": idx,
            "word": bw["word"],
            "bonafide_start_s": b_start / sample_rate,
            "bonafide_end_s": b_end / sample_rate,
            "bonafide_start_raw_s": b_start_raw / sample_rate,
            "bonafide_end_raw_s": b_end_raw / sample_rate,
            "parakeet_start_s": parakeet_start_s,
            "parakeet_end_s": parakeet_end_s,
            "energy_refine_shift_start_ms": round(
                (refined_start_s - parakeet_start_s) * 1000, 2
            ),
            "energy_refine_shift_end_ms": round(
                (refined_end_s - parakeet_end_s) * 1000, 2
            ),
            "valley_snap_start_ms": round((b_start - b_start_raw) * 1000 / sample_rate, 2),
            "valley_snap_end_ms": round((b_end - b_end_raw) * 1000 / sample_rate, 2),
            "cloned_start_s": c_start / sample_rate,
            "cloned_end_s": c_end / sample_rate,
            "cloned_parakeet_start_s": parakeet_cloned_start_s,
            "cloned_parakeet_end_s": parakeet_cloned_end_s,
            "cloned_refine_shift_start_ms": round(
                (refined_cloned_start_s - parakeet_cloned_start_s) * 1000, 2
            ),
            "cloned_refine_shift_end_ms": round(
                (refined_cloned_end_s - parakeet_cloned_end_s) * 1000, 2
            ),
            "duration_ratio": round(stretch_ratio, 4),
            "stretch_ratio": round(stretch_ratio, 4),
            "crossfade_ms": round(overlap_ms, 2),
            "effective_crossfade_ms": round(effective_cf * 1000 / sample_rate, 2),
            "splice_method": method.value,
            "slot_preserved": True,
        })

        logger.debug(
            f"Spliced word '{bw['word']}' (idx={idx}) | method={method.value} "
            f"| stretch={stretch_ratio:.2f}x | cf={effective_cf}smp "
            f"({round(effective_cf*1000/sample_rate,1)}ms) | slot={slot_len}smp"
        )

    splice_details.sort(key=lambda d: d["word_index"])
    return result, splice_details


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer value to [low, high] range.

    Args:
        value: Value to clamp.
        low: Minimum allowed value.
        high: Maximum allowed value.

    Returns:
        Clamped value.
    """
    return max(low, min(value, high))
