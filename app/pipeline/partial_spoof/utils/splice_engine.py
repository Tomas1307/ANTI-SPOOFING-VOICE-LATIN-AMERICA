"""
Core word-level audio splicing engine.

Replaces selected word segments in bonafide audio with corresponding
segments from cloned audio, handling duration mismatches via silence
stealing and time compression. Processes replacements in reverse order
to preserve sample positions.
"""
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple

from app.pipeline.partial_spoof.utils.crossfade import (
    apply_crossfade,
    find_nearest_zero_crossing,
    normalize_energy,
)


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


def splice_words(
    bonafide_audio: np.ndarray,
    cloned_audio: np.ndarray,
    bonafide_words: List[Dict],
    cloned_words: List[Dict],
    selected_indices: List[int],
    sample_rate: int,
    crossfade_ms: float,
    max_silence_steal_ms: float,
    max_stretch_ratio: float,
) -> Tuple[np.ndarray, List[Dict]]:
    """Replace selected words in bonafide audio with cloned word segments.

    Uses text-based matching to find the corresponding word in the cloned
    audio rather than positional index matching. This prevents splicing
    wrong word content when the clone has fewer or different words.

    Processes word replacements in reverse order (right-to-left) so that
    earlier splices do not shift sample positions of later words.

    Word dicts must have keys: 'word' (str), 'start' (float seconds),
    'end' (float seconds).

    Args:
        bonafide_audio: Full bonafide waveform (1-D float32 array).
        cloned_audio: Full cloned waveform (1-D float32 array).
        bonafide_words: Word-level timestamps for bonafide audio.
        cloned_words: Word-level timestamps for cloned audio.
        selected_indices: Zero-based indices of words to replace.
        sample_rate: Audio sample rate in Hz.
        crossfade_ms: Crossfade duration at splice boundaries (ms).
        max_silence_steal_ms: Max silence to steal from adjacent pauses (ms).
        max_stretch_ratio: Max time compression ratio for duration mismatch.

    Returns:
        Tuple of (spliced_audio, splice_details) where splice_details is a
        list of dicts with per-word splice metadata.
    """
    crossfade_samples = int(crossfade_ms * sample_rate / 1000)
    max_silence_steal_samples = int(max_silence_steal_ms * sample_rate / 1000)

    cloned_map = _build_cloned_word_map(cloned_words)

    result = bonafide_audio.copy()
    splice_details = []

    for idx in sorted(selected_indices, reverse=True):
        if idx >= len(bonafide_words):
            logger.warning(
                f"Word index {idx} out of range for bonafide ({len(bonafide_words)} words). Skipping."
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

        b_start_raw = _clamp(int(bw["start"] * sample_rate), 0, len(result))
        b_end_raw = _clamp(int(bw["end"] * sample_rate), b_start_raw, len(result))
        c_start_raw = _clamp(int(cw["start"] * sample_rate), 0, len(cloned_audio))
        c_end_raw = _clamp(int(cw["end"] * sample_rate), c_start_raw, len(cloned_audio))

        b_start = find_nearest_zero_crossing(result, b_start_raw)
        b_end = find_nearest_zero_crossing(result, b_end_raw)

        if b_start >= b_end:
            b_start, b_end = b_start_raw, b_end_raw

        prev_end = int(cloned_words[cw_idx - 1]["end"] * sample_rate) if cw_idx > 0 else 0
        next_start = int(cloned_words[cw_idx + 1]["start"] * sample_rate) if cw_idx < len(cloned_words) - 1 else len(cloned_audio)
        gap_before = c_start_raw - prev_end
        gap_after = next_start - c_end_raw
        margin_before = min(crossfade_samples, max(0, gap_before))
        margin_after = min(crossfade_samples, max(0, gap_after))

        c_start_margin = _clamp(c_start_raw - margin_before, 0, len(cloned_audio))
        c_end_margin = _clamp(c_end_raw + margin_after, 0, len(cloned_audio))
        c_start = find_nearest_zero_crossing(cloned_audio, c_start_margin)
        c_end = find_nearest_zero_crossing(cloned_audio, c_end_margin)

        if c_start >= c_end:
            c_start, c_end = c_start_margin, c_end_margin

        bf_len = b_end - b_start
        bonafide_region = result[b_start:b_end].copy()
        cloned_segment = cloned_audio[c_start:c_end].copy()
        cl_len = len(cloned_segment)

        if cl_len == 0 or bf_len == 0:
            logger.warning(f"Zero-length segment for word index {idx}, skipping.")
            continue

        cloned_segment = normalize_energy(cloned_segment, bonafide_region)

        cf = min(crossfade_samples, b_start, len(cloned_segment) // 2)

        before = result[:b_start]
        after = result[b_end:]

        if cf > 0 and len(after) >= cf:
            joined = apply_crossfade(before, cloned_segment, cf)
            result = apply_crossfade(joined, after, cf)
        else:
            result = np.concatenate([before, cloned_segment, after])

        splice_details.append({
            "word_index": idx,
            "word": bw["word"],
            "bonafide_start_s": bw["start"],
            "bonafide_end_s": bw["end"],
            "cloned_start_s": cw["start"],
            "cloned_end_s": cw["end"],
            "duration_ratio": round(cl_len / bf_len, 4),
            "crossfade_ms": crossfade_ms,
        })

    splice_details.sort(key=lambda d: d["word_index"])
    return result, splice_details


def _resolve_duration_mismatch(
    result: np.ndarray,
    cloned_segment: np.ndarray,
    b_end: int,
    bf_len: int,
    duration_diff: int,
    max_silence_steal_samples: int,
    max_stretch_ratio: float,
    sample_rate: int,
    word_index: int,
) -> np.ndarray:
    """Resolve duration mismatch when cloned segment is longer than bonafide.

    Strategy (in order of preference):
    1. Steal silence from the gap after the bonafide word.
    2. Time-compress the cloned segment within the allowed ratio.
    3. Truncate the cloned segment as a last resort.

    Args:
        result: Current working waveform.
        cloned_segment: Cloned word segment to fit.
        b_end: End sample position of bonafide word.
        bf_len: Length of bonafide word in samples.
        duration_diff: Excess samples (cloned - bonafide).
        max_silence_steal_samples: Maximum silence samples to absorb.
        max_stretch_ratio: Maximum compression ratio allowed.
        sample_rate: Audio sample rate in Hz.
        word_index: Word index for logging.

    Returns:
        Adjusted cloned segment.
    """
    silence = _measure_silence_after(result, b_end, sample_rate)
    stealable = min(silence, max_silence_steal_samples, duration_diff)
    remaining = duration_diff - stealable

    if remaining <= 0:
        return cloned_segment

    target_len = bf_len + stealable
    ratio = len(cloned_segment) / target_len

    if ratio <= max_stretch_ratio:
        return _time_compress(cloned_segment, target_len)

    logger.warning(
        f"Word index {word_index}: compression ratio {ratio:.2f} exceeds "
        f"max {max_stretch_ratio}. Truncating cloned segment."
    )
    return cloned_segment[:target_len]


def _measure_silence_after(
    audio: np.ndarray,
    position: int,
    sample_rate: int,
    silence_threshold: float = 0.01,
) -> int:
    """Measure consecutive near-silent samples after a given position.

    Args:
        audio: Full audio waveform.
        position: Sample position to start measuring from.
        sample_rate: Audio sample rate in Hz.
        silence_threshold: RMS threshold below which a window is silent.

    Returns:
        Number of consecutive silent samples after position.
    """
    if position >= len(audio):
        return 0

    window_samples = int(0.01 * sample_rate)
    silent_count = 0
    idx = position

    while idx + window_samples <= len(audio):
        window = audio[idx:idx + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms < silence_threshold:
            silent_count += window_samples
            idx += window_samples
        else:
            break

    return silent_count


def _time_compress(segment: np.ndarray, target_length: int) -> np.ndarray:
    """Compress audio segment to target length via linear interpolation.

    Suitable for small adjustments (up to ~10 percent). For larger changes,
    a phase-vocoder approach would preserve quality better.

    Args:
        segment: Audio segment (1-D float32).
        target_length: Desired number of output samples.

    Returns:
        Compressed segment of exactly target_length samples.
    """
    if target_length <= 0 or len(segment) == 0:
        return segment

    indices = np.linspace(0, len(segment) - 1, target_length)
    return np.interp(indices, np.arange(len(segment)), segment).astype(np.float32)


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
