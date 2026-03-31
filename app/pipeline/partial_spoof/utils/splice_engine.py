"""Core audio splicing engine for partial spoof generation.

Handles the mechanics of replacing word segments in bonafide audio
with corresponding segments from cloned audio, including duration
mismatch resolution and crossfade application.
"""
import numpy as np
from typing import Dict, List, Tuple

from loguru import logger

from app.pipeline.partial_spoof.utils.crossfade import apply_crossfade


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

    Processes word replacements in reverse order (right to left) so that
    earlier splice operations do not shift the sample positions of later
    words in the waveform.

    Args:
        bonafide_audio: Full bonafide waveform as 1-D float32 array.
        cloned_audio: Full cloned waveform as 1-D float32 array.
        bonafide_words: List of word dicts with 'word', 'start', 'end' keys.
        cloned_words: List of word dicts with 'word', 'start', 'end' keys.
        selected_indices: Sorted list of word indices to replace.
        sample_rate: Audio sample rate in Hz.
        crossfade_ms: Crossfade duration in milliseconds at splice boundaries.
        max_silence_steal_ms: Maximum silence to steal from adjacent pauses in ms.
        max_stretch_ratio: Maximum time-compression ratio allowed.

    Returns:
        Tuple of:
            - Spliced waveform as 1-D float32 array.
            - List of splice detail dicts for each replaced word, containing
              bonafide/cloned start/end times and the applied crossfade.
    """
    crossfade_samples = int(crossfade_ms * sample_rate / 1000)
    max_silence_steal_samples = int(max_silence_steal_ms * sample_rate / 1000)

    result = bonafide_audio.copy()
    splice_details = []

    for idx in sorted(selected_indices, reverse=True):
        if idx >= len(bonafide_words) or idx >= len(cloned_words):
            logger.warning(
                f"Word index {idx} out of range (bonafide={len(bonafide_words)}, "
                f"cloned={len(cloned_words)}). Skipping."
            )
            continue

        bw = bonafide_words[idx]
        cw = cloned_words[idx]

        b_start_sample = int(bw["start"] * sample_rate)
        b_end_sample = int(bw["end"] * sample_rate)
        c_start_sample = int(cw["start"] * sample_rate)
        c_end_sample = int(cw["end"] * sample_rate)

        b_start_sample = max(0, min(b_start_sample, len(result)))
        b_end_sample = max(b_start_sample, min(b_end_sample, len(result)))
        c_start_sample = max(0, min(c_start_sample, len(cloned_audio)))
        c_end_sample = max(c_start_sample, min(c_end_sample, len(cloned_audio)))

        bonafide_segment_len = b_end_sample - b_start_sample
        cloned_segment = cloned_audio[c_start_sample:c_end_sample].copy()
        cloned_segment_len = len(cloned_segment)

        if cloned_segment_len == 0 or bonafide_segment_len == 0:
            logger.warning(f"Zero-length segment for word index {idx}, skipping.")
            continue

        duration_diff = cloned_segment_len - bonafide_segment_len

        if duration_diff > 0:
            cloned_segment = _handle_longer_cloned(
                result=result,
                cloned_segment=cloned_segment,
                b_end_sample=b_end_sample,
                bonafide_segment_len=bonafide_segment_len,
                duration_diff=duration_diff,
                max_silence_steal_samples=max_silence_steal_samples,
                max_stretch_ratio=max_stretch_ratio,
                sample_rate=sample_rate,
                word_index=idx,
            )

        actual_crossfade = min(
            crossfade_samples,
            b_start_sample,
            len(result) - b_end_sample,
            len(cloned_segment) // 2,
        )

        if actual_crossfade > 0 and len(cloned_segment) > 2 * actual_crossfade:
            before_splice = result[:b_start_sample]
            after_splice = result[b_end_sample:]

            left_joined = apply_crossfade(
                segment_before=before_splice[-(actual_crossfade):] if len(before_splice) >= actual_crossfade else before_splice,
                segment_after=cloned_segment[:actual_crossfade],
                crossfade_samples=min(actual_crossfade, len(before_splice)),
            ) if actual_crossfade > 0 and len(before_splice) >= actual_crossfade else cloned_segment[:0]

            right_joined = apply_crossfade(
                segment_before=cloned_segment[-(actual_crossfade):],
                segment_after=after_splice[:actual_crossfade] if len(after_splice) >= actual_crossfade else after_splice,
                crossfade_samples=min(actual_crossfade, len(after_splice)),
            ) if actual_crossfade > 0 and len(after_splice) >= actual_crossfade else cloned_segment[0:0]

            result = np.concatenate([
                before_splice[:-(actual_crossfade)] if actual_crossfade > 0 else before_splice,
                left_joined,
                cloned_segment[actual_crossfade:-(actual_crossfade)] if actual_crossfade > 0 else cloned_segment,
                right_joined,
                after_splice[actual_crossfade:] if actual_crossfade > 0 else after_splice,
            ])
        else:
            result = np.concatenate([
                result[:b_start_sample],
                cloned_segment,
                result[b_end_sample:],
            ])

        splice_details.append({
            "word_index": idx,
            "word": bw["word"],
            "bonafide_start_s": bw["start"],
            "bonafide_end_s": bw["end"],
            "cloned_start_s": cw["start"],
            "cloned_end_s": cw["end"],
            "duration_ratio": cloned_segment_len / bonafide_segment_len,
            "crossfade_ms": actual_crossfade * 1000 / sample_rate,
        })

    splice_details.reverse()
    return result, splice_details


def _handle_longer_cloned(
    result: np.ndarray,
    cloned_segment: np.ndarray,
    b_end_sample: int,
    bonafide_segment_len: int,
    duration_diff: int,
    max_silence_steal_samples: int,
    max_stretch_ratio: float,
    sample_rate: int,
    word_index: int,
) -> np.ndarray:
    """Handle case where cloned word segment is longer than bonafide.

    Attempts to resolve the duration mismatch by:
    1. Stealing silence from the gap after the word.
    2. Time-compressing the cloned segment within tolerance.
    3. Truncating as a last resort.

    Args:
        result: Current working waveform.
        cloned_segment: Extracted cloned word segment.
        b_end_sample: End sample position of bonafide word.
        bonafide_segment_len: Length of bonafide word segment in samples.
        duration_diff: Excess samples (cloned - bonafide).
        max_silence_steal_samples: Maximum silence samples to absorb.
        max_stretch_ratio: Maximum compression ratio allowed.
        sample_rate: Audio sample rate.
        word_index: Index of the word being processed (for logging).

    Returns:
        Adjusted cloned segment (possibly compressed or truncated).
    """
    silence_after = _measure_silence_after(result, b_end_sample, sample_rate)
    stealable = min(silence_after, max_silence_steal_samples, duration_diff)

    remaining_diff = duration_diff - stealable

    if remaining_diff <= 0:
        return cloned_segment

    target_len = bonafide_segment_len + stealable
    compression_ratio = len(cloned_segment) / target_len

    if compression_ratio <= max_stretch_ratio:
        indices = np.linspace(0, len(cloned_segment) - 1, target_len).astype(int)
        return cloned_segment[indices]

    logger.warning(
        f"Word index {word_index}: cloned segment {len(cloned_segment)} samples "
        f"exceeds bonafide {bonafide_segment_len} by {duration_diff} samples. "
        f"Compression ratio {compression_ratio:.2f} exceeds max {max_stretch_ratio}. "
        f"Truncating to fit."
    )
    return cloned_segment[:target_len]


def _measure_silence_after(
    audio: np.ndarray,
    position: int,
    sample_rate: int,
    window_ms: float = 100.0,
    threshold_ratio: float = 0.02,
) -> int:
    """Measure consecutive near-silent samples after a position.

    Args:
        audio: Full audio waveform.
        position: Sample position to start measuring from.
        sample_rate: Audio sample rate in Hz.
        window_ms: Analysis window in milliseconds.
        threshold_ratio: Silence threshold as fraction of peak amplitude.

    Returns:
        Number of near-silent samples after the position.
    """
    if position >= len(audio):
        return 0

    window_samples = int(window_ms * sample_rate / 1000)
    end = min(position + window_samples, len(audio))
    segment = audio[position:end]

    if len(segment) == 0:
        return 0

    peak = np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1.0
    threshold = peak * threshold_ratio

    silent_count = 0
    for sample in segment:
        if abs(sample) < threshold:
            silent_count += 1
        else:
            break

    return silent_count
