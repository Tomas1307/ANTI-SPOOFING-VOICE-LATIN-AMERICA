"""
Energy-based refinement of forced-alignment word boundaries.

Parakeet TDT (and any CTC/RNN-T forced aligner) occasionally produces
word timestamps that drift 100-300 ms from the actual acoustic word
position. This is especially common in fast phrases where adjacent
words are pronounced as a unit (e.g. "la casa", "tu casa", "mi cosa"):
the aligner often greedily merges the silence between the two words
into one word's boundary and pushes the next word's start into the
true silence that follows. When the splice engine then trusts the
drifted boundary, the spoofed slot lands in the wrong region: the
real acoustic word survives outside the replaced range, and the
listener hears both the original word AND the cloned word
sequentially.

This module recovers the true acoustic position by:

1. Searching a fixed radius around Parakeet's word centre.
2. Computing a short-time RMS envelope across the search window.
3. Detecting contiguous speech segments above a silence threshold.
4. Picking the segment whose centre is closest to Parakeet's centre.

The selected segment's [start, end] becomes the refined word
boundary. Parakeet's call is only used as a coarse "approximately
here" hint -- the real boundary is the energy-based one.

The function is pure and stateless: takes audio plus the original
boundaries, returns the refined boundaries. The splice engine calls
it before computing slot positions, so the rest of the pipeline
(valley snap, crossfade, stretch ratio gate) operates on accurate
boundaries.
"""
from typing import Tuple

import numpy as np


def refine_word_boundary_by_energy(
    audio: np.ndarray,
    parakeet_start_s: float,
    parakeet_end_s: float,
    sample_rate: int,
    search_radius_s: float = 0.300,
    silence_threshold_rms: float = 0.015,
    min_segment_dur_ratio: float = 0.40,
    merge_gap_ms: float = 60.0,
    window_ms: float = 10.0,
    hop_ms: float = 5.0,
) -> Tuple[float, float]:
    """Refine a forced-alignment word boundary using acoustic energy.

    Two stages:

    1. **Detect speech segments** in a window around Parakeet's centre
       (RMS > ``silence_threshold_rms``). Adjacent segments separated
       by short gaps (< ``merge_gap_ms``) are merged: speech often
       contains brief intra-word silences between phonemes (stops,
       voiceless fricatives) that would otherwise fragment one word
       into multiple short segments.

    2. **Filter and pick**:
       - Reject segments shorter than
         ``min_segment_dur_ratio * parakeet_duration``. Filters out
         room noise, breath, and lip smacks that happen to fall
         close to Parakeet's centre.
       - Among qualified segments, pick the one whose centre is
         closest to Parakeet's centre. That's the segment most
         likely to be the word Parakeet was pointing at, just at a
         shifted position.

    If no segment qualifies, fall back to Parakeet's boundaries so
    the splice still proceeds (possibly with a residual ghost) rather
    than getting skipped at the stretch-ratio gate downstream.

    Args:
        audio: Bonafide waveform (1-D float32, mono).
        parakeet_start_s: Parakeet's word start in seconds.
        parakeet_end_s: Parakeet's word end in seconds.
        sample_rate: Audio sample rate in Hz.
        search_radius_s: Half-width of the search window around
            Parakeet's centre. ``0.0`` disables refinement.
        silence_threshold_rms: RMS classified as silence.
        min_segment_dur_ratio: Minimum segment duration as a fraction
            of Parakeet's claimed word duration. A short artefact
            (e.g. 50 ms breath) near Parakeet's centre would
            otherwise be picked over the real 150 ms word segment,
            shrinking the splice slot below the stretch envelope and
            getting rejected. 0.40 keeps "casa" 150 ms vs. Parakeet
            240 ms (ratio 0.625) while rejecting 50 ms artefacts
            (ratio 0.21).
        merge_gap_ms: Inter-segment silence gap (in ms) below which
            two segments are merged. Plosives, unvoiced fricatives,
            and inter-phoneme TTS pauses commonly create 30-60 ms
            internal silences; without merging they split a word into
            pieces too small to qualify or fragment the segment so
            only half the word is selected. 60 ms catches typical
            stop closures (k, t, p) while still excluding inter-word
            pauses (typically > 100 ms).
        window_ms: RMS analysis window length in milliseconds.
        hop_ms: Step between successive analysis windows.

    Returns:
        Tuple ``(refined_start_s, refined_end_s)``. Returns the input
        unchanged when ``search_radius_s == 0.0`` or no qualifying
        segment is found.
    """
    if search_radius_s <= 0.0:
        return parakeet_start_s, parakeet_end_s

    audio_dur_s = len(audio) / sample_rate
    if audio_dur_s <= 0.0:
        return parakeet_start_s, parakeet_end_s

    parakeet_centre_s = (parakeet_start_s + parakeet_end_s) / 2.0
    parakeet_duration_s = max(0.001, parakeet_end_s - parakeet_start_s)
    min_segment_dur_s = parakeet_duration_s * min_segment_dur_ratio

    search_start_s = max(0.0, parakeet_centre_s - search_radius_s)
    search_end_s = min(audio_dur_s, parakeet_centre_s + search_radius_s)

    if search_end_s - search_start_s < 0.020:
        return parakeet_start_s, parakeet_end_s

    window_samples = max(1, int(window_ms * sample_rate / 1000))
    hop_samples = max(1, int(hop_ms * sample_rate / 1000))
    merge_gap_frames = max(1, int(merge_gap_ms / hop_ms))

    start_sample = int(search_start_s * sample_rate)
    end_sample = min(len(audio), int(search_end_s * sample_rate))

    rms_values = []
    centres_s = []
    for p in range(start_sample, end_sample - window_samples + 1, hop_samples):
        segment = audio[p : p + window_samples]
        if len(segment) == 0:
            continue
        rms = float(np.sqrt(np.mean(segment.astype(np.float32) ** 2) + 1e-12))
        rms_values.append(rms)
        centres_s.append((p + window_samples // 2) / sample_rate)

    if not rms_values:
        return parakeet_start_s, parakeet_end_s

    rms_arr = np.array(rms_values, dtype=np.float32)
    centres_arr = np.array(centres_s, dtype=np.float32)
    is_speech = rms_arr > silence_threshold_rms

    # Detect contiguous speech segments, merging gaps shorter than
    # merge_gap_frames so intra-word stops/fricatives do not split
    # the word into multiple sub-segments.
    segments = []
    n = len(is_speech)
    i = 0
    while i < n:
        if not is_speech[i]:
            i += 1
            continue
        seg_start_idx = i
        seg_end_idx = i
        j = i + 1
        while j < n:
            if is_speech[j]:
                seg_end_idx = j
                j += 1
                continue
            k = j
            while k < n and not is_speech[k] and (k - j) < merge_gap_frames:
                k += 1
            if k < n and is_speech[k]:
                j = k
                continue
            break
        seg_start_s = float(centres_arr[seg_start_idx])
        seg_end_s = float(centres_arr[seg_end_idx])
        if (seg_end_s - seg_start_s) >= min_segment_dur_s:
            segments.append((seg_start_s, seg_end_s))
        i = j

    if not segments:
        return parakeet_start_s, parakeet_end_s

    # Pick the LONGEST qualifying segment, breaking ties by closeness
    # to Parakeet's centre. Rationale: when the search window catches
    # multiple speech regions, the longest one is by far the most
    # likely to be the actual word -- adjacent breath, lip smacks, or
    # TTS padding artefacts produce shorter blips. The previous
    # "closest to centre" heuristic preferred a 50 ms artefact next
    # to Parakeet's drifted centre over the real 150 ms word offset
    # by ~200 ms, blowing the stretch envelope and rejecting the
    # splice. The duration filter already excluded tiny segments;
    # this picks the dominant one from what survived.
    best_seg = max(
        segments,
        key=lambda s: (
            s[1] - s[0],
            -abs((s[0] + s[1]) / 2.0 - parakeet_centre_s),
        ),
    )
    return best_seg[0], best_seg[1]
