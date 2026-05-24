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
    min_speech_segment_ms: float = 40.0,
    window_ms: float = 10.0,
    hop_ms: float = 5.0,
) -> Tuple[float, float]:
    """Refine a forced-alignment word boundary using acoustic energy.

    Searches a fixed radius around Parakeet's word centre, detects
    contiguous speech segments above ``silence_threshold_rms``, and
    returns the segment whose centre is closest to Parakeet's centre.

    The refinement is robust to alignment drift up to roughly
    ``search_radius_s``. If no speech segment is found inside the
    search window (e.g. very quiet word, threshold too high), the
    original Parakeet boundaries are returned unchanged so callers
    fail open instead of producing nonsense.

    Args:
        audio: Bonafide waveform (1-D float32, mono) sampled at
            ``sample_rate``.
        parakeet_start_s: Parakeet's word start in seconds. Used as
            a coarse hint for where to search; the refined start may
            differ by up to ``search_radius_s``.
        parakeet_end_s: Parakeet's word end in seconds. Used together
            with ``parakeet_start_s`` to compute the search centre.
        sample_rate: Audio sample rate in Hz.
        search_radius_s: Half-width of the search window around
            Parakeet's centre. ``0.0`` disables refinement (returns
            the Parakeet boundaries unchanged).
        silence_threshold_rms: RMS value below which a frame is
            classified as silence. Calibrate per dataset: HABLA at
            16 kHz mono sits around 0.005-0.020 for room noise vs.
            0.05+ for clear speech.
        min_speech_segment_ms: Minimum duration for a detected
            segment to qualify as "a word". Filters out clicks,
            breath noise, and lip smacks.
        window_ms: RMS analysis window length in milliseconds.
        hop_ms: Step between successive analysis windows.

    Returns:
        Tuple ``(refined_start_s, refined_end_s)``. Equal to the input
        when ``search_radius_s == 0.0``, when the search window is
        degenerate, or when no qualifying speech segment is found.
    """
    if search_radius_s <= 0.0:
        return parakeet_start_s, parakeet_end_s

    audio_dur_s = len(audio) / sample_rate
    if audio_dur_s <= 0.0:
        return parakeet_start_s, parakeet_end_s

    parakeet_centre_s = (parakeet_start_s + parakeet_end_s) / 2.0
    search_start_s = max(0.0, parakeet_centre_s - search_radius_s)
    search_end_s = min(audio_dur_s, parakeet_centre_s + search_radius_s)

    if search_end_s - search_start_s < 0.020:
        return parakeet_start_s, parakeet_end_s

    window_samples = max(1, int(window_ms * sample_rate / 1000))
    hop_samples = max(1, int(hop_ms * sample_rate / 1000))
    min_segment_frames = max(1, int(min_speech_segment_ms / hop_ms))

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

    segments = []
    i = 0
    n = len(is_speech)
    while i < n:
        if is_speech[i]:
            j = i
            while j < n and is_speech[j]:
                j += 1
            if (j - i) >= min_segment_frames:
                segments.append((float(centres_arr[i]), float(centres_arr[j - 1])))
            i = j
        else:
            i += 1

    if not segments:
        return parakeet_start_s, parakeet_end_s

    best_seg = min(
        segments,
        key=lambda s: abs((s[0] + s[1]) / 2.0 - parakeet_centre_s),
    )
    return best_seg[0], best_seg[1]
