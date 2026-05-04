"""
Word overlap manipulation for boundary jitter (Step 5b).

Shifts the right word backward in time so its onset overlaps with the tail
of the left word, then sums the two regions with optional fade shaping.
This mimics the OLA-Hanning crossfade artifact produced by the splice
engine at spoof boundaries.

Magnitude range matches the LlamaPartialSpoof crossfade range
(30-80 ms uniform; Luong et al. 2024) so jittered boundaries are
indistinguishable from splice boundaries in temporal scale.
"""
from typing import Tuple

import numpy as np


def overlap_at_boundary(
    audio: np.ndarray,
    boundary_sample: int,
    overlap_samples: int,
    fade: str = "hanning",
) -> Tuple[np.ndarray, int]:
    """Overlap the tail of the left word with the head of the right word.

    Removes ``overlap_samples`` of audio after the boundary (the right
    word's onset) and sums it onto the equivalent number of samples
    immediately preceding the boundary (the left word's tail), applying
    a fade window to each side. The total audio length decreases by
    ``overlap_samples``.

    Args:
        audio: 1-D float32 audio array.
        boundary_sample: Sample index of the boundary.
        overlap_samples: Number of samples in the overlap region.
        fade: Fade shape applied to both sides of the overlap. ``"hanning"``
            applies a half-Hanning window (matches OLA-Hanning literature);
            ``"linear"`` applies a linear crossfade.

    Returns:
        Tuple of:
            - Modified audio array (length reduced by overlap_samples).
            - Length delta in samples (negative, since audio shrinks).

    Raises:
        ValueError: If fade is not 'hanning' or 'linear', or if
            overlap_samples is negative.
    """
    if fade not in ("hanning", "linear"):
        raise ValueError(f"fade must be 'hanning' or 'linear', got '{fade}'.")
    if overlap_samples < 0:
        raise ValueError(f"overlap_samples must be non-negative, got {overlap_samples}.")
    if overlap_samples == 0:
        return audio, 0

    left_available = boundary_sample
    right_available = len(audio) - boundary_sample
    n = min(overlap_samples, left_available, right_available)
    if n <= 0:
        return audio, 0

    left_tail = audio[boundary_sample - n:boundary_sample].astype(np.float32, copy=True)
    right_head = audio[boundary_sample:boundary_sample + n].astype(np.float32, copy=True)

    if fade == "hanning":
        full_window = np.hanning(2 * n)
        fade_out = full_window[n:]
        fade_in = full_window[:n]
    else:
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)

    blended = left_tail * fade_out + right_head * fade_in

    new_audio = np.concatenate([
        audio[:boundary_sample - n],
        blended,
        audio[boundary_sample + n:],
    ])
    delta = -n

    return new_audio, delta
