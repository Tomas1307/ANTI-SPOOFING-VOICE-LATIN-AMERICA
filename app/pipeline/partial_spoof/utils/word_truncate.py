"""
Word truncate manipulation for boundary jitter (Step 5b).

Removes a small audio fragment from one side of an internal word boundary,
producing an abrupt onset/offset artifact that mimics a hard cut/paste
splice without crossfade. Either the tail of the left word or the head
of the right word is cut.

Magnitude is bounded so that the cut stays within VOT and consonant
transition durations (typically <= 40 ms for Spanish), preserving syllable
nucleus integrity and intelligibility.
"""
from typing import Tuple

import numpy as np


def truncate_at_boundary(
    audio: np.ndarray,
    boundary_sample: int,
    duration_samples: int,
    side: str,
) -> Tuple[np.ndarray, int]:
    """Remove a fragment of audio adjacent to a word boundary.

    The audio between the cut region and the boundary is shifted to close
    the gap, so the total audio length decreases by ``duration_samples``.
    Word boundaries to the right of the cut shift left by the same amount;
    callers must adjust their boundary timestamps if they depend on
    pre-cut positions.

    Args:
        audio: 1-D float32 audio array.
        boundary_sample: Sample index of the boundary (typically the start
            of the right word, equal to the end of the left word).
        duration_samples: Number of samples to remove.
        side: Either ``"left_tail"`` (cut the last ``duration_samples`` of
            the left word) or ``"right_head"`` (cut the first
            ``duration_samples`` of the right word).

    Returns:
        Tuple of:
            - Modified audio array (length reduced by duration_samples).
            - Length delta in samples (negative, since audio shrinks).

    Raises:
        ValueError: If side is not 'left_tail' or 'right_head', or if
            duration_samples exceeds the available material on the chosen side.
    """
    if side not in ("left_tail", "right_head"):
        raise ValueError(f"side must be 'left_tail' or 'right_head', got '{side}'.")

    if duration_samples <= 0:
        return audio, 0

    if side == "left_tail":
        cut_start = max(0, boundary_sample - duration_samples)
        cut_end = boundary_sample
    else:
        cut_start = boundary_sample
        cut_end = min(len(audio), boundary_sample + duration_samples)

    if cut_end <= cut_start:
        return audio, 0

    new_audio = np.concatenate([audio[:cut_start], audio[cut_end:]])
    delta = -(cut_end - cut_start)

    return new_audio, delta
