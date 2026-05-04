"""
Word bleed manipulation for boundary jitter (Step 5b).

Inserts a fragment of one adjacent word into the other, creating a brief
"pre-echo" or "post-echo" of foreign content at the boundary. Mimics the
tail bleed that crossfade splices leave at the boundary (the audio after
the splice still carries some energy from the previous segment).

Magnitude range covers Spanish VOT plus consonant-vowel transition (20 ms)
up to consonant onset plus vowel attack (60 ms), so the inserted fragment
sounds like a partial phoneme without inserting a complete second phoneme.
"""
from typing import Tuple

import numpy as np


def bleed_at_boundary(
    audio: np.ndarray,
    boundary_sample: int,
    duration_samples: int,
    direction: str,
) -> Tuple[np.ndarray, int]:
    """Insert a fragment of one adjacent word into the other at the boundary.

    Two directions are supported:
        - ``"right_to_left"``: Take the first ``duration_samples`` of the
          right word (after the boundary) and insert them at the end of
          the left word (just before the boundary). The right word stays
          intact; the left word now ends with a pre-echo of the next word.
        - ``"left_to_right"``: Take the last ``duration_samples`` of the
          left word (before the boundary) and insert them at the start of
          the right word (just after the boundary). The left word stays
          intact; the right word now begins with a post-echo of the
          previous word.

    Total audio length increases by ``duration_samples`` because content
    is inserted (not moved).

    Args:
        audio: 1-D float32 audio array.
        boundary_sample: Sample index of the boundary.
        duration_samples: Number of samples to copy and insert.
        direction: Either ``"right_to_left"`` or ``"left_to_right"``.

    Returns:
        Tuple of:
            - Modified audio array (length grew by duration_samples).
            - Length delta in samples (positive, since audio grew).

    Raises:
        ValueError: If direction is invalid or duration_samples is negative.
    """
    if direction not in ("right_to_left", "left_to_right"):
        raise ValueError(
            f"direction must be 'right_to_left' or 'left_to_right', got '{direction}'."
        )
    if duration_samples < 0:
        raise ValueError(
            f"duration_samples must be non-negative, got {duration_samples}."
        )
    if duration_samples == 0:
        return audio, 0

    if direction == "right_to_left":
        copy_start = boundary_sample
        copy_end = min(len(audio), boundary_sample + duration_samples)
        if copy_end <= copy_start:
            return audio, 0
        fragment = audio[copy_start:copy_end].copy()
        new_audio = np.concatenate([
            audio[:boundary_sample],
            fragment,
            audio[boundary_sample:],
        ])
    else:
        copy_start = max(0, boundary_sample - duration_samples)
        copy_end = boundary_sample
        if copy_end <= copy_start:
            return audio, 0
        fragment = audio[copy_start:copy_end].copy()
        new_audio = np.concatenate([
            audio[:boundary_sample],
            fragment,
            audio[boundary_sample:],
        ])

    delta = copy_end - copy_start

    return new_audio, delta
