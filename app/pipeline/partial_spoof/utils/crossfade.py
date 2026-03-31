"""Audio crossfade utility for smooth splice boundary transitions.

Applies linear crossfade at the junction between two audio segments
to eliminate audible clicks and discontinuities at splice points.
"""
import numpy as np


def apply_crossfade(
    segment_before: np.ndarray,
    segment_after: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    """Join two audio segments with a linear crossfade at their boundary.

    Creates a smooth transition by linearly fading out the tail of
    segment_before and fading in the head of segment_after over a
    shared overlap region.

    Args:
        segment_before: Audio samples ending at the splice point.
            Must have at least crossfade_samples elements.
        segment_after: Audio samples starting at the splice point.
            Must have at least crossfade_samples elements.
        crossfade_samples: Number of samples in the crossfade overlap.
            Typically computed from CROSSFADE_MS * SAMPLE_RATE / 1000.

    Returns:
        Concatenated audio with crossfade applied at the junction.

    Raises:
        ValueError: If either segment is shorter than crossfade_samples.
    """
    if len(segment_before) < crossfade_samples:
        raise ValueError(
            f"segment_before ({len(segment_before)} samples) is shorter "
            f"than crossfade_samples ({crossfade_samples})."
        )
    if len(segment_after) < crossfade_samples:
        raise ValueError(
            f"segment_after ({len(segment_after)} samples) is shorter "
            f"than crossfade_samples ({crossfade_samples})."
        )

    if crossfade_samples <= 0:
        return np.concatenate([segment_before, segment_after])

    fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)

    overlap = (
        segment_before[-crossfade_samples:] * fade_out
        + segment_after[:crossfade_samples] * fade_in
    )

    return np.concatenate([
        segment_before[:-crossfade_samples],
        overlap,
        segment_after[crossfade_samples:],
    ])
