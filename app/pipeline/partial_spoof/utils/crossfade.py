"""
Linear crossfade utility for splice boundary smoothing.

Applies a linear fade-out on the tail of the preceding segment and a
linear fade-in on the head of the following segment, summing them in
the overlap region to create a smooth transition.
"""
import numpy as np


def apply_crossfade(
    segment_before: np.ndarray,
    segment_after: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    """Apply linear crossfade between two adjacent audio segments.

    Creates a smooth transition by fading out the end of segment_before
    and fading in the start of segment_after over the specified number
    of samples.

    Args:
        segment_before: Audio samples preceding the splice point (1-D float array).
        segment_after: Audio samples following the splice point (1-D float array).
        crossfade_samples: Number of samples for the crossfade overlap region.

    Returns:
        Concatenated audio with crossfade applied at the junction.

    Raises:
        ValueError: If crossfade_samples exceeds either segment length.
    """
    if crossfade_samples <= 0:
        return np.concatenate([segment_before, segment_after])

    if crossfade_samples > len(segment_before):
        raise ValueError(
            f"Crossfade samples ({crossfade_samples}) exceeds "
            f"segment_before length ({len(segment_before)})"
        )
    if crossfade_samples > len(segment_after):
        raise ValueError(
            f"Crossfade samples ({crossfade_samples}) exceeds "
            f"segment_after length ({len(segment_after)})"
        )

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
