"""
Crossfade utility for splice boundary smoothing.

Applies a raised-cosine (Hann) crossfade at splice boundaries for
smoother transitions than a linear fade. Also provides zero-crossing
snapping and energy normalization to minimize audible artifacts.
"""
import numpy as np


def apply_crossfade(
    segment_before: np.ndarray,
    segment_after: np.ndarray,
    crossfade_samples: int,
) -> np.ndarray:
    """Apply raised-cosine crossfade between two adjacent audio segments.

    Uses a Hann window shape (raised cosine) instead of linear fades
    for a smoother perceptual transition at splice boundaries.

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

    t = np.linspace(0, np.pi, crossfade_samples, dtype=np.float32)
    fade_out = (0.5 * (1.0 + np.cos(t))).astype(np.float32)
    fade_in = (0.5 * (1.0 - np.cos(t))).astype(np.float32)

    overlap = (
        segment_before[-crossfade_samples:] * fade_out
        + segment_after[:crossfade_samples] * fade_in
    )

    return np.concatenate([
        segment_before[:-crossfade_samples],
        overlap,
        segment_after[crossfade_samples:],
    ])


def find_nearest_zero_crossing(
    audio: np.ndarray,
    position: int,
    search_range: int = 80,
) -> int:
    """Find the nearest zero-crossing to a given sample position.

    A zero-crossing is where the waveform crosses zero amplitude.
    Cutting at a zero-crossing prevents the audible click that occurs
    when splicing at a non-zero amplitude.

    Args:
        audio: Full audio waveform (1-D float array).
        position: Target sample position.
        search_range: Number of samples to search in each direction.

    Returns:
        Sample position of the nearest zero-crossing, or the original
        position if no crossing is found within range.
    """
    if position <= 0 or position >= len(audio) - 1:
        return position

    start = max(1, position - search_range)
    end = min(len(audio) - 1, position + search_range)

    signs = np.sign(audio[start:end])
    crossings = np.where(np.diff(signs) != 0)[0] + start

    if len(crossings) == 0:
        return position

    distances = np.abs(crossings - position)
    return int(crossings[np.argmin(distances)])


def normalize_energy(
    cloned_segment: np.ndarray,
    bonafide_region: np.ndarray,
    margin_samples: int = 160,
) -> np.ndarray:
    """Normalize the energy of a cloned segment to match the bonafide region.

    Computes RMS energy of the bonafide region around the splice point
    and scales the cloned segment to match. Uses a margin at the edges
    of the bonafide region to capture the local energy level.

    Args:
        cloned_segment: The cloned word audio to be normalized.
        bonafide_region: The bonafide audio region being replaced (same word).
        margin_samples: Extra samples from bonafide context for energy estimation.

    Returns:
        Energy-normalized cloned segment.
    """
    if len(cloned_segment) == 0 or len(bonafide_region) == 0:
        return cloned_segment

    rms_bonafide = np.sqrt(np.mean(bonafide_region ** 2))
    rms_cloned = np.sqrt(np.mean(cloned_segment ** 2))

    if rms_cloned < 1e-8 or rms_bonafide < 1e-8:
        return cloned_segment

    scale = rms_bonafide / rms_cloned
    return (cloned_segment * scale).astype(np.float32)
