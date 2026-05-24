"""
Crossfade utility for splice boundary smoothing.

Implements seven splice boundary methods (direct cut-paste plus six distinct
fade-curve variants) used to blend bonafide and cloned word segments. Provides
per-splice random method selection weighted by SPLICE_METHOD_WEIGHTS, along with
zero-crossing snap and RMS energy normalization helpers.

Fade-curve summary for t in [0, 1] (fade_in shown; fade_out = fade_in(1-t)):
    LINEAR     : t                          equal-gain, straight diagonal
    OLA_HANNING: 0.5*(1 - cos(pi*t))       equal-gain, smooth S-curve
    COSINE     : sin(pi*t/2)               equal-power, quarter-sine concave
    HALF_SINE  : sqrt(t)                   equal-power, square-root law
    LOGARITHMIC: log(1 + 9*t) / log(10)   aggressive initial rise, plateau
    PARABOLA   : 1 - (1-t)^2              equal-gain, concave inverted parabola
"""
import numpy as np

from app.pipeline.partial_spoof.utils.splice_method import SpliceMethod, SPLICE_METHOD_WEIGHTS


def draw_splice_method(rng: np.random.Generator) -> SpliceMethod:
    """Draw a splice method randomly according to SPLICE_METHOD_WEIGHTS.

    Args:
        rng: NumPy random generator (caller is responsible for seeding).

    Returns:
        A SpliceMethod variant sampled from the weight distribution.
    """
    methods = list(SPLICE_METHOD_WEIGHTS.keys())
    weights = np.array([SPLICE_METHOD_WEIGHTS[m] for m in methods], dtype=np.float64)
    weights /= weights.sum()
    idx = rng.choice(len(methods), p=weights)
    return methods[idx]


def apply_crossfade(
    segment_before: np.ndarray,
    segment_after: np.ndarray,
    crossfade_samples: int,
    method: SpliceMethod = SpliceMethod.OLA_HANNING,
) -> np.ndarray:
    """Blend two adjacent audio segments at their boundary using the chosen method.

    For all methods except CUT_PASTE, the last crossfade_samples of segment_before
    and the first crossfade_samples of segment_after are blended according to the
    fade curve. The output length is len(segment_before) + len(segment_after)
    minus crossfade_samples (the overlap region is merged, not doubled).

    For CUT_PASTE (or crossfade_samples <= 0), the segments are concatenated
    with no blending.

    Args:
        segment_before: Audio samples preceding the splice point (1-D float32).
        segment_after: Audio samples following the splice point (1-D float32).
        crossfade_samples: Number of samples in the overlap/blend region.
        method: Fade-curve variant to apply.

    Returns:
        Blended audio array.

    Raises:
        ValueError: If crossfade_samples exceeds either segment length.
    """
    if method is SpliceMethod.CUT_PASTE or crossfade_samples <= 0:
        return np.concatenate([segment_before, segment_after])

    if crossfade_samples > len(segment_before):
        raise ValueError(
            f"crossfade_samples ({crossfade_samples}) exceeds "
            f"segment_before length ({len(segment_before)})"
        )
    if crossfade_samples > len(segment_after):
        raise ValueError(
            f"crossfade_samples ({crossfade_samples}) exceeds "
            f"segment_after length ({len(segment_after)})"
        )

    t = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
    fade_in, fade_out = _compute_fade_curves(t, method)

    overlap = (
        segment_before[-crossfade_samples:] * fade_out
        + segment_after[:crossfade_samples] * fade_in
    )

    return np.concatenate([
        segment_before[:-crossfade_samples],
        overlap,
        segment_after[crossfade_samples:],
    ])


def _compute_fade_curves(
    t: np.ndarray,
    method: SpliceMethod,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fade-in and fade-out envelopes for the given method.

    Args:
        t: Linearly spaced values in [0, 1] for the crossfade region.
        method: Fade-curve variant.

    Returns:
        Tuple of (fade_in, fade_out) arrays, each of shape (len(t),).
        At t=0: fade_in=0, fade_out=1. At t=1: fade_in=1, fade_out=0.
    """
    if method is SpliceMethod.LINEAR:
        fade_in = t
        fade_out = 1.0 - t

    elif method is SpliceMethod.OLA_HANNING:
        fade_in = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)
        fade_out = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)

    elif method is SpliceMethod.COSINE:
        fade_in = np.sin(0.5 * np.pi * t).astype(np.float32)
        fade_out = np.cos(0.5 * np.pi * t).astype(np.float32)

    elif method is SpliceMethod.HALF_SINE:
        fade_in = np.sqrt(t).astype(np.float32)
        fade_out = np.sqrt(1.0 - t).astype(np.float32)

    elif method is SpliceMethod.LOGARITHMIC:
        fade_in = (np.log1p(9.0 * t) / np.log(10.0)).astype(np.float32)
        fade_out = (np.log1p(9.0 * (1.0 - t)) / np.log(10.0)).astype(np.float32)

    elif method is SpliceMethod.PARABOLA:
        fade_in = (1.0 - (1.0 - t) ** 2).astype(np.float32)
        fade_out = ((1.0 - t) ** 2).astype(np.float32)

    else:
        raise ValueError(f"Unhandled SpliceMethod: {method!r}")

    return fade_in, fade_out


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


def find_nearest_valley(
    audio: np.ndarray,
    position: int,
    sample_rate: int,
    search_ms: float = 50.0,
    window_ms: float = 10.0,
) -> int:
    """Find the nearest low-energy valley to a sample position.

    Slides a short RMS window around ``position`` (within +/- search_ms)
    and returns the centre of the lowest-RMS frame. Used by the splice
    engine to snap word slot boundaries onto silent regions before
    crossfade. When the boundary falls inside speech, the bonafide
    component of the crossfade bleeds through and the listener hears
    the original word AND the cloned word simultaneously ("ghost").
    Snapping to a valley makes the bonafide component near-silent in
    the fade region, so only the cloned signal is audible.

    Independent from ``find_nearest_zero_crossing``: that function
    targets cut-paste click avoidance over a millisecond window; this
    function targets ghost avoidance over a tens-of-milliseconds window.

    Args:
        audio: Full audio waveform (1-D float array).
        position: Target sample position (e.g. Parakeet word boundary
            in samples).
        sample_rate: Audio sample rate in Hz.
        search_ms: Half-width of the search window in milliseconds.
            ``0.0`` disables snapping (returns ``position`` unchanged).
        window_ms: RMS analysis window in milliseconds.

    Returns:
        Sample index of the lowest-RMS frame within the search window,
        or the original ``position`` if the search range is empty or
        ``search_ms == 0``.
    """
    if search_ms <= 0.0 or position < 0 or position >= len(audio):
        return position

    search_samples = max(1, int(search_ms * sample_rate / 1000))
    window_samples = max(1, int(window_ms * sample_rate / 1000))
    hop = max(1, window_samples // 4)

    start = max(0, position - search_samples)
    end = min(len(audio) - window_samples, position + search_samples)

    if start >= end:
        return position

    best_rms = float("inf")
    best_pos = position
    for p in range(start, end + 1, hop):
        segment = audio[p : p + window_samples]
        if len(segment) == 0:
            continue
        rms = float(np.sqrt(np.mean(segment.astype(np.float32) ** 2) + 1e-12))
        if rms < best_rms:
            best_rms = rms
            best_pos = p + window_samples // 2

    return int(best_pos)


def normalize_energy(
    cloned_segment: np.ndarray,
    bonafide_region: np.ndarray,
    margin_samples: int = 160,
) -> np.ndarray:
    """Normalize the energy of a cloned segment to match the bonafide region.

    Computes RMS energy of the bonafide region and scales the cloned segment
    to match, preventing loudness discontinuities at splice boundaries.

    Args:
        cloned_segment: The cloned word audio to be normalized.
        bonafide_region: The bonafide audio region being replaced (same word).
        margin_samples: Unused; kept for API compatibility.

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
