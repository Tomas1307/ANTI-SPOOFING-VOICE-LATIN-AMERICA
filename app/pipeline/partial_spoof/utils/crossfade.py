"""
Crossfade utility for splice boundary smoothing.

Implements seven splice boundary methods (direct cut-paste plus six distinct
fade-curve variants) used to blend bonafide and cloned word segments. Provides
per-splice random method selection weighted by SPLICE_METHOD_WEIGHTS, along with
the nearest-valley snap helper used to relocate splice seams onto silence.
Loudness matching lives in utils/loudness.py.

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


def find_nearest_valley(
    audio: np.ndarray,
    position: int,
    sample_rate: int,
    search_ms: float = 50.0,
    window_ms: float = 10.0,
    direction: str = "both",
) -> int:
    """Find the lowest-RMS frame near a sample position.

    Slides a short RMS window around ``position`` and returns the
    centre of the lowest-RMS frame within the configured search range.
    Used by the splice engine to snap word slot boundaries onto silent
    regions before crossfade. When the boundary falls inside speech,
    the bonafide component of the crossfade bleeds through and the
    listener hears the original word AND the cloned word simultaneously
    ("ghost"). Snapping to a valley makes the bonafide component
    near-silent in the fade region, so only the cloned signal is
    audible.

    The ``direction`` argument controls which side of ``position`` is
    searched. The splice engine uses ``"earlier"`` for slot-start
    boundaries and ``"later"`` for slot-end boundaries, so the slot
    can only expand outward. A symmetric search ("both") can move a
    boundary *inward* and shrink the slot, leaving the bonafide
    onset/offset uncovered and audible -- exactly the failure mode
    that motivated this function.

    Args:
        audio: Full audio waveform (1-D float array).
        position: Target sample position (e.g. Parakeet word boundary
            in samples).
        sample_rate: Audio sample rate in Hz.
        search_ms: Half-width (one-sided width when ``direction`` is
            ``"earlier"`` / ``"later"``) of the search window in
            milliseconds. ``0.0`` disables snapping.
        window_ms: RMS analysis window in milliseconds.
        direction: ``"earlier"`` searches only positions <= ``position``,
            ``"later"`` only positions >= ``position``, ``"both"``
            searches symmetrically. Default ``"both"`` matches the
            legacy interface but the splice engine should pass the
            asymmetric values to avoid slot shrink.

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

    if direction == "earlier":
        start = max(0, position - search_samples)
        end = min(len(audio) - window_samples, position)
    elif direction == "later":
        start = max(0, position - window_samples // 2)
        end = min(len(audio) - window_samples, position + search_samples)
    else:
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
