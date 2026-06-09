"""
Loudness-matching utilities for the partial-spoof splice engine.

Provides silence-gated voiced RMS estimation, RMS-ratio gain scaling, and a
global peak ceiling. These helpers replace the legacy ``normalize_energy``
function whose bonafide reference was measured over the valley-snapped slot
(word plus flanking silence), deflating the target and scaling cloned words
systematically too quiet.

The replacement strategy measures loudness on voiced frames only (frames whose
RMS clears an adaptive gate), so the cloned word and the bonafide reference are
compared like-with-like. The reference is computed once per utterance on the
original bonafide host, giving a stable per-file anchor that keeps every spoof
word in a multi-word (W2/W3) sample mutually consistent.

All loudness matching is linear-RMS ratio based, which is mathematically
equivalent to matching RMS-dB while avoiding the unreliable ~400 ms gating of
LUFS measurement on sub-word segments.
"""
import numpy as np


def compute_voiced_rms(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    gate_fraction: float = 0.15,
    silence_rms: float = 0.015,
) -> float:
    """Compute the RMS of an audio signal over its voiced frames only.

    The signal is split into fixed-size frames; per-frame RMS is computed and
    frames below an adaptive energy gate are discarded as silence before the
    overall RMS is taken. The gate is the larger of an absolute silence floor
    and a fraction of the 95th-percentile frame RMS, so the estimate adapts to
    the signal's own dynamic range while never dipping into recording noise.

    Args:
        audio: Audio waveform as a 1-D float array.
        sample_rate: Audio sample rate in Hz.
        frame_ms: Frame length in milliseconds used for the RMS envelope.
        gate_fraction: Fraction of the 95th-percentile frame RMS used as the
            relative component of the voiced gate.
        silence_rms: Absolute RMS floor below which frames are always treated
            as silence.

    Returns:
        The RMS of the voiced frames, or ``0.0`` when no frame clears the gate
        (e.g. an all-silence input). Returning zero lets callers detect a
        degenerate reference and skip scaling rather than divide by it.
    """
    if len(audio) == 0:
        return 0.0

    frame_samples = max(1, int(frame_ms * sample_rate / 1000))
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return 0.0

    frames = audio[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    frame_rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))

    gate = max(silence_rms, gate_fraction * float(np.percentile(frame_rms, 95)))
    voiced = frame_rms[frame_rms >= gate]
    if voiced.size == 0:
        return 0.0

    return float(np.sqrt(np.mean(voiced ** 2)))


def voiced_match_gain(
    segment: np.ndarray,
    sample_rate: int,
    reference_rms: float,
    frame_ms: float = 20.0,
    gate_fraction: float = 0.15,
    silence_rms: float = 0.015,
    rms_floor: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Scale a segment so its VOICED RMS matches a reference RMS.

    The segment's loudness is measured with the same silence-gated voiced RMS
    used to compute the bonafide reference, so voiced energy is matched
    apples-to-apples. Matching a segment's plain RMS to a voiced reference
    instead systematically overshoots: intra-word silence drags the plain RMS
    down, so the gain needed to reach the voiced reference pushes the segment's
    actual voiced content above it (heard as the spoof word sitting louder than
    the surrounding bonafide).

    Args:
        segment: Audio segment to scale as a 1-D float array.
        sample_rate: Audio sample rate in Hz.
        reference_rms: Target voiced RMS the segment should match.
        frame_ms: Frame length (ms) for the voiced RMS measurement.
        gate_fraction: Relative voiced-gate fraction (see compute_voiced_rms).
        silence_rms: Absolute silence floor for the voiced gate.
        rms_floor: Magnitude below which the reference or the segment's voiced
            RMS is treated as degenerate; the segment is then returned
            unchanged with a unit gain.

    Returns:
        Tuple of (scaled_segment, scale). On degenerate input the original
        segment and a scale of ``1.0`` are returned.
    """
    if len(segment) == 0 or reference_rms < rms_floor:
        return segment, 1.0

    segment_voiced_rms = compute_voiced_rms(
        segment, sample_rate, frame_ms, gate_fraction, silence_rms
    )
    if segment_voiced_rms < rms_floor:
        return segment, 1.0

    scale = reference_rms / segment_voiced_rms
    return (segment * scale).astype(np.float32), float(scale)


def apply_peak_ceiling(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Down-scale an entire signal so its peak amplitude stays under a ceiling.

    A single uniform gain is applied to the whole signal so the relative levels
    of spoof and bonafide regions are preserved. A per-sample limiter or hard
    clip would distort only the loud region and re-break the loudness match.
    This guard is required because the downstream FLAC export uses integer PCM,
    which hard-clips any sample outside [-1, 1].

    Args:
        audio: Audio waveform as a 1-D float array.
        ceiling: Maximum permitted absolute sample amplitude.

    Returns:
        The original signal when its peak is already within the ceiling,
        otherwise the signal scaled down so its peak equals the ceiling.
    """
    if len(audio) == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling or peak < 1e-12:
        return audio

    return (audio * (ceiling / peak)).astype(np.float32)
