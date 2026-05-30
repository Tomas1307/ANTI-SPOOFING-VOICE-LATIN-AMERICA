"""
Verification for the spoof-word loudness matching fix.

Builds synthetic bonafide and cloned audio (no GPU, no model, no real
files), runs the splice engine, and asserts that the spliced spoof
region's voiced RMS matches the bonafide host's voiced loudness anchor.
Also exercises the legacy ablation path, the peak ceiling, and the
all-silence edge case.

Run on ml-server03 (or any host with numpy + loguru) from the repo root:

    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
    python3 -m app.tests.test_loudness_matching
    deactivate
"""
import numpy as np

from app.pipeline.partial_spoof.utils.loudness import (
    apply_peak_ceiling,
    compute_voiced_rms,
    voiced_match_gain,
)
from app.pipeline.partial_spoof.utils.splice_engine import splice_words

SAMPLE_RATE = 16000
# +/- 1.5 dB tolerance expressed as a linear ratio band.
DB_TOLERANCE = 1.5
RATIO_LOW = 10 ** (-DB_TOLERANCE / 20)
RATIO_HIGH = 10 ** (DB_TOLERANCE / 20)


def _tone(amplitude: float, duration_s: float, freq: float = 220.0) -> np.ndarray:
    """Generate a sine burst (voiced proxy) of the given peak amplitude."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    """Generate near-silence (very low amplitude noise floor)."""
    n = int(duration_s * SAMPLE_RATE)
    return (1e-4 * np.ones(n, dtype=np.float32))


def _build_synthetic():
    """Build a 2-word bonafide host and a quiet 1-word clone of 'hola'."""
    bonafide = np.concatenate([
        _silence(0.10),
        _tone(0.20, 0.30),   # word 0: "hola"  -> [0.10, 0.40] s
        _silence(0.10),
        _tone(0.20, 0.30),   # word 1: "mundo" -> [0.50, 0.80] s
        _silence(0.10),
    ]).astype(np.float32)
    bonafide_words = [
        {"word": "hola", "start": 0.10, "end": 0.40},
        {"word": "mundo", "start": 0.50, "end": 0.80},
    ]
    # The cloned word carries an internal silent gap so its plain RMS is
    # below its voiced RMS. This exposes the plain-vs-voiced matching
    # asymmetry: a plain-RMS match overshoots and leaves the spoof region
    # louder than the host, while a voiced match lands on it.
    cloned_word = np.concatenate([
        _tone(0.02, 0.12),
        _silence(0.06),
        _tone(0.02, 0.12),
    ])  # 0.30 s total, 10x quieter than host
    cloned = np.concatenate([
        _silence(0.10),
        cloned_word,
        _silence(0.10),
    ]).astype(np.float32)
    cloned_words = [{"word": "hola", "start": 0.10, "end": 0.40}]
    return bonafide, bonafide_words, cloned, cloned_words


def _spoof_region_rms(result, details):
    """Voiced RMS of the spliced spoof region from its detail entry."""
    d = details[0]
    start = int(d["bonafide_start_s"] * SAMPLE_RATE)
    end = int(d["bonafide_end_s"] * SAMPLE_RATE)
    return compute_voiced_rms(result[start:end], SAMPLE_RATE)


def test_unit_helpers():
    """compute_voiced_rms / voiced_match_gain / apply_peak_ceiling."""
    # Voiced RMS ignores silence: a sine of amplitude A has RMS A/sqrt(2).
    sig = np.concatenate([_silence(0.2), _tone(0.2, 0.2)])
    rms = compute_voiced_rms(sig, SAMPLE_RATE)
    assert abs(rms - 0.2 / np.sqrt(2)) < 0.01, rms

    # voiced_match_gain brings a quiet segment's VOICED RMS to the reference,
    # measuring the segment the same (voiced) way despite its internal silence.
    quiet = np.concatenate([_tone(0.02, 0.15), _silence(0.10), _tone(0.02, 0.15)])
    scaled, scale = voiced_match_gain(quiet, SAMPLE_RATE, rms)
    assert scale > 1.0
    assert abs(compute_voiced_rms(scaled, SAMPLE_RATE) - rms) < 1e-3

    # Degenerate reference -> no-op, unit gain.
    _, unit = voiced_match_gain(quiet, SAMPLE_RATE, 0.0)
    assert unit == 1.0

    # Peak ceiling clamps and preserves ratios.
    loud = _tone(2.0, 0.2)
    clamped = apply_peak_ceiling(loud, 0.99)
    assert float(np.max(np.abs(clamped))) <= 0.99 + 1e-6
    # Already-quiet signal is untouched.
    assert np.array_equal(apply_peak_ceiling(quiet, 0.99), quiet)
    print("[PASS] unit helpers")


def test_matching_enabled():
    """Spoof region voiced RMS matches the host anchor within +/-1.5 dB."""
    bonafide, bw, cloned, cw = _build_synthetic()
    anchor = compute_voiced_rms(bonafide, SAMPLE_RATE)

    for cf in (0.0, 30.0):  # cf=0 -> CUT_PASTE branch; cf=30 -> crossfade branch
        result, details = splice_words(
            bonafide_audio=bonafide,
            cloned_audio=cloned,
            bonafide_words=bw,
            cloned_words=cw,
            selected_indices=[0],
            sample_rate=SAMPLE_RATE,
            crossfade_min_ms=cf,
            crossfade_max_ms=cf,
            splice_seed=7,
            loudness_match_enabled=True,
        )
        assert len(details) == 1
        spoof_rms = _spoof_region_rms(result, details)
        ratio = spoof_rms / anchor
        assert RATIO_LOW <= ratio <= RATIO_HIGH, (
            f"cf={cf}: spoof/host ratio {ratio:.3f} outside +/-1.5 dB band"
        )
        assert details[0]["loudness_applied_scale"] > 1.0
        print(f"[PASS] matching enabled (cf={cf}): ratio={ratio:.3f}")


def test_matching_disabled_is_quieter():
    """Legacy ablation: disabled matching leaves the spoof region quiet."""
    bonafide, bw, cloned, cw = _build_synthetic()
    anchor = compute_voiced_rms(bonafide, SAMPLE_RATE)

    result, details = splice_words(
        bonafide_audio=bonafide,
        cloned_audio=cloned,
        bonafide_words=bw,
        cloned_words=cw,
        selected_indices=[0],
        sample_rate=SAMPLE_RATE,
        crossfade_min_ms=0.0,
        crossfade_max_ms=0.0,
        splice_seed=7,
        loudness_match_enabled=False,
    )
    spoof_rms = _spoof_region_rms(result, details)
    assert spoof_rms < anchor * RATIO_LOW, (
        f"disabled spoof RMS {spoof_rms:.4f} should be well below anchor {anchor:.4f}"
    )
    assert details[0]["loudness_applied_scale"] == 1.0
    print(f"[PASS] matching disabled: spoof RMS {spoof_rms:.4f} << anchor {anchor:.4f}")


def test_all_silence_host():
    """All-silence bonafide host: no exception, unit gain, no scaling."""
    bonafide = _silence(1.0)
    bonafide_words = [{"word": "hola", "start": 0.10, "end": 0.40}]
    cloned = np.concatenate([_silence(0.10), _tone(0.02, 0.30), _silence(0.10)])
    cloned_words = [{"word": "hola", "start": 0.10, "end": 0.40}]

    result, details = splice_words(
        bonafide_audio=bonafide.astype(np.float32),
        cloned_audio=cloned.astype(np.float32),
        bonafide_words=bonafide_words,
        cloned_words=cloned_words,
        selected_indices=[0],
        sample_rate=SAMPLE_RATE,
        crossfade_min_ms=0.0,
        crossfade_max_ms=0.0,
        splice_seed=7,
        loudness_match_enabled=True,
    )
    assert len(details) == 1
    assert details[0]["loudness_target_rms"] == 0.0
    assert details[0]["loudness_applied_scale"] == 1.0
    print("[PASS] all-silence host: anchor 0.0, scale 1.0, no crash")


def test_unknown_reference_mode_raises():
    """An unsupported reference mode fails fast."""
    bonafide, bw, cloned, cw = _build_synthetic()
    try:
        splice_words(
            bonafide_audio=bonafide,
            cloned_audio=cloned,
            bonafide_words=bw,
            cloned_words=cw,
            selected_indices=[0],
            sample_rate=SAMPLE_RATE,
            crossfade_min_ms=0.0,
            crossfade_max_ms=0.0,
            loudness_match_enabled=True,
            loudness_reference_mode="bogus",
        )
    except ValueError:
        print("[PASS] unknown reference mode raises ValueError")
        return
    raise AssertionError("expected ValueError for unknown reference mode")


if __name__ == "__main__":
    test_unit_helpers()
    test_matching_enabled()
    test_matching_disabled_is_quieter()
    test_all_silence_host()
    test_unknown_reference_mode_raises()
    print("\nAll loudness-matching checks passed.")
