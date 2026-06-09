"""
Objective verification of spoof-word loudness matching in spliced audio.

For every spliced sample in one or more partial-spoof cells, this measures the
voiced RMS of the spoof regions against the voiced RMS of the surrounding
bonafide and reports the loudness gap in dB. A working loudness match keeps the
gap near 0 dB (within +/- a couple dB); a large systematic gap means the spoof
words still stand out. It also reports the peak amplitude per file so clipping
introduced by upward scaling is caught.

Spoof region boundaries are read from splice_metadata.json
(spoofed_words[*].bonafide_start_s / bonafide_end_s, which are positions in the
SPLICED audio). Everything outside those regions is treated as bonafide.

Usage on ml-server03 (any env with numpy + librosa):
    python3 scripts/check_loudness_match.py
    python3 scripts/check_loudness_match.py data/partial_spoof_output/qwen/not_jittered
"""
import glob
import json
import sys
from pathlib import Path

import librosa
import numpy as np

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.utils.loudness import compute_voiced_rms

TOLERANCE_DB = 1.5


def _spoof_mask(num_samples: int, spoofed_words: list, sample_rate: int) -> np.ndarray:
    """Build a boolean mask marking spoof-region samples in the spliced audio."""
    mask = np.zeros(num_samples, dtype=bool)
    for word in spoofed_words:
        start = int(word["bonafide_start_s"] * sample_rate)
        end = int(word["bonafide_end_s"] * sample_rate)
        start = max(0, min(start, num_samples))
        end = max(0, min(end, num_samples))
        mask[start:end] = True
    return mask


def _evaluate_metadata(metadata_path: Path) -> list:
    """Return per-sample (delta_db, peak) tuples for one cell's metadata."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    results = []
    for entry in metadata.values():
        wav_path = Path(entry["spliced_audio_path"])
        if not wav_path.exists():
            continue
        audio, sr = librosa.load(str(wav_path), sr=settings.SAMPLE_RATE, mono=True)
        if len(audio) == 0:
            continue

        mask = _spoof_mask(len(audio), entry["spoofed_words"], sr)
        spoof = audio[mask]
        bonafide = audio[~mask]
        if len(spoof) == 0 or len(bonafide) == 0:
            continue

        spoof_rms = compute_voiced_rms(spoof, sr)
        bonafide_rms = compute_voiced_rms(bonafide, sr)
        if spoof_rms <= 0.0 or bonafide_rms <= 0.0:
            continue

        delta_db = 20.0 * np.log10(spoof_rms / bonafide_rms)
        peak = float(np.max(np.abs(audio)))
        results.append((float(delta_db), peak))
    return results


def _report(label: str, results: list) -> None:
    """Print aggregate loudness-match statistics for a result set."""
    if not results:
        print(f"{label}: no measurable samples")
        return

    deltas = np.array([r[0] for r in results])
    peaks = np.array([r[1] for r in results])
    within = float(np.mean(np.abs(deltas) <= TOLERANCE_DB) * 100.0)

    print(f"{label}: {len(results)} samples")
    print(f"  spoof-vs-bonafide dB delta: mean {deltas.mean():+.2f}  "
          f"median {np.median(deltas):+.2f}  "
          f"abs-p90 {np.percentile(np.abs(deltas), 90):.2f}")
    print(f"  within +/-{TOLERANCE_DB} dB        : {within:.1f}%")
    print(f"  peak amplitude            : max {peaks.max():.3f}  "
          f"(clipping if > 1.0)")


def main() -> None:
    """Evaluate one cell directory or every cell under the output root."""
    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1]) / "splice_metadata.json"]
    else:
        targets = [
            Path(p) for p in sorted(
                glob.glob("data/partial_spoof_output/*/*/splice_metadata.json")
            )
        ]

    overall = []
    for metadata_path in targets:
        if not metadata_path.exists():
            print(f"{metadata_path}: missing")
            continue
        cell = f"{metadata_path.parent.parent.name}/{metadata_path.parent.name}"
        results = _evaluate_metadata(metadata_path)
        _report(cell, results)
        overall.extend(results)

    print("-" * 60)
    _report("OVERALL", overall)


if __name__ == "__main__":
    main()
