"""
Diagnostic: empirically measure pre-speech gap and RMS energy for OmniVoice
generated samples to inform a non-verbal prefix trim heuristic.

For each WAV file in the configured generated/ directory:
    1. Run Parakeet TDT to get word-level timestamps and full transcription.
    2. Compute audio duration.
    3. Extract the first word's start time (T_first).
    4. Compute RMS dBFS over [0, T_first] (pre-speech window).
    5. Compute RMS dBFS over [T_first, T_first + 0.5s] (speech reference).
    6. Print a per-sample row, persist all rows to JSON, print a final summary.

Output JSON is written to settings.OUTPUT_DIR / 'prefix_diagnostic.json' so the
data survives terminal scrollback truncation.

Usage on ml-server03:
    source envs/omnivoice_env/bin/activate
    export CUDA_VISIBLE_DEVICES=1
    # Run on all generated samples:
    python -m app.scripts.diagnose_omnivoice_prefix
    # Or filter by substring (e.g. just TEXT_00001):
    python -m app.scripts.diagnose_omnivoice_prefix TEXT_00001
    deactivate

The output table tells us:
    - How long is the pre-speech gap when an artifact is present vs absent
    - Whether the pre-speech RMS is meaningfully louder than silence
    - Where to set min_gap_seconds and silence_floor_db for the production
      detector
"""
import json
import sys
from pathlib import Path
from typing import List

import librosa
import numpy as np
from loguru import logger

from app.pipeline.omnivoice_attack.settings import settings
from app.utils.parakeet_transcriber import ParakeetTranscriber
from app.utils.word_timestamp import WordTimestamp


def _rms_dbfs(samples: np.ndarray) -> float:
    """Compute RMS energy of a waveform in dBFS (relative to peak digital).

    Args:
        samples: 1D numpy array of audio samples in [-1.0, 1.0] range.

    Returns:
        RMS energy in dBFS. Returns -120.0 for empty or pure-silence inputs.
    """
    if len(samples) == 0:
        return -120.0
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms < 1e-9:
        return -120.0
    return 20.0 * np.log10(rms)


def _format_timestamps(timestamps: List[WordTimestamp], max_words: int = 8) -> str:
    """Build a compact one-line representation of the first N word timestamps.

    Args:
        timestamps: Word-level timestamps from Parakeet.
        max_words: Cap on the number of words to render.

    Returns:
        Space-separated "word(start-end)" tokens.
    """
    parts = [f"{wt.word}({wt.start:.3f}-{wt.end:.3f})" for wt in timestamps[:max_words]]
    if len(timestamps) > max_words:
        parts.append("...")
    return " ".join(parts)


def main() -> int:
    """Diagnose OmniVoice prefix artifacts via Parakeet timestamp analysis.

    Optionally accepts a filter substring as the first CLI argument; only
    WAV files whose stem contains the substring are processed.

    Returns:
        Process exit code (0 on success, 1 on missing input directory).
    """
    name_filter = sys.argv[1] if len(sys.argv) > 1 else None

    generated_dir = settings.OUTPUT_DIR / "generated"
    if not generated_dir.exists():
        logger.error(f"Generated directory not found: {generated_dir}")
        return 1

    wav_files = sorted(generated_dir.glob("*.wav"))
    if name_filter:
        wav_files = [w for w in wav_files if name_filter in w.stem]
        logger.info(f"Filter '{name_filter}' matched {len(wav_files)} files")

    if not wav_files:
        logger.error(f"No WAV files match in {generated_dir}")
        return 1

    logger.info(f"Processing {len(wav_files)} WAV files from {generated_dir}")

    transcriber = ParakeetTranscriber()
    transcriber.load(model_id=settings.PARAKEET_MODEL_ID, device=settings.DEVICE)

    header = (
        f"{'sample':<48} | {'dur(s)':>6} | "
        f"{'T_first':>7} | {'pre_RMS':>8} | {'spch_RMS':>8} | {'gap?':<5} | first_words"
    )

    print()
    print("=" * 120)
    print(header)
    print("-" * 120)

    rows = []
    for wav_path in wav_files:
        audio, sr = librosa.load(str(wav_path), sr=settings.SAMPLE_RATE)
        duration = len(audio) / sr

        text, timestamps = transcriber.transcribe_with_timestamps(wav_path)

        if not timestamps:
            line = (
                f"{wav_path.stem:<48} | {duration:>6.2f} | "
                f"{'N/A':>7} | {'N/A':>8} | {'N/A':>8} | {'no-ts':<5} | <no timestamps>"
            )
            print(line)
            rows.append({
                "stem": wav_path.stem,
                "duration": duration,
                "t_first": None,
                "pre_rms_db": None,
                "speech_rms_db": None,
                "transcription": text,
                "first_words_repr": "<no timestamps>",
            })
            continue

        t_first = timestamps[0].start

        pre_samples = audio[: int(t_first * sr)]
        pre_rms_db = _rms_dbfs(pre_samples)

        speech_end_idx = min(int((t_first + 0.5) * sr), len(audio))
        speech_samples = audio[int(t_first * sr) : speech_end_idx]
        speech_rms_db = _rms_dbfs(speech_samples)

        gap_flag = "YES" if t_first >= 0.020 else "no"

        words_repr = _format_timestamps(timestamps)
        line = (
            f"{wav_path.stem:<48} | {duration:>6.2f} | "
            f"{t_first:>7.3f} | {pre_rms_db:>8.1f} | {speech_rms_db:>8.1f} | "
            f"{gap_flag:<5} | {words_repr}"
        )
        print(line)

        rows.append({
            "stem": wav_path.stem,
            "duration": float(duration),
            "t_first": float(t_first),
            "pre_rms_db": float(pre_rms_db),
            "speech_rms_db": float(speech_rms_db),
            "gap_flag": gap_flag,
            "transcription": text,
            "first_words_repr": words_repr,
            "all_timestamps": [
                {"word": wt.word, "start": float(wt.start), "end": float(wt.end)}
                for wt in timestamps
            ],
        })

    print("-" * 120)
    print()

    output_path = settings.OUTPUT_DIR / "prefix_diagnostic.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info(f"Diagnostic results persisted to: {output_path}")

    valid_rows = [r for r in rows if r.get("t_first") is not None]
    if valid_rows:
        gaps = [r["t_first"] for r in valid_rows]
        pre_rms = [r["pre_rms_db"] for r in valid_rows]
        speech_rms = [r["speech_rms_db"] for r in valid_rows]

        print("SUMMARY")
        print(f"  T_first  : min={min(gaps):.3f}s  median={float(np.median(gaps)):.3f}s  max={max(gaps):.3f}s")
        print(f"  pre_RMS  : min={min(pre_rms):.1f}dB  median={float(np.median(pre_rms)):.1f}dB  max={max(pre_rms):.1f}dB")
        print(f"  speech   : min={min(speech_rms):.1f}dB  median={float(np.median(speech_rms)):.1f}dB  max={max(speech_rms):.1f}dB")
        print()
        print("INTERPRETATION GUIDE")
        print("  - Samples with a real prefix artifact should show a clear T_first gap")
        print("    AND pre_RMS noticeably louder than the silence floor (~ -55 dB or below).")
        print("  - Clean samples should show small T_first or pre_RMS near the silence floor.")
        print("  - Set min_gap_seconds slightly below the smallest 'artifact' T_first observed.")
        print("  - Set silence_floor_db slightly above the largest 'clean' pre_RMS observed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
