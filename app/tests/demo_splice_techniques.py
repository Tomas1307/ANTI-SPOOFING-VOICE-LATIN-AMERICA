"""
Demo: splice the SAME real word using each of the 7 techniques.

Duration-preserving approach: the cloned word is fitted to EXACTLY the
same time slot as the bonafide word. Total audio duration never changes.
Gaps between words are perfectly preserved.

Output in data/qwen_partial_spoof/technique_demos/

Run locally:
    cd d:/Andes/Maestria/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    python app/tests/demo_splice_techniques.py
"""
import json
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for mod in ["torch", "jiwer", "nemo", "nemo.collections", "nemo.collections.asr",
            "torchaudio", "speechbrain", "torchmetrics"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import librosa
import numpy as np
import soundfile as sf

crossfade_mod = importlib.import_module("app.pipeline.partial_spoof.utils.crossfade")
find_nearest_zero_crossing = crossfade_mod.find_nearest_zero_crossing
normalize_energy = crossfade_mod.normalize_energy
_compute_fade_curves = crossfade_mod._compute_fade_curves

splice_method_mod = importlib.import_module("app.pipeline.partial_spoof.utils.splice_method")
SpliceMethod = splice_method_mod.SpliceMethod

SAMPLE_RATE = 16000
CROSSFADE_MS = 15.0
DATA_DIR = PROJECT_ROOT / "data" / "qwen_partial_spoof"


def _normalize_word(word: str) -> str:
    """Normalize word for text matching."""
    import unicodedata
    stripped = word.lower().strip(".,;:!?()[]{}\"'")
    nfkd = unicodedata.normalize("NFKD", stripped)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def time_stretch(segment: np.ndarray, target_length: int) -> np.ndarray:
    """Stretch or compress audio segment to target length via interpolation.

    Args:
        segment: Audio segment (1-D float32).
        target_length: Desired number of samples.

    Returns:
        Resampled segment of exactly target_length samples.
    """
    if len(segment) == 0 or target_length <= 0:
        return segment
    if len(segment) == target_length:
        return segment.copy()
    indices = np.linspace(0, len(segment) - 1, target_length)
    return np.interp(indices, np.arange(len(segment)), segment).astype(np.float32)


def splice_duration_preserving(
    bonafide_audio: np.ndarray,
    cloned_audio: np.ndarray,
    bonafide_words: list,
    cloned_words: list,
    word_index: int,
    method: SpliceMethod,
    crossfade_samples: int,
) -> tuple:
    """Splice a cloned word into bonafide, preserving exact bonafide duration.

    The cloned word is time-stretched to fit the bonafide word's time slot.
    Total audio length never changes. Gaps between words are preserved.

    The crossfade happens INSIDE the slot boundaries: a few ms at the start
    blend from bonafide into cloned, and a few ms at the end blend back.

    Args:
        bonafide_audio: Full bonafide waveform.
        cloned_audio: Full cloned waveform.
        bonafide_words: Bonafide word timestamps.
        cloned_words: Cloned word timestamps.
        word_index: Index of the word to replace in bonafide.
        method: Splice technique to use.
        crossfade_samples: Crossfade length for boundary blending.

    Returns:
        Tuple of (spliced_audio, info_dict).
    """
    bw = bonafide_words[word_index]
    target = _normalize_word(bw["word"])

    cw = None
    for c in cloned_words:
        if _normalize_word(c["word"]) == target:
            cw = c
            break

    if cw is None:
        return bonafide_audio.copy(), {"error": f"word '{bw['word']}' not found in clone"}

    result = bonafide_audio.copy()

    b_start = int(bw["start"] * SAMPLE_RATE)
    b_end = int(bw["end"] * SAMPLE_RATE)
    slot_len = b_end - b_start

    c_start = int(cw["start"] * SAMPLE_RATE)
    c_end = int(cw["end"] * SAMPLE_RATE)
    cloned_word = cloned_audio[c_start:c_end].copy()

    bonafide_dur_ms = slot_len * 1000 / SAMPLE_RATE
    cloned_dur_ms = len(cloned_word) * 1000 / SAMPLE_RATE
    stretch_ratio = len(cloned_word) / slot_len if slot_len > 0 else 1.0

    fitted = time_stretch(cloned_word, slot_len)

    bonafide_slot = result[b_start:b_end].copy()
    fitted = normalize_energy(fitted, bonafide_slot)

    if method is SpliceMethod.CUT_PASTE:
        result[b_start:b_end] = fitted
    else:
        cf = min(crossfade_samples, slot_len // 4)
        if cf > 0:
            t = np.linspace(0.0, 1.0, cf, dtype=np.float32)
            fade_in, fade_out = _compute_fade_curves(t, method)

            fitted[:cf] = bonafide_slot[:cf] * fade_out + fitted[:cf] * fade_in

            t_end = np.linspace(0.0, 1.0, cf, dtype=np.float32)
            fade_in_end, fade_out_end = _compute_fade_curves(t_end, method)
            fitted[-cf:] = fitted[-cf:] * fade_out_end + bonafide_slot[-cf:] * fade_in_end

        result[b_start:b_end] = fitted

    info = {
        "word": bw["word"],
        "bonafide_dur_ms": round(bonafide_dur_ms, 1),
        "cloned_dur_ms": round(cloned_dur_ms, 1),
        "stretch_ratio": round(stretch_ratio, 3),
        "crossfade_ms": round(cf * 1000 / SAMPLE_RATE, 1) if method is not SpliceMethod.CUT_PASTE else 0,
        "method": method.value,
    }
    return result, info


def resolve_bonafide_path(raw_path: str) -> Path:
    """Resolve bonafide path: try v2 first, fall back to v1."""
    p = PROJECT_ROOT / raw_path
    if p.exists():
        return p
    return PROJECT_ROOT / raw_path.replace(
        "bonafide_dataset_by_speaker_v2", "bonafide_dataset_by_speaker"
    )


def main() -> None:
    """Generate technique comparison demos from real audio."""
    align_path = DATA_DIR / "alignment_metadata.json"
    select_path = DATA_DIR / "word_selection_metadata.json"

    if not align_path.exists() or not select_path.exists():
        print(f"ERROR: metadata not found in {DATA_DIR}")
        return

    with open(align_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)
    with open(select_path, "r", encoding="utf-8") as f:
        selections = json.load(f)

    sample_key = None
    bonafide_path = None
    cloned_path = None
    entry = None

    for key, ent in alignment.items():
        bf_p = resolve_bonafide_path(ent["bonafide_audio_path"])
        cl_p = PROJECT_ROOT / ent["cloned_audio_path"]
        if bf_p.exists() and cl_p.exists() and key in selections:
            sample_key = key
            bonafide_path = bf_p
            cloned_path = cl_p
            entry = ent
            break

    if sample_key is None:
        print("ERROR: No sample found with both audio files on disk.")
        return

    print(f"Sample:     {sample_key}")
    print(f"Transcript: {entry['transcript']}")

    bonafide_audio, _ = librosa.load(str(bonafide_path), sr=SAMPLE_RATE, mono=True)
    cloned_audio, _ = librosa.load(str(cloned_path), sr=SAMPLE_RATE, mono=True)

    sel = selections[sample_key]["selections"][0]
    word_idx = sel["selected_indices"][0]
    bw = entry["bonafide_words"][word_idx]
    print(f"\nWord: '{bw['word']}' (index {word_idx})")

    prev_end = entry["bonafide_words"][word_idx - 1]["end"] if word_idx > 0 else 0
    next_start = entry["bonafide_words"][word_idx + 1]["start"] if word_idx < len(entry["bonafide_words"]) - 1 else len(bonafide_audio) / SAMPLE_RATE
    gap_before = (bw["start"] - prev_end) * 1000
    gap_after = (next_start - bw["end"]) * 1000

    print(f"  Bonafide: {bw['start']:.2f}s - {bw['end']:.2f}s "
          f"({(bw['end']-bw['start'])*1000:.0f}ms)")

    cw_text = _normalize_word(bw["word"])
    for c in entry["cloned_words"]:
        if _normalize_word(c["word"]) == cw_text:
            print(f"  Cloned:   {c['start']:.2f}s - {c['end']:.2f}s "
                  f"({(c['end']-c['start'])*1000:.0f}ms)")
            break

    print(f"  Gap before: {gap_before:.0f}ms | Gap after: {gap_after:.0f}ms")
    print(f"  Duration-preserving: cloned word stretched to fit bonafide slot")

    out_dir = DATA_DIR / "technique_demos"
    out_dir.mkdir(exist_ok=True)

    crossfade_samples = int(CROSSFADE_MS * SAMPLE_RATE / 1000)

    ref_bf = out_dir / "00_bonafide_original.wav"
    sf.write(str(ref_bf), bonafide_audio, SAMPLE_RATE)
    print(f"\n  [REF] {ref_bf.name} -- bonafide sin editar "
          f"({len(bonafide_audio)/SAMPLE_RATE:.2f}s)")

    ref_cl = out_dir / "00_cloned_full.wav"
    sf.write(str(ref_cl), cloned_audio, SAMPLE_RATE)
    print(f"  [REF] {ref_cl.name} -- clon completo "
          f"({len(cloned_audio)/SAMPLE_RATE:.2f}s)")

    techniques = [
        ("01", SpliceMethod.CUT_PASTE, "Corte directo"),
        ("02", SpliceMethod.OLA_HANNING, "OLA Hanning"),
        ("03", SpliceMethod.LINEAR, "Linear"),
        ("04", SpliceMethod.COSINE, "Cosine equal-power"),
        ("05", SpliceMethod.HALF_SINE, "Square-root"),
        ("06", SpliceMethod.LOGARITHMIC, "Logarithmic"),
        ("07", SpliceMethod.PARABOLA, "Inverted parabola"),
    ]

    for num, method, description in techniques:
        result, info = splice_duration_preserving(
            bonafide_audio=bonafide_audio,
            cloned_audio=cloned_audio,
            bonafide_words=entry["bonafide_words"],
            cloned_words=entry["cloned_words"],
            word_index=word_idx,
            method=method,
            crossfade_samples=crossfade_samples,
        )
        fname = f"{num}_{method.value}.wav"
        out_path = out_dir / fname
        sf.write(str(out_path), result, SAMPLE_RATE)
        dur_info = (f"bf={info['bonafide_dur_ms']}ms, "
                    f"cl={info['cloned_dur_ms']}ms, "
                    f"stretch={info['stretch_ratio']}x")
        print(f"  [{num}] {fname} -- {description} | {dur_info} | "
              f"len={len(result)/SAMPLE_RATE:.2f}s")

    print(f"\nListo. Archivos en: {out_dir}")
    print(f"TODAS las salidas tienen EXACTAMENTE la misma duracion que el bonafide.")
    print(f"La palabra clonada fue estirada/comprimida para caber en el slot.")


if __name__ == "__main__":
    main()
