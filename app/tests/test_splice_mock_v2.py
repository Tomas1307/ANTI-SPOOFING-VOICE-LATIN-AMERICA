"""
Mock test v2 for the partial spoof splice algorithm.

5 realistic Spanish sentence pairs with simulated timestamps.
Tests the text-matching splice engine across varied scenarios.
"""
import numpy as np
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

for mod in ["torch", "jiwer", "nemo", "nemo.collections", "nemo.collections.asr",
            "librosa", "soundfile", "torchaudio", "speechbrain", "torchmetrics"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

crossfade_mod = importlib.import_module("app.pipeline.partial_spoof.utils.crossfade")
splice_mod = importlib.import_module("app.pipeline.partial_spoof.utils.splice_engine")
splice_words = splice_mod.splice_words

SAMPLE_RATE = 16000


def generate_tone(freq: float, duration_s: float, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave tone."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_silence(duration_s: float) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def build_audio_from_timestamps(timestamps: list, base_freq: float = 200) -> np.ndarray:
    """Build mock audio from timestamp list. Each word gets a unique frequency."""
    if not timestamps:
        return generate_silence(1.0)

    segments = []
    prev_end = 0.0

    for i, ts in enumerate(timestamps):
        gap = ts["start"] - prev_end
        if gap > 0:
            segments.append(generate_silence(gap))

        duration = ts["end"] - ts["start"]
        freq = base_freq + i * 30
        segments.append(generate_tone(freq, duration))
        prev_end = ts["end"]

    segments.append(generate_silence(0.3))
    return np.concatenate(segments)


def run_splice(bonafide_ts, cloned_ts, selected, label=""):
    """Run splice and print results."""
    bonafide_audio = build_audio_from_timestamps(bonafide_ts, base_freq=200)
    cloned_audio = build_audio_from_timestamps(cloned_ts, base_freq=300)

    result, details = splice_words(
        bonafide_audio=bonafide_audio,
        cloned_audio=cloned_audio,
        bonafide_words=bonafide_ts,
        cloned_words=cloned_ts,
        selected_indices=selected,
        sample_rate=SAMPLE_RATE,
        crossfade_min_ms=5.0,
        crossfade_max_ms=5.0,
        max_silence_steal_ms=50.0,
        max_stretch_ratio=1.1,
    )

    target_words = [bonafide_ts[i]["word"] for i in selected if i < len(bonafide_ts)]
    spliced_words = [d["word"] for d in details]

    print(f"  Target:  {target_words}")
    print(f"  Spliced: {spliced_words}")
    print(f"  Count:   {len(details)}/{len(selected)}")
    for d in details:
        print(f"    [{d['word_index']}] '{d['word']}' "
              f"bf=[{d['bonafide_start_s']:.2f}-{d['bonafide_end_s']:.2f}] "
              f"cl=[{d['cloned_start_s']:.2f}-{d['cloned_end_s']:.2f}] "
              f"ratio={d['duration_ratio']:.3f}")

    return details


# ══════════════════════════════════════════════════════════════
# MOCK DATA 1: Perfect clone — all words match exactly
# Sentence: "El presidente anuncio nuevas medidas economicas"
# ══════════════════════════════════════════════════════════════

bonafide_1 = [
    {"word": "El",         "start": 0.30, "end": 0.45},
    {"word": "presidente", "start": 0.48, "end": 1.10},
    {"word": "anuncio",    "start": 1.14, "end": 1.62},
    {"word": "nuevas",     "start": 1.66, "end": 2.05},
    {"word": "medidas",    "start": 2.08, "end": 2.55},
    {"word": "economicas", "start": 2.58, "end": 3.30},
]

cloned_1 = [
    {"word": "El",         "start": 0.28, "end": 0.42},
    {"word": "presidente", "start": 0.46, "end": 1.12},
    {"word": "anuncio",    "start": 1.16, "end": 1.58},
    {"word": "nuevas",     "start": 1.62, "end": 2.00},
    {"word": "medidas",    "start": 2.04, "end": 2.52},
    {"word": "economicas", "start": 2.56, "end": 3.25},
]


# ══════════════════════════════════════════════════════════════
# MOCK DATA 2: Clone has minor ASR differences (punctuation, case)
# Sentence: "Hay varios cines cerca de tu hotel, estas buscando algo?"
# ══════════════════════════════════════════════════════════════

bonafide_2 = [
    {"word": "Hay",      "start": 0.32, "end": 0.58},
    {"word": "varios",   "start": 0.62, "end": 1.02},
    {"word": "cines",    "start": 1.06, "end": 1.42},
    {"word": "cerca",    "start": 1.46, "end": 1.76},
    {"word": "de",       "start": 1.80, "end": 1.92},
    {"word": "tu",       "start": 1.96, "end": 2.10},
    {"word": "hotel,",   "start": 2.14, "end": 2.56},
    {"word": "estas",    "start": 2.66, "end": 3.00},
    {"word": "buscando", "start": 3.04, "end": 3.50},
    {"word": "algo?",    "start": 3.54, "end": 3.84},
]

# Clone ASR: slightly different punctuation/casing
cloned_2 = [
    {"word": "hay",      "start": 0.30, "end": 0.55},
    {"word": "varios",   "start": 0.60, "end": 1.00},
    {"word": "cines",    "start": 1.04, "end": 1.38},
    {"word": "cerca",    "start": 1.42, "end": 1.72},
    {"word": "de",       "start": 1.76, "end": 1.88},
    {"word": "tu",       "start": 1.92, "end": 2.06},
    {"word": "hotel",    "start": 2.10, "end": 2.50},  # no comma
    {"word": "estas",    "start": 2.60, "end": 2.96},
    {"word": "buscando", "start": 3.00, "end": 3.46},
    {"word": "algo",     "start": 3.50, "end": 3.80},  # no question mark
]


# ══════════════════════════════════════════════════════════════
# MOCK DATA 3: Clone missed 2 words (ASR dropped them)
# Sentence: "Los ninos tienen mucha imaginacion para crear historias"
# ══════════════════════════════════════════════════════════════

bonafide_3 = [
    {"word": "Los",          "start": 0.30, "end": 0.52},
    {"word": "ninos",        "start": 0.56, "end": 0.96},
    {"word": "tienen",       "start": 1.00, "end": 1.35},
    {"word": "mucha",        "start": 1.40, "end": 1.72},
    {"word": "imaginacion",  "start": 1.76, "end": 2.50},
    {"word": "para",         "start": 2.54, "end": 2.72},
    {"word": "crear",        "start": 2.76, "end": 3.10},
    {"word": "historias",    "start": 3.14, "end": 3.70},
]

# Clone missed "mucha" and "para" — ASR merged them into adjacent words
cloned_3 = [
    {"word": "Los",          "start": 0.28, "end": 0.50},
    {"word": "ninos",        "start": 0.54, "end": 0.92},
    {"word": "tienen",       "start": 0.96, "end": 1.30},
    {"word": "imaginacion",  "start": 1.40, "end": 2.20},  # "mucha" absorbed
    {"word": "crear",        "start": 2.30, "end": 2.65},  # "para" absorbed
    {"word": "historias",    "start": 2.70, "end": 3.30},
]


# ══════════════════════════════════════════════════════════════
# MOCK DATA 4: Clone has extra hallucinated word at start
# Sentence: "Dame unicamente el dato mas relevante"
# ══════════════════════════════════════════════════════════════

bonafide_4 = [
    {"word": "Dame",        "start": 0.32, "end": 0.68},
    {"word": "unicamente",  "start": 0.72, "end": 1.40},
    {"word": "el",          "start": 1.44, "end": 1.58},
    {"word": "dato",        "start": 1.62, "end": 1.98},
    {"word": "mas",         "start": 2.02, "end": 2.22},
    {"word": "relevante.",  "start": 2.26, "end": 3.00},
]

# Clone hallucinated "Eh" at the start, rest is correct
cloned_4 = [
    {"word": "Eh",          "start": 0.10, "end": 0.25},  # hallucinated
    {"word": "Dame",        "start": 0.30, "end": 0.65},
    {"word": "unicamente",  "start": 0.70, "end": 1.35},
    {"word": "el",          "start": 1.40, "end": 1.55},
    {"word": "dato",        "start": 1.58, "end": 1.95},
    {"word": "mas",         "start": 2.00, "end": 2.18},
    {"word": "relevante",   "start": 2.22, "end": 2.95},
]


# ══════════════════════════════════════════════════════════════
# MOCK DATA 5: Duplicate words in sentence
# Sentence: "De la casa de mi abuela de toda la vida"
# ══════════════════════════════════════════════════════════════

bonafide_5 = [
    {"word": "De",     "start": 0.30, "end": 0.42},
    {"word": "la",     "start": 0.46, "end": 0.58},
    {"word": "casa",   "start": 0.62, "end": 0.95},
    {"word": "de",     "start": 0.98, "end": 1.10},
    {"word": "mi",     "start": 1.14, "end": 1.28},
    {"word": "abuela", "start": 1.32, "end": 1.75},
    {"word": "de",     "start": 1.78, "end": 1.90},
    {"word": "toda",   "start": 1.94, "end": 2.25},
    {"word": "la",     "start": 2.28, "end": 2.40},
    {"word": "vida",   "start": 2.44, "end": 2.85},
]

cloned_5 = [
    {"word": "De",     "start": 0.28, "end": 0.40},
    {"word": "la",     "start": 0.44, "end": 0.56},
    {"word": "casa",   "start": 0.60, "end": 0.92},
    {"word": "de",     "start": 0.96, "end": 1.08},
    {"word": "mi",     "start": 1.12, "end": 1.26},
    {"word": "abuela", "start": 1.30, "end": 1.72},
    {"word": "de",     "start": 1.76, "end": 1.88},
    {"word": "toda",   "start": 1.92, "end": 2.22},
    {"word": "la",     "start": 2.26, "end": 2.38},
    {"word": "vida",   "start": 2.42, "end": 2.82},
]


def main():
    passed = 0
    failed = 0

    # ── TEST 1: Perfect clone, W3 ──
    print("=" * 70)
    print("MOCK 1: Perfect clone — W3 (3 words)")
    print("  'El presidente anuncio nuevas medidas economicas'")
    print("  Replace: 'presidente', 'nuevas', 'economicas' (indices 1, 3, 5)")
    print("-" * 70)
    d1 = run_splice(bonafide_1, cloned_1, [1, 3, 5])
    if len(d1) == 3 and all(d["word"] in ["presidente", "nuevas", "economicas"] for d in d1):
        print("  >> PASSED\n")
        passed += 1
    else:
        print("  >> FAILED\n")
        failed += 1

    # ── TEST 2: Punctuation/case differences, W2 ──
    print("=" * 70)
    print("MOCK 2: Punctuation differences — W2 (2 words)")
    print("  bonafide: 'hotel,' / 'algo?'  vs  cloned: 'hotel' / 'algo'")
    print("  Replace: 'hotel,' and 'algo?' (indices 6, 9)")
    print("-" * 70)
    d2 = run_splice(bonafide_2, cloned_2, [6, 9])
    if len(d2) == 2:
        print("  >> PASSED — punctuation normalization works\n")
        passed += 1
    else:
        print(f"  >> FAILED — expected 2 splices, got {len(d2)}\n")
        failed += 1

    # ── TEST 3: Clone missing words, W2 ──
    print("=" * 70)
    print("MOCK 3: Clone missing 2 words — W2 (request 2, expect partial)")
    print("  bonafide has 'mucha' and 'para', clone does NOT")
    print("  Replace: 'mucha' and 'historias' (indices 3, 7)")
    print("-" * 70)
    d3 = run_splice(bonafide_3, cloned_3, [3, 7])
    if len(d3) == 1 and d3[0]["word"] == "historias":
        print("  >> PASSED — 'mucha' rejected, 'historias' spliced\n")
        passed += 1
    else:
        print(f"  >> FAILED — expected [historias], got {[d['word'] for d in d3]}\n")
        failed += 1

    # ── TEST 4: Clone has hallucinated prefix, W1 ──
    print("=" * 70)
    print("MOCK 4: Clone has hallucinated 'Eh' at start — W1 (1 word)")
    print("  Replace: 'dato' (index 3)")
    print("-" * 70)
    d4 = run_splice(bonafide_4, cloned_4, [3])
    if len(d4) == 1 and d4[0]["word"] == "dato":
        print("  >> PASSED — found 'dato' despite hallucinated prefix\n")
        passed += 1
    else:
        print(f"  >> FAILED — expected [dato], got {[d['word'] for d in d4]}\n")
        failed += 1

    # ── TEST 5: Duplicate words, W3 ──
    print("=" * 70)
    print("MOCK 5: Duplicate words ('de' x3, 'la' x2) — W3 (3 words)")
    print("  Replace: 1st 'de' (idx 0), 'abuela' (idx 5), 2nd 'la' (idx 8)")
    print("-" * 70)
    d5 = run_splice(bonafide_5, cloned_5, [0, 5, 8])
    spliced_words = {d["word"].lower() for d in d5}
    if len(d5) == 3 and "de" in spliced_words and "abuela" in spliced_words and "la" in spliced_words:
        # Verify the correct "de" and "la" were matched (check timestamps)
        de_splice = [d for d in d5 if d["word"].lower() == "de"][0]
        la_splice = [d for d in d5 if d["word"].lower() == "la"][0]
        de_correct = abs(de_splice["bonafide_start_s"] - 0.30) < 0.01  # 1st "De"
        la_correct = abs(la_splice["bonafide_start_s"] - 2.28) < 0.01  # 2nd "la"
        if de_correct and la_correct:
            print("  >> PASSED — correct duplicate instances matched\n")
            passed += 1
        else:
            print(f"  >> FAILED — wrong duplicate matched. de@{de_splice['bonafide_start_s']}, la@{la_splice['bonafide_start_s']}\n")
            failed += 1
    else:
        print(f"  >> FAILED — expected 3 splices, got {len(d5)}: {[d['word'] for d in d5]}\n")
        failed += 1

    # ── SUMMARY ──
    print("=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
