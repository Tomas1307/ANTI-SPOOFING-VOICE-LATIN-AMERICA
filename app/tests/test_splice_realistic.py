"""
Realistic splice tests using ACTUAL metadata from the partial spoof test run.

Uses real Parakeet timestamps from bonafide_transcripts.json and
alignment_metadata.json, plus simulated good/bad/partial clones.
No GPU needed — pure numpy with synthetic audio tones.
"""
import json
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

importlib.import_module("app.pipeline.partial_spoof.utils.crossfade")
splice_mod = importlib.import_module("app.pipeline.partial_spoof.utils.splice_engine")
splice_words = splice_mod.splice_words

SAMPLE_RATE = 16000
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "attacks" / "qwen_partial_spoof"


def build_audio(timestamps, base_freq=200):
    """Build mock audio from real timestamps."""
    if not timestamps:
        return np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)

    total_dur = max(ts["end"] for ts in timestamps) + 0.5
    audio = np.zeros(int(SAMPLE_RATE * total_dur), dtype=np.float32)

    for i, ts in enumerate(timestamps):
        freq = base_freq + i * 25
        start_s = int(ts["start"] * SAMPLE_RATE)
        end_s = int(ts["end"] * SAMPLE_RATE)
        t = np.linspace(0, ts["end"] - ts["start"], end_s - start_s, endpoint=False)
        audio[start_s:end_s] = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)

    return audio


def simulate_good_clone(bonafide_words):
    """Simulate a good clone: same words, slightly shifted timestamps."""
    cloned = []
    offset = np.random.uniform(-0.05, 0.05)
    for w in bonafide_words:
        stretch = np.random.uniform(0.9, 1.1)
        dur = (w["end"] - w["start"]) * stretch
        start = max(0, w["start"] + offset)
        cloned.append({
            "word": w["word"],
            "start": round(start, 4),
            "end": round(start + dur, 4),
        })
        offset += np.random.uniform(-0.02, 0.02)
    return cloned


def simulate_partial_clone(bonafide_words, drop_indices):
    """Simulate a clone that missed some words (ASR didn't detect them)."""
    cloned = []
    for i, w in enumerate(bonafide_words):
        if i in drop_indices:
            continue
        stretch = np.random.uniform(0.9, 1.1)
        dur = (w["end"] - w["start"]) * stretch
        start = max(0, w["start"] + np.random.uniform(-0.03, 0.03))
        cloned.append({
            "word": w["word"],
            "start": round(start, 4),
            "end": round(start + dur, 4),
        })
    return cloned


def run_test(name, bonafide_ts, cloned_ts, selected, expected_spliced):
    """Run a single splice test and verify."""
    bonafide_audio = build_audio(bonafide_ts, base_freq=200)
    cloned_audio = build_audio(cloned_ts, base_freq=350)

    result, details = splice_words(
        bonafide_audio=bonafide_audio,
        cloned_audio=cloned_audio,
        bonafide_words=bonafide_ts,
        cloned_words=cloned_ts,
        selected_indices=selected,
        sample_rate=SAMPLE_RATE,
        crossfade_ms=5.0,
        max_silence_steal_ms=50.0,
        max_stretch_ratio=1.1,
    )

    spliced_words = [d["word"] for d in details]
    target_words = [bonafide_ts[i]["word"] for i in selected]
    ok = len(details) == expected_spliced

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    print(f"    Target:   {target_words}")
    print(f"    Spliced:  {spliced_words} ({len(details)}/{expected_spliced})")
    for d in details:
        print(f"      [{d['word_index']}] '{d['word']}' "
              f"bf=[{d['bonafide_start_s']:.2f}-{d['bonafide_end_s']:.2f}] "
              f"cl=[{d['cloned_start_s']:.2f}-{d['cloned_end_s']:.2f}] "
              f"ratio={d['duration_ratio']:.3f}")
    return ok


def main():
    np.random.seed(42)

    # Load real bonafide timestamps
    bf_path = DATA_DIR / "bonafide_transcripts.json"
    if bf_path.exists():
        with open(bf_path, "r", encoding="utf-8") as f:
            bonafide_data = json.load(f)
        print(f"Loaded {len(bonafide_data)} real bonafide transcripts from test run\n")
    else:
        print(f"WARNING: {bf_path} not found, using hardcoded data\n")
        bonafide_data = None

    # Also load real alignment (bad clones) for comparison
    align_path = DATA_DIR / "alignment_metadata.json"
    if align_path.exists():
        with open(align_path, "r", encoding="utf-8") as f:
            alignment_data = json.load(f)
    else:
        alignment_data = None

    passed = 0
    failed = 0
    total = 0

    # ══════════════════════════════════════════════════
    # SAMPLE 1: "Dame unicamente el dato mas relevante." (6 words)
    # ══════════════════════════════════════════════════
    print("=" * 70)
    print("SAMPLE 1: 'Dame unicamente el dato mas relevante.' (6 words)")
    print("=" * 70)

    bf1 = bonafide_data["arf_00295_arf_00295_00001008290"]["word_timestamps"] if bonafide_data else []

    # 1a: Good clone — all words match
    cl1_good = simulate_good_clone(bf1)
    total += 1
    if run_test("W1 good clone — replace 'dato' (idx 3)", bf1, cl1_good, [3], 1):
        passed += 1
    else:
        failed += 1

    # 1b: Real bad clone (empty — as actually happened)
    cl1_bad = alignment_data["arf_00295_arf_00295_00001008290"]["cloned_words"] if alignment_data else []
    total += 1
    if run_test("W1 real bad clone (empty transcript)", bf1, cl1_bad, [3], 0):
        passed += 1
    else:
        failed += 1

    # ══════════════════════════════════════════════════
    # SAMPLE 2: "Hay varios cines cerca de tu hotel..." (11 words)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAMPLE 2: 'Hay varios cines cerca de tu hotel...' (11 words)")
    print("=" * 70)

    bf2 = bonafide_data["arf_00295_arf_00295_00020689215"]["word_timestamps"] if bonafide_data else []

    # 2a: Good clone — W2
    cl2_good = simulate_good_clone(bf2)
    total += 1
    if run_test("W2 good clone — replace 'varios'+'buscando' (idx 1,7)", bf2, cl2_good, [1, 7], 2):
        passed += 1
    else:
        failed += 1

    # 2b: Partial clone — dropped 'hotel. Estas' and 'especial?'
    cl2_partial = simulate_partial_clone(bf2, drop_indices={6, 10})
    total += 1
    if run_test("W2 partial clone — 'hotel' dropped, 'buscando' OK", bf2, cl2_partial, [6, 7], 1):
        passed += 1
    else:
        failed += 1

    # 2c: Real bad clone from alignment
    cl2_bad = alignment_data["arf_00295_arf_00295_00020689215"]["cloned_words"] if alignment_data else []
    total += 1
    if run_test("W2 real bad clone ('Esta letra luk lu')", bf2, cl2_bad, [1, 7], 0):
        passed += 1
    else:
        failed += 1

    # ══════════════════════════════════════════════════
    # SAMPLE 3: "Los ninos tienen mucha imaginacion." (5 words)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAMPLE 3: 'Los ninos tienen mucha imaginacion.' (5 words)")
    print("=" * 70)

    bf3 = bonafide_data["arf_00295_arf_00295_00023658548"]["word_timestamps"] if bonafide_data else []

    # 3a: Good clone — W1
    cl3_good = simulate_good_clone(bf3)
    total += 1
    if run_test("W1 good clone — replace 'tienen' (idx 2)", bf3, cl3_good, [2], 1):
        passed += 1
    else:
        failed += 1

    # 3b: Clone missing 'mucha' — try replacing it, should fail
    cl3_partial = simulate_partial_clone(bf3, drop_indices={3})
    total += 1
    if run_test("W1 clone missing 'mucha' — try idx 3, expect 0", bf3, cl3_partial, [3], 0):
        passed += 1
    else:
        failed += 1

    # 3c: Clone missing 'mucha' — try 'Los' instead, should work
    total += 1
    if run_test("W1 clone missing 'mucha' — try 'Los' (idx 0), expect 1", bf3, cl3_partial, [0], 1):
        passed += 1
    else:
        failed += 1

    # ══════════════════════════════════════════════════
    # SAMPLE 4: "Los presidentes estan muy peleados..." (11 words)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAMPLE 4: 'Los presidentes estan muy peleados...' (11 words)")
    print("=" * 70)

    bf4 = bonafide_data["arf_00295_arf_00295_00075310490"]["word_timestamps"] if bonafide_data else []

    # 4a: Good clone — W3
    cl4_good = simulate_good_clone(bf4)
    total += 1
    if run_test("W3 good clone — replace idx 1,4,8", bf4, cl4_good, [1, 4, 8], 3):
        passed += 1
    else:
        failed += 1

    # 4b: Clone missing 3 words — W3 should partially fail
    cl4_partial = simulate_partial_clone(bf4, drop_indices={1, 7, 9})
    total += 1
    if run_test("W3 partial — 'presidentes','ultimo','de' dropped", bf4, cl4_partial, [1, 4, 8], 2):
        passed += 1
    else:
        failed += 1

    # 4c: Real bad clone
    cl4_bad = alignment_data["arf_00295_arf_00295_00075310490"]["cloned_words"] if alignment_data else []
    total += 1
    if run_test("W3 real bad clone ('The Lid Halpa repositologis')", bf4, cl4_bad, [1, 4, 8], 0):
        passed += 1
    else:
        failed += 1

    # ══════════════════════════════════════════════════
    # SAMPLE 5: "Como es tu estado psicologico?" (5 words)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAMPLE 5: 'Como es tu estado psicologico?' (5 words)")
    print("  NOTE: Parakeet stores '?Como' with leading punctuation")
    print("=" * 70)

    bf5 = bonafide_data["arf_00295_arf_00295_00091439585"]["word_timestamps"] if bonafide_data else []

    # 5a: Good clone with different punctuation
    cl5_good = simulate_good_clone(bf5)
    # Simulate ASR difference: clone has "Como" instead of "?Como"
    if cl5_good:
        cl5_good[0]["word"] = "Como"
    total += 1
    if run_test("W1 punctuation mismatch — '?Como' vs 'Como'", bf5, cl5_good, [0], 1):
        passed += 1
    else:
        failed += 1

    # 5b: Real bad clone (empty)
    cl5_bad = alignment_data["arf_00295_arf_00295_00091439585"]["cloned_words"] if alignment_data else []
    total += 1
    if run_test("W1 real bad clone (empty)", bf5, cl5_bad, [0], 0):
        passed += 1
    else:
        failed += 1

    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
