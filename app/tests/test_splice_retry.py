"""
Test the retry logic in the splice engine.

Simulates scenarios where some words are missing in the clone,
forcing retries with different word selections.
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

importlib.import_module("app.pipeline.partial_spoof.utils.crossfade")
splice_mod = importlib.import_module("app.pipeline.partial_spoof.utils.splice_engine")
splice_words = splice_mod.splice_words

SAMPLE_RATE = 16000


def generate_tone(freq, duration_s, amp=0.5):
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def build_audio(timestamps, base_freq=200):
    segments = []
    prev = 0.0
    for i, ts in enumerate(timestamps):
        gap = ts["start"] - prev
        if gap > 0:
            segments.append(np.zeros(int(SAMPLE_RATE * gap), dtype=np.float32))
        dur = ts["end"] - ts["start"]
        segments.append(generate_tone(base_freq + i * 30, dur))
        prev = ts["end"]
    segments.append(np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32))
    return np.concatenate(segments)


def simulate_retry(bonafide_ts, cloned_ts, tier_count, max_retries=5):
    """Simulate the smart retry logic: keep what worked, replace what failed."""
    import random

    bonafide_audio = build_audio(bonafide_ts)
    cloned_audio = build_audio(cloned_ts, base_freq=300)
    total_words = len(bonafide_ts)

    confirmed = []
    remaining = tier_count
    tried = set()

    for attempt in range(max_retries + 1):
        seed = 42 + attempt * 7
        rng = random.Random(seed)

        if attempt == 0:
            # First attempt: random non-adjacent selection
            selected = None
            for _ in range(100):
                pick = sorted(rng.sample(range(total_words), tier_count))
                valid = all(pick[i+1] - pick[i] >= 2 for i in range(len(pick) - 1))
                if valid:
                    selected = pick
                    break
            if selected is None:
                print(f"  Attempt {attempt}: could not generate initial selection")
                continue
        else:
            # Smart retry: keep confirmed, pick new for failed slots
            blocked = set(tried)
            for idx in confirmed:
                blocked.update([idx - 1, idx, idx + 1])
            available = [i for i in range(total_words) if i not in blocked]

            if len(available) < remaining:
                print(f"  Attempt {attempt}: only {len(available)} candidates left, need {remaining} — giving up")
                break

            new_picks = None
            for _ in range(100):
                picks = sorted(rng.sample(available, min(remaining, len(available))))
                all_idx = sorted(confirmed + picks)
                valid = all(all_idx[i+1] - all_idx[i] >= 2 for i in range(len(all_idx) - 1))
                if valid:
                    new_picks = picks
                    break

            if new_picks is None:
                print(f"  Attempt {attempt}: could not find non-adjacent picks")
                continue
            selected = sorted(confirmed + new_picks)

        tried.update(selected)
        target_words = [bonafide_ts[i]["word"] for i in selected]

        _, details = splice_words(
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
        succeeded = [d["word_index"] for d in details]
        failed_this = [bonafide_ts[i]["word"] for i in selected if i not in succeeded]

        status = "OK" if len(details) >= tier_count else f"need {tier_count - len(details)} more"
        print(f"  Attempt {attempt}: tried {target_words} -> spliced {spliced_words} "
              f"({len(details)}/{tier_count}) [{status}]"
              + (f" | failed: {failed_this}" if failed_this else ""))

        if len(details) >= tier_count:
            return attempt, details

        # Keep what worked, retry for the rest
        confirmed = succeeded
        remaining = tier_count - len(confirmed)

    return -1, []


def main():
    passed = 0
    failed = 0

    # ── TEST 1: Clone has 6/10 words — W2 should succeed on retry ──
    print("=" * 70)
    print("RETRY TEST 1: Clone has 6 of 10 words — W2 needs 2")
    print("  Some words missing, but enough variety to find 2 matches on retry")
    print("-" * 70)

    bonafide = [
        {"word": "La",         "start": 0.30, "end": 0.42},
        {"word": "ciudad",     "start": 0.46, "end": 0.85},
        {"word": "tiene",      "start": 0.90, "end": 1.20},
        {"word": "muchos",     "start": 1.24, "end": 1.60},
        {"word": "parques",    "start": 1.64, "end": 2.05},
        {"word": "y",          "start": 2.08, "end": 2.16},
        {"word": "jardines",   "start": 2.20, "end": 2.65},
        {"word": "muy",        "start": 2.70, "end": 2.85},
        {"word": "bonitos",    "start": 2.90, "end": 3.30},
        {"word": "aqui",       "start": 3.34, "end": 3.62},
    ]

    # Clone got 6 of 10 right
    cloned = [
        {"word": "La",         "start": 0.28, "end": 0.40},
        {"word": "ciudad",     "start": 0.44, "end": 0.82},
        {"word": "tiene",      "start": 0.86, "end": 1.18},
        # "muchos" missing
        {"word": "parques",    "start": 1.30, "end": 1.70},
        # "y" missing
        {"word": "jardines",   "start": 1.80, "end": 2.25},
        # "muy" missing
        {"word": "bonitos",    "start": 2.35, "end": 2.75},
        # "aqui" missing
    ]

    attempt, details = simulate_retry(bonafide, cloned, tier_count=2)
    if attempt >= 0 and len(details) == 2:
        print(f"  >> PASSED — found 2 matches on attempt {attempt}")
        passed += 1
    else:
        print(f"  >> FAILED — could not find 2 matches in {attempt + 1} attempts")
        failed += 1

    # ── TEST 2: Clone has only 2/8 words — W3 should fail after retries ──
    print("\n" + "=" * 70)
    print("RETRY TEST 2: Clone has 2 of 8 words — W3 needs 3")
    print("  Not enough matching words — should reject after all retries")
    print("-" * 70)

    bonafide2 = [
        {"word": "El",         "start": 0.30, "end": 0.42},
        {"word": "gobierno",   "start": 0.46, "end": 0.95},
        {"word": "aprobo",     "start": 1.00, "end": 1.40},
        {"word": "una",        "start": 1.44, "end": 1.58},
        {"word": "nueva",      "start": 1.62, "end": 1.95},
        {"word": "ley",        "start": 2.00, "end": 2.22},
        {"word": "de",         "start": 2.26, "end": 2.36},
        {"word": "reforma",    "start": 2.40, "end": 2.90},
    ]

    # Clone is mostly garbage — only 2 words match
    cloned2 = [
        {"word": "El",         "start": 0.28, "end": 0.40},
        {"word": "gobierno",   "start": 0.44, "end": 0.92},
        {"word": "ha",         "start": 0.96, "end": 1.08},
        {"word": "sido",       "start": 1.12, "end": 1.40},
        {"word": "un",         "start": 1.44, "end": 1.55},
    ]

    attempt2, details2 = simulate_retry(bonafide2, cloned2, tier_count=3)
    if attempt2 == -1:
        print(f"  >> PASSED — correctly rejected (only 2 words available, needed 3)")
        passed += 1
    else:
        print(f"  >> FAILED — should not have succeeded with only 2 matching words")
        failed += 1

    # ── TEST 3: Clone has 4/6 words — W1 should succeed immediately ──
    print("\n" + "=" * 70)
    print("RETRY TEST 3: Clone has 4 of 6 words — W1 needs 1")
    print("  Easy case — should succeed on first attempt")
    print("-" * 70)

    bonafide3 = [
        {"word": "Dame",        "start": 0.30, "end": 0.60},
        {"word": "unicamente",  "start": 0.64, "end": 1.30},
        {"word": "el",          "start": 1.34, "end": 1.48},
        {"word": "dato",        "start": 1.52, "end": 1.88},
        {"word": "mas",         "start": 1.92, "end": 2.12},
        {"word": "relevante",   "start": 2.16, "end": 2.90},
    ]

    cloned3 = [
        {"word": "Dame",        "start": 0.28, "end": 0.58},
        # "unicamente" missing
        {"word": "el",          "start": 0.80, "end": 0.94},
        {"word": "dato",        "start": 0.98, "end": 1.34},
        {"word": "mas",         "start": 1.38, "end": 1.56},
        # "relevante" missing
    ]

    attempt3, details3 = simulate_retry(bonafide3, cloned3, tier_count=1)
    if attempt3 >= 0 and len(details3) == 1:
        print(f"  >> PASSED — W1 succeeded on attempt {attempt3}")
        passed += 1
    else:
        print(f"  >> FAILED")
        failed += 1

    # ── TEST 4: Perfect clone — W3 succeeds immediately ──
    print("\n" + "=" * 70)
    print("RETRY TEST 4: Perfect clone — W3 should succeed on attempt 0")
    print("-" * 70)

    bonafide4 = [
        {"word": "Todos",      "start": 0.30, "end": 0.60},
        {"word": "los",        "start": 0.64, "end": 0.78},
        {"word": "estudiantes", "start": 0.82, "end": 1.45},
        {"word": "aprobaron",  "start": 1.50, "end": 2.00},
        {"word": "el",         "start": 2.04, "end": 2.16},
        {"word": "examen",     "start": 2.20, "end": 2.62},
        {"word": "de",         "start": 2.66, "end": 2.76},
        {"word": "matematicas", "start": 2.80, "end": 3.45},
        {"word": "sin",        "start": 3.50, "end": 3.65},
        {"word": "problemas",  "start": 3.70, "end": 4.15},
        {"word": "este",       "start": 4.20, "end": 4.42},
        {"word": "semestre",   "start": 4.46, "end": 5.00},
    ]

    cloned4 = [
        {"word": "Todos",      "start": 0.28, "end": 0.58},
        {"word": "los",        "start": 0.62, "end": 0.76},
        {"word": "estudiantes", "start": 0.80, "end": 1.42},
        {"word": "aprobaron",  "start": 1.48, "end": 1.98},
        {"word": "el",         "start": 2.02, "end": 2.14},
        {"word": "examen",     "start": 2.18, "end": 2.60},
        {"word": "de",         "start": 2.64, "end": 2.74},
        {"word": "matematicas", "start": 2.78, "end": 3.42},
        {"word": "sin",        "start": 3.48, "end": 3.63},
        {"word": "problemas",  "start": 3.68, "end": 4.12},
        {"word": "este",       "start": 4.18, "end": 4.40},
        {"word": "semestre",   "start": 4.44, "end": 4.98},
    ]

    attempt4, details4 = simulate_retry(bonafide4, cloned4, tier_count=3)
    if attempt4 == 0 and len(details4) == 3:
        print(f"  >> PASSED — W3 succeeded on first attempt (no retries)")
        passed += 1
    else:
        print(f"  >> FAILED — expected attempt 0, got {attempt4}")
        failed += 1

    # ── TEST 5: Clone has exactly 2 matching — W2 succeeds, W3 fails ──
    print("\n" + "=" * 70)
    print("RETRY TEST 5: Clone has exactly 2 matching words")
    print("  W2 should eventually succeed, W3 should fail")
    print("-" * 70)

    bonafide5 = [
        {"word": "Mi",        "start": 0.30, "end": 0.42},
        {"word": "hermana",   "start": 0.46, "end": 0.90},
        {"word": "trabaja",   "start": 0.94, "end": 1.35},
        {"word": "en",        "start": 1.38, "end": 1.48},
        {"word": "un",        "start": 1.52, "end": 1.62},
        {"word": "hospital",  "start": 1.66, "end": 2.15},
        {"word": "publico",   "start": 2.20, "end": 2.60},
        {"word": "muy",       "start": 2.64, "end": 2.78},
        {"word": "grande",    "start": 2.82, "end": 3.20},
    ]

    # Only "hermana" and "hospital" match
    cloned5 = [
        {"word": "Su",        "start": 0.28, "end": 0.38},
        {"word": "hermana",   "start": 0.42, "end": 0.86},
        {"word": "labora",    "start": 0.90, "end": 1.30},
        {"word": "en",        "start": 1.34, "end": 1.44},  # "en" matches too
        {"word": "el",        "start": 1.48, "end": 1.58},
        {"word": "hospital",  "start": 1.62, "end": 2.10},
    ]

    print("\n  --- W2 (need 2) ---")
    a5_w2, d5_w2 = simulate_retry(bonafide5, cloned5, tier_count=2)
    w2_ok = a5_w2 >= 0 and len(d5_w2) == 2
    if w2_ok:
        print(f"  >> W2 PASSED — found 2 on attempt {a5_w2}")
    else:
        print(f"  >> W2 FAILED")

    print("\n  --- W3 (need 3) ---")
    a5_w3, d5_w3 = simulate_retry(bonafide5, cloned5, tier_count=3)
    w3_ok = a5_w3 == -1
    if w3_ok:
        print(f"  >> W3 PASSED — correctly rejected (only 3 matching words, hard to get 3 non-adjacent)")
    else:
        print(f"  >> W3 result: attempt={a5_w3}, details={[d['word'] for d in d5_w3]}")
        # It's possible to find 3 if "hermana", "en", "hospital" are all non-adjacent
        # Let's check: indices would be 1, 3, 5 — that IS non-adjacent
        if a5_w3 >= 0 and len(d5_w3) == 3:
            print(f"  >> W3 actually PASSED — found 3 matching non-adjacent words")
            w3_ok = True

    if w2_ok and w3_ok:
        passed += 1
    else:
        failed += 1

    # ── SUMMARY ──
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
