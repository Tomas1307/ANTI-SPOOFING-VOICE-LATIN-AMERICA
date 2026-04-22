"""
Mock test for the partial spoof splice algorithm.

Simulates bonafide and cloned audio with known word timestamps,
runs the splice engine, and verifies the output is correct.
No GPU, no real audio, no TTS models needed — pure numpy.
"""
import numpy as np
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock heavy dependencies to avoid import chain pulling in torch/jiwer/nemo
for mod in ["torch", "jiwer", "nemo", "nemo.collections", "nemo.collections.asr",
            "librosa", "soundfile", "torchaudio", "speechbrain", "torchmetrics"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Direct import of just the files we need
crossfade_mod = importlib.import_module("app.pipeline.partial_spoof.utils.crossfade")
splice_mod = importlib.import_module("app.pipeline.partial_spoof.utils.splice_engine")
splice_words = splice_mod.splice_words


SAMPLE_RATE = 16000


def generate_tone(freq: float, duration_s: float, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave tone at a given frequency."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_silence(duration_s: float) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def build_mock_audio_with_words(words_spec: list) -> tuple:
    """Build a mock audio signal from word specifications.

    Each word_spec is a dict: {"word": str, "freq": float, "duration_s": float, "gap_after_s": float}
    Returns (audio_array, word_timestamps_list)
    """
    segments = []
    timestamps = []
    current_time = 0.3  # 300ms initial silence

    segments.append(generate_silence(0.3))

    for spec in words_spec:
        start = current_time
        tone = generate_tone(spec["freq"], spec["duration_s"])
        segments.append(tone)
        end = start + spec["duration_s"]

        timestamps.append({
            "word": spec["word"],
            "start": round(start, 4),
            "end": round(end, 4),
        })

        current_time = end
        if spec.get("gap_after_s", 0) > 0:
            segments.append(generate_silence(spec["gap_after_s"]))
            current_time += spec["gap_after_s"]

    segments.append(generate_silence(0.3))
    audio = np.concatenate(segments)
    return audio, timestamps


def test_basic_w1_splice():
    """Test W1 (1 word replaced) with matching word counts."""
    print("=" * 60)
    print("TEST 1: Basic W1 splice — replace 1 word")
    print("=" * 60)

    bonafide_words = [
        {"word": "Los", "freq": 200, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "presidentes", "freq": 250, "duration_s": 0.7, "gap_after_s": 0.05},
        {"word": "estan", "freq": 300, "duration_s": 0.4, "gap_after_s": 0.05},
        {"word": "muy", "freq": 350, "duration_s": 0.2, "gap_after_s": 0.05},
        {"word": "peleados", "freq": 400, "duration_s": 0.5, "gap_after_s": 0.0},
    ]

    cloned_words = [
        {"word": "Los", "freq": 210, "duration_s": 0.35, "gap_after_s": 0.04},
        {"word": "presidentes", "freq": 260, "duration_s": 0.65, "gap_after_s": 0.06},
        {"word": "estan", "freq": 310, "duration_s": 0.38, "gap_after_s": 0.04},
        {"word": "muy", "freq": 360, "duration_s": 0.22, "gap_after_s": 0.05},
        {"word": "peleados", "freq": 410, "duration_s": 0.48, "gap_after_s": 0.0},
    ]

    bonafide_audio, bonafide_ts = build_mock_audio_with_words(bonafide_words)
    cloned_audio, cloned_ts = build_mock_audio_with_words(cloned_words)

    print(f"\nBonafide: {len(bonafide_audio)} samples ({len(bonafide_audio)/SAMPLE_RATE:.3f}s)")
    print(f"Cloned:   {len(cloned_audio)} samples ({len(cloned_audio)/SAMPLE_RATE:.3f}s)")
    print(f"\nBonafide timestamps:")
    for w in bonafide_ts:
        print(f"  [{w['start']:.3f}-{w['end']:.3f}] {w['word']}")
    print(f"\nCloned timestamps:")
    for w in cloned_ts:
        print(f"  [{w['start']:.3f}-{w['end']:.3f}] {w['word']}")

    selected = [2]  # Replace "estan"
    print(f"\nReplacing word index {selected}: '{bonafide_ts[2]['word']}'")

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

    print(f"\nResult: {len(result)} samples ({len(result)/SAMPLE_RATE:.3f}s)")
    print(f"Splice details: {details}")
    print(f"Words spliced: {len(details)}")
    assert len(details) == 1, f"Expected 1 splice, got {len(details)}"
    assert details[0]["word"] == "estan"
    print("PASSED")


def test_w2_non_adjacent():
    """Test W2 (2 words replaced, non-adjacent)."""
    print("\n" + "=" * 60)
    print("TEST 2: W2 splice — replace 2 non-adjacent words")
    print("=" * 60)

    words_spec = [
        {"word": "Hay", "freq": 200, "duration_s": 0.25, "gap_after_s": 0.05},
        {"word": "varios", "freq": 250, "duration_s": 0.4, "gap_after_s": 0.05},
        {"word": "cines", "freq": 300, "duration_s": 0.35, "gap_after_s": 0.05},
        {"word": "cerca", "freq": 350, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "de", "freq": 370, "duration_s": 0.15, "gap_after_s": 0.03},
        {"word": "tu", "freq": 380, "duration_s": 0.15, "gap_after_s": 0.05},
        {"word": "hotel", "freq": 400, "duration_s": 0.4, "gap_after_s": 0.1},
        {"word": "estas", "freq": 420, "duration_s": 0.35, "gap_after_s": 0.05},
        {"word": "buscando", "freq": 440, "duration_s": 0.45, "gap_after_s": 0.05},
        {"word": "algo", "freq": 460, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "especial", "freq": 480, "duration_s": 0.5, "gap_after_s": 0.0},
    ]

    cloned_spec = [
        {"word": "Hay", "freq": 205, "duration_s": 0.28, "gap_after_s": 0.04},
        {"word": "varios", "freq": 255, "duration_s": 0.42, "gap_after_s": 0.06},
        {"word": "cines", "freq": 305, "duration_s": 0.33, "gap_after_s": 0.04},
        {"word": "cerca", "freq": 355, "duration_s": 0.32, "gap_after_s": 0.06},
        {"word": "de", "freq": 375, "duration_s": 0.13, "gap_after_s": 0.03},
        {"word": "tu", "freq": 385, "duration_s": 0.16, "gap_after_s": 0.04},
        {"word": "hotel", "freq": 405, "duration_s": 0.38, "gap_after_s": 0.12},
        {"word": "estas", "freq": 425, "duration_s": 0.37, "gap_after_s": 0.04},
        {"word": "buscando", "freq": 445, "duration_s": 0.43, "gap_after_s": 0.06},
        {"word": "algo", "freq": 465, "duration_s": 0.28, "gap_after_s": 0.04},
        {"word": "especial", "freq": 485, "duration_s": 0.52, "gap_after_s": 0.0},
    ]

    bonafide_audio, bonafide_ts = build_mock_audio_with_words(words_spec)
    cloned_audio, cloned_ts = build_mock_audio_with_words(cloned_spec)

    print(f"\nBonafide: {len(bonafide_ts)} words, {len(bonafide_audio)/SAMPLE_RATE:.3f}s")
    print(f"Cloned:   {len(cloned_ts)} words, {len(cloned_audio)/SAMPLE_RATE:.3f}s")

    selected = [1, 7]  # Replace "varios" and "estas" (non-adjacent)
    print(f"\nReplacing indices {selected}: '{bonafide_ts[1]['word']}' and '{bonafide_ts[7]['word']}'")

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

    print(f"\nResult: {len(result)} samples ({len(result)/SAMPLE_RATE:.3f}s)")
    print(f"Splice details:")
    for d in details:
        print(f"  [{d['word_index']}] '{d['word']}' ratio={d['duration_ratio']:.3f}")
    assert len(details) == 2, f"Expected 2 splices, got {len(details)}"
    print("PASSED")


def test_mismatched_word_count():
    """Test when cloned audio has FEWER words than bonafide — the real bug."""
    print("\n" + "=" * 60)
    print("TEST 3: Mismatched word count — cloned has fewer words")
    print("  This is the REAL BUG: clone produces 4 words, bonafide has 11")
    print("=" * 60)

    bonafide_words = [
        {"word": "Hay", "freq": 200, "duration_s": 0.25, "gap_after_s": 0.05},
        {"word": "varios", "freq": 250, "duration_s": 0.4, "gap_after_s": 0.05},
        {"word": "cines", "freq": 300, "duration_s": 0.35, "gap_after_s": 0.05},
        {"word": "cerca", "freq": 350, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "de", "freq": 370, "duration_s": 0.15, "gap_after_s": 0.03},
        {"word": "tu", "freq": 380, "duration_s": 0.15, "gap_after_s": 0.05},
        {"word": "hotel", "freq": 400, "duration_s": 0.4, "gap_after_s": 0.1},
        {"word": "estas", "freq": 420, "duration_s": 0.35, "gap_after_s": 0.05},
        {"word": "buscando", "freq": 440, "duration_s": 0.45, "gap_after_s": 0.05},
        {"word": "algo", "freq": 460, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "especial", "freq": 480, "duration_s": 0.5, "gap_after_s": 0.0},
    ]

    # Simulate bad clone: only 4 garbled words instead of 11
    cloned_words_bad = [
        {"word": "Esta", "freq": 210, "duration_s": 0.3, "gap_after_s": 0.1},
        {"word": "letra", "freq": 260, "duration_s": 0.2, "gap_after_s": 0.15},
        {"word": "luk", "freq": 310, "duration_s": 0.8, "gap_after_s": 0.1},
        {"word": "lu", "freq": 330, "duration_s": 0.1, "gap_after_s": 0.0},
    ]

    bonafide_audio, bonafide_ts = build_mock_audio_with_words(bonafide_words)
    cloned_audio, cloned_ts = build_mock_audio_with_words(cloned_words_bad)

    print(f"\nBonafide: {len(bonafide_ts)} words, {len(bonafide_audio)/SAMPLE_RATE:.3f}s")
    print(f"Cloned:   {len(cloned_ts)} words, {len(cloned_audio)/SAMPLE_RATE:.3f}s")
    print(f"  (Clone has garbled text — different words entirely)")

    selected_w1 = [0]  # Index 0 exists in both
    selected_w2 = [0, 7]  # Index 7 only in bonafide
    selected_w3 = [1, 5, 9]  # Indices 5 and 9 only in bonafide

    print(f"\n--- W1: Replace index {selected_w1} ---")
    result1, details1 = splice_words(
        bonafide_audio=bonafide_audio.copy(),
        cloned_audio=cloned_audio,
        bonafide_words=bonafide_ts,
        cloned_words=cloned_ts,
        selected_indices=selected_w1,
        sample_rate=SAMPLE_RATE,
        crossfade_ms=5.0,
        max_silence_steal_ms=50.0,
        max_stretch_ratio=1.1,
    )
    print(f"  Spliced: {len(details1)} words (expected 1)")
    for d in details1:
        print(f"    [{d['word_index']}] bonafide='{bonafide_ts[d['word_index']]['word']}' "
              f"-> cloned='{cloned_ts[d['word_index']]['word']}' ratio={d['duration_ratio']:.3f}")
    print(f"  NOTE: bonafide word 'Hay' replaced with cloned word 'Esta' — WRONG WORD!")

    print(f"\n--- W2: Replace indices {selected_w2} ---")
    result2, details2 = splice_words(
        bonafide_audio=bonafide_audio.copy(),
        cloned_audio=cloned_audio,
        bonafide_words=bonafide_ts,
        cloned_words=cloned_ts,
        selected_indices=selected_w2,
        sample_rate=SAMPLE_RATE,
        crossfade_ms=5.0,
        max_silence_steal_ms=50.0,
        max_stretch_ratio=1.1,
    )
    print(f"  Spliced: {len(details2)} words (expected 2, likely got 1 — index 7 out of range)")

    print(f"\n--- W3: Replace indices {selected_w3} ---")
    result3, details3 = splice_words(
        bonafide_audio=bonafide_audio.copy(),
        cloned_audio=cloned_audio,
        bonafide_words=bonafide_ts,
        cloned_words=cloned_ts,
        selected_indices=selected_w3,
        sample_rate=SAMPLE_RATE,
        crossfade_ms=5.0,
        max_silence_steal_ms=50.0,
        max_stretch_ratio=1.1,
    )
    print(f"  Spliced: {len(details3)} words (expected 3, likely got 1 — indices 5,9 out of range)")

    print("\n" + "-" * 60)
    print("DIAGNOSIS: The splice engine uses POSITIONAL matching (index 0 -> index 0)")
    print("  When clone has different words, it splices the WRONG word content!")
    print("  'Hay' (bonafide[0]) gets replaced with 'Esta' (cloned[0])")
    print("  This is semantically wrong but algorithmically 'correct'")
    print("")
    print("THE REAL FIX: We need WORD-LEVEL matching, not positional matching.")
    print("  Find 'Hay' in the cloned timestamps by matching the word text,")
    print("  not by assuming index 0 = index 0.")


def test_text_matching_with_bad_clone():
    """Test that text-matching correctly REJECTS bad clones."""
    print("\n" + "=" * 60)
    print("TEST 4: Text-matching with BAD clone — should reject all")
    print("=" * 60)

    bonafide_words = [
        {"word": "Hay", "freq": 200, "duration_s": 0.25, "gap_after_s": 0.05},
        {"word": "varios", "freq": 250, "duration_s": 0.4, "gap_after_s": 0.05},
        {"word": "cines", "freq": 300, "duration_s": 0.35, "gap_after_s": 0.05},
        {"word": "cerca", "freq": 350, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "de", "freq": 370, "duration_s": 0.15, "gap_after_s": 0.03},
    ]

    cloned_words_bad = [
        {"word": "Esta", "freq": 210, "duration_s": 0.3, "gap_after_s": 0.1},
        {"word": "letra", "freq": 260, "duration_s": 0.2, "gap_after_s": 0.15},
        {"word": "luk", "freq": 310, "duration_s": 0.8, "gap_after_s": 0.0},
    ]

    bonafide_audio, bonafide_ts = build_mock_audio_with_words(bonafide_words)
    cloned_audio, cloned_ts = build_mock_audio_with_words(cloned_words_bad)

    print(f"\nBonafide words: {[w['word'] for w in bonafide_ts]}")
    print(f"Cloned words:   {[w['word'] for w in cloned_ts]}")
    print(f"No overlap in vocabulary — all splices should be rejected")

    selected = [0, 2]  # "Hay" and "cines" — neither exists in clone
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

    print(f"\nSpliced: {len(details)} words (expected 0 — no matching words)")
    assert len(details) == 0, f"Expected 0 splices with bad clone, got {len(details)}"
    print("PASSED — bad clones correctly produce zero splices")


def test_text_matching_with_good_clone():
    """Test that text-matching works with good clones (same words)."""
    print("\n" + "=" * 60)
    print("TEST 5: Text-matching with GOOD clone — should splice correctly")
    print("=" * 60)

    words_spec = [
        {"word": "Los", "freq": 200, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "presidentes", "freq": 250, "duration_s": 0.7, "gap_after_s": 0.05},
        {"word": "estan", "freq": 300, "duration_s": 0.4, "gap_after_s": 0.05},
        {"word": "muy", "freq": 350, "duration_s": 0.2, "gap_after_s": 0.05},
        {"word": "peleados", "freq": 400, "duration_s": 0.5, "gap_after_s": 0.05},
        {"word": "por", "freq": 420, "duration_s": 0.15, "gap_after_s": 0.05},
        {"word": "el", "freq": 430, "duration_s": 0.1, "gap_after_s": 0.05},
        {"word": "ultimo", "freq": 440, "duration_s": 0.35, "gap_after_s": 0.05},
        {"word": "caso", "freq": 450, "duration_s": 0.3, "gap_after_s": 0.0},
    ]

    cloned_spec = [
        {"word": "Los", "freq": 205, "duration_s": 0.28, "gap_after_s": 0.06},
        {"word": "presidentes", "freq": 255, "duration_s": 0.72, "gap_after_s": 0.04},
        {"word": "estan", "freq": 305, "duration_s": 0.42, "gap_after_s": 0.06},
        {"word": "muy", "freq": 355, "duration_s": 0.18, "gap_after_s": 0.04},
        {"word": "peleados", "freq": 405, "duration_s": 0.52, "gap_after_s": 0.06},
        {"word": "por", "freq": 425, "duration_s": 0.13, "gap_after_s": 0.04},
        {"word": "el", "freq": 435, "duration_s": 0.12, "gap_after_s": 0.06},
        {"word": "ultimo", "freq": 445, "duration_s": 0.33, "gap_after_s": 0.04},
        {"word": "caso", "freq": 455, "duration_s": 0.32, "gap_after_s": 0.0},
    ]

    bonafide_audio, bonafide_ts = build_mock_audio_with_words(words_spec)
    cloned_audio, cloned_ts = build_mock_audio_with_words(cloned_spec)

    print(f"\nBonafide: {[w['word'] for w in bonafide_ts]}")
    print(f"Cloned:   {[w['word'] for w in cloned_ts]}")

    selected = [1, 4, 7]  # "presidentes", "peleados", "ultimo" (W3, non-adjacent)
    print(f"Replacing: {[bonafide_ts[i]['word'] for i in selected]}")

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

    print(f"\nSpliced: {len(details)} words (expected 3)")
    for d in details:
        print(f"  [{d['word_index']}] '{d['word']}' ratio={d['duration_ratio']:.3f}")
    assert len(details) == 3, f"Expected 3 splices, got {len(details)}"
    assert all(d["word"] == bonafide_ts[d["word_index"]]["word"] for d in details)
    print("PASSED — all 3 words correctly matched and spliced")


def test_partial_match():
    """Test when clone has SOME matching words but not all."""
    print("\n" + "=" * 60)
    print("TEST 6: Partial match — clone has 3 of 5 requested words")
    print("=" * 60)

    bonafide_words = [
        {"word": "Hay", "freq": 200, "duration_s": 0.25, "gap_after_s": 0.05},
        {"word": "varios", "freq": 250, "duration_s": 0.4, "gap_after_s": 0.05},
        {"word": "cines", "freq": 300, "duration_s": 0.35, "gap_after_s": 0.05},
        {"word": "cerca", "freq": 350, "duration_s": 0.3, "gap_after_s": 0.05},
        {"word": "de", "freq": 370, "duration_s": 0.15, "gap_after_s": 0.0},
    ]

    # Clone got some right, some wrong
    cloned_words_partial = [
        {"word": "Hay", "freq": 210, "duration_s": 0.28, "gap_after_s": 0.05},
        {"word": "muchos", "freq": 260, "duration_s": 0.35, "gap_after_s": 0.05},  # WRONG
        {"word": "cines", "freq": 310, "duration_s": 0.33, "gap_after_s": 0.05},
        {"word": "cerca", "freq": 355, "duration_s": 0.32, "gap_after_s": 0.05},
        {"word": "del", "freq": 375, "duration_s": 0.13, "gap_after_s": 0.0},  # WRONG
    ]

    bonafide_audio, bonafide_ts = build_mock_audio_with_words(bonafide_words)
    cloned_audio, cloned_ts = build_mock_audio_with_words(cloned_words_partial)

    print(f"\nBonafide: {[w['word'] for w in bonafide_ts]}")
    print(f"Cloned:   {[w['word'] for w in cloned_ts]}")
    print(f"Match:    Hay=yes, varios/muchos=NO, cines=yes, cerca=yes, de/del=NO")

    selected = [0, 1, 3]  # "Hay", "varios", "cerca"
    print(f"\nReplacing: {[bonafide_ts[i]['word'] for i in selected]}")

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

    print(f"\nSpliced: {len(details)} words (expected 2 — 'Hay' and 'cerca', 'varios' rejected)")
    for d in details:
        print(f"  [{d['word_index']}] '{d['word']}' ratio={d['duration_ratio']:.3f}")
    assert len(details) == 2, f"Expected 2 splices, got {len(details)}"
    spliced_words = {d["word"] for d in details}
    assert "Hay" in spliced_words, "'Hay' should have been spliced"
    assert "cerca" in spliced_words, "'cerca' should have been spliced"
    assert "varios" not in spliced_words, "'varios' should NOT have been spliced"
    print("PASSED — only matching words spliced, mismatches rejected")


if __name__ == "__main__":
    test_basic_w1_splice()
    test_w2_non_adjacent()
    test_mismatched_word_count()
    test_text_matching_with_bad_clone()
    test_text_matching_with_good_clone()
    test_partial_match()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
