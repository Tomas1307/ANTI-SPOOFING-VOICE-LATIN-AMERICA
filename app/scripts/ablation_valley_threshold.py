"""
Ablation: sweep the valley-score eligibility threshold and measure
word-selection yield (self-contained, no app.pipeline imports).

Reuses the exact scoring logic from
app/pipeline/partial_spoof/utils/valley_scorer.py and the selection logic
from app/pipeline/partial_spoof/steps/step_04_select_words.py, copied here
standalone so this script has zero dependency on torch (the partial_spoof
package's __init__.py pulls in pipeline_facade -> settings -> torch, which
is not actually needed for word selection and was unavailable in the
venvs tested on 2026-10-24).

Reads already-generated clone audio and forced-alignment metadata from
disk (Steps 1-3 outputs of the production run). Writes nothing back to the
production directory; only reads from it.

Usage:
    python3 app/scripts/ablation_valley_threshold.py <production_output_dir>

    <production_output_dir> is the attack system's partial-spoof output
    directory that already contains alignment_metadata.json, e.g.
    data/partial_spoof_output/omnivoice/canonical
"""
import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np

THRESHOLDS = [0.3, 0.5, 0.65, 0.8, 1.0]

VALLEY_SCORE_WINDOW_MS = 100.0
VALLEY_SCORE_FRAME_MS = 5.0
MIN_WORD_DURATION_MS = 200.0
MAX_STRETCH_RATIO = 1.25
MIN_WORDS = {"W1": 4, "W2": 8, "W3": 12}
TIER_COUNT = {"W1": 1, "W2": 2, "W3": 3}
ENABLED_TIERS = ["W1", "W2", "W3"]
RANDOM_SEED = 42
SAMPLE_RATE = 16000

_FRAME_SAMPLES = int(VALLEY_SCORE_FRAME_MS * SAMPLE_RATE / 1000)
_WINDOW_SAMPLES = int(VALLEY_SCORE_WINDOW_MS * SAMPLE_RATE / 1000)


def compute_energy_envelope(audio: np.ndarray) -> np.ndarray:
    n_frames = len(audio) // _FRAME_SAMPLES
    if n_frames == 0:
        return np.zeros(1, dtype=np.float32)
    frames = audio[: n_frames * _FRAME_SAMPLES].reshape(n_frames, _FRAME_SAMPLES)
    return np.sqrt(np.mean(frames**2, axis=1)).astype(np.float32)


def boundary_score(energy: np.ndarray, boundary_s: float) -> float:
    center_frame = int(boundary_s * 1000 / VALLEY_SCORE_FRAME_MS)
    window_frames = int(VALLEY_SCORE_WINDOW_MS / VALLEY_SCORE_FRAME_MS)
    start = max(0, center_frame - window_frames)
    end = min(len(energy), center_frame + window_frames)
    if start >= end:
        return 1.0
    local = energy[start:end]
    avg_val = float(np.mean(local))
    if avg_val < 1e-8:
        return 0.0
    return float(np.min(local)) / avg_val


def score_words(cloned_audio, cloned_words, bonafide_words):
    """Returns list of dicts: word_index, combined_score, duration_ms, eligible_base."""
    energy = compute_energy_envelope(cloned_audio)
    n_words = min(len(cloned_words), len(bonafide_words))
    scores = []
    for i in range(n_words):
        cw, bw = cloned_words[i], bonafide_words[i]
        c_dur_ms = (cw["end"] - cw["start"]) * 1000
        b_dur_ms = (bw["end"] - bw["start"]) * 1000
        stretch = c_dur_ms / b_dur_ms if b_dur_ms > 0 else 999.0

        left = boundary_score(energy, cw["start"])
        right = boundary_score(energy, cw["end"])
        if i == 0:
            combined = right
        elif i == n_words - 1:
            combined = left
        else:
            combined = max(left, right)

        duration_ok = c_dur_ms >= MIN_WORD_DURATION_MS
        stretch_ok = (1.0 / MAX_STRETCH_RATIO) <= stretch <= MAX_STRETCH_RATIO

        scores.append({
            "word_index": i,
            "word": cw["word"],
            "combined_score": combined,
            "duration_ok": duration_ok,
            "stretch_ok": stretch_ok,
        })
    scores.sort(key=lambda s: s["combined_score"])
    return scores


def greedy_select(candidates, n_select):
    selected = []
    for c in candidates:
        if all(abs(c["word_index"] - s) >= 2 for s in selected):
            selected.append(c["word_index"])
            if len(selected) == n_select:
                return sorted(selected)
    return None


def select_by_valley_score(eligible, n_select, rng):
    if len(eligible) < n_select:
        return None
    selected = greedy_select(eligible, n_select)
    if selected is not None:
        return selected
    top_pool = eligible[: min(len(eligible), n_select * 4)]
    for _ in range(50):
        shuffled = list(top_pool)
        rng.shuffle(shuffled)
        selected = greedy_select(shuffled, n_select)
        if selected is not None:
            return selected
    return None


def run_threshold(alignment_data, threshold):
    tier_counts = {t: 0 for t in ENABLED_TIERS}
    total_selections = 0
    skipped_no_eligible = 0

    for sample_key, entry in alignment_data.items():
        word_count = entry["word_count"]
        bonafide_words = entry["bonafide_words"]
        cloned_words = entry["cloned_words"]

        audio_path = entry["cloned_audio_path"]
        try:
            cloned_audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception:
            continue

        all_scores = score_words(cloned_audio, cloned_words, bonafide_words)
        eligible = [
            s for s in all_scores
            if s["combined_score"] <= threshold and s["duration_ok"] and s["stretch_ok"]
        ]
        if not eligible:
            skipped_no_eligible += 1
            continue

        for tier in ENABLED_TIERS:
            if word_count < MIN_WORDS[tier]:
                continue
            n_replace = TIER_COUNT[tier]
            per_sample_seed = RANDOM_SEED + hash(sample_key + tier) % (2**31)
            rng = random.Random(per_sample_seed)
            selected = select_by_valley_score(eligible, n_replace, rng)
            if selected is None:
                continue
            tier_counts[tier] += 1
            total_selections += 1

    return {
        "threshold": threshold,
        "total_selections": total_selections,
        "tier_counts": tier_counts,
        "skipped_no_eligible": skipped_no_eligible,
        "n_utterances": len(alignment_data),
    }


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    production_output_dir = Path(sys.argv[1])
    alignment_src = production_output_dir / "alignment_metadata.json"
    if not alignment_src.exists():
        raise FileNotFoundError(f"{alignment_src} not found.")

    with open(alignment_src, "r", encoding="utf-8") as f:
        alignment_data = json.load(f)

    results = []
    for threshold in THRESHOLDS:
        print(f"Running threshold {threshold:.2f} ({len(alignment_data)} utterances)...")
        row = run_threshold(alignment_data, threshold)
        results.append(row)
        print(f"  -> {row}")

    system_name = production_output_dir.parent.name if production_output_dir.name in (
        "canonical", "jittered"
    ) else production_output_dir.name
    out_dir = Path("data/ablation_valley_threshold")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{system_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Valley score threshold ablation summary ===")
    print(f"{'Threshold':>10} | {'Total plans':>12} | Tier counts")
    for row in results:
        print(f"{row['threshold']:>10.2f} | {row['total_selections']:>12} | {row['tier_counts']}")
    print(f"\nFull results written to {summary_path}")


if __name__ == "__main__":
    main()
