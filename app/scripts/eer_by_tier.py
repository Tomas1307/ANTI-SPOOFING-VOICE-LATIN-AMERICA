"""
Compute partial-spoof EER pooled by tier (W1/W2/W3), across all six
attack systems, for each of the four baseline detectors.

Self-contained (numpy only, no app.pipeline imports) to avoid the
torch-dependency issues hit earlier this session. Reads the raw score
files the training pipeline already wrote (audio_id attack_id key score,
space-delimited, ASVspoof convention), which is the same data
`per_attack_eer` in result.json was computed from, just not pre-pooled
by tier.

Usage:
    python3 app/scripts/eer_by_tier.py <scores_file> <min_clips>
"""
import sys
from pathlib import Path

import numpy as np


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> float:
    """Equal error rate (%) between bonafide (target) and spoof (nontarget) scores."""
    thresholds = np.concatenate([target_scores, nontarget_scores])
    thresholds = np.sort(np.unique(thresholds))
    frr = np.array([(target_scores < t).mean() for t in thresholds])
    far = np.array([(nontarget_scores >= t).mean() for t in thresholds])
    idx = np.nanargmin(np.abs(frr - far))
    return float((frr[idx] + far[idx]) / 2.0) * 100.0


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    scores_path = Path(sys.argv[1])
    min_clips = int(sys.argv[2])

    audio_ids, attack_ids, keys, scores = [], [], [], []
    with open(scores_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            audio_ids.append(parts[0])
            attack_ids.append(parts[1])
            keys.append(parts[2])
            scores.append(float(parts[3]))

    attack_ids = np.array(attack_ids)
    keys = np.array(keys)
    scores = np.array(scores)

    bonafide_scores = scores[keys == "bonafide"]
    print(f"Loaded {len(scores)} rows, {len(bonafide_scores)} bonafide.")

    for tier_suffix in ["-psw1", "-psw2", "-psw3"]:
        mask = np.array([a.endswith(tier_suffix) for a in attack_ids])
        tier_spoof_scores = scores[mask]
        n = len(tier_spoof_scores)
        if n < min_clips:
            print(f"{tier_suffix}: n={n} < {min_clips}, SKIPPED (low confidence)")
            continue
        eer = compute_eer(bonafide_scores, tier_spoof_scores)
        print(f"{tier_suffix}: n={n}, EER={eer:.3f}%")


if __name__ == "__main__":
    main()
