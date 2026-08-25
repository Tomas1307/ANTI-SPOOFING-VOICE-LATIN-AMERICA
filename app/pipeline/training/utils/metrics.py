"""
Detection-error metrics for countermeasure scores.

The equal error rate follows the ASVspoof convention: scores are bonafide
likelihoods, so a higher score means more genuine, bonafide clips are the
target class and spoof clips are the non-target class.
"""
from typing import Dict, List, Tuple

import numpy as np


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the detection error trade-off curve.

    Args:
        target_scores: Scores of bonafide clips.
        nontarget_scores: Scores of spoof clips.

    Returns:
        A tuple of (false_rejection_rates, false_acceptance_rates,
        thresholds), each of length ``len(target) + len(nontarget) + 1``.

    Raises:
        ValueError: If either class is empty.
    """
    if target_scores.size == 0 or nontarget_scores.size == 0:
        raise ValueError(
            "DET curve needs both classes: "
            f"{target_scores.size} target, {nontarget_scores.size} non-target"
        )

    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, all_scores.size + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[order[0]] - 0.001), all_scores[order])
    )
    return frr, far, thresholds


def compute_eer(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> Tuple[float, float]:
    """Compute the equal error rate.

    Args:
        target_scores: Scores of bonafide clips.
        nontarget_scores: Scores of spoof clips.

    Returns:
        A tuple of (eer_percent, threshold) where the rate is expressed as a
        percentage.

    Raises:
        ValueError: If either class is empty.
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    index = np.nanargmin(np.abs(frr - far))
    eer = float((frr[index] + far[index]) / 2.0) * 100.0
    return eer, float(thresholds[index])


def compute_per_attack_eer(
    scores: np.ndarray, labels: np.ndarray, attack_ids: List[str]
) -> Dict[str, float]:
    """Compute one equal error rate per attack system.

    Each attack is scored against the full bonafide pool of the split, which
    is the standard ASVspoof per-attack decomposition. Attacks whose clips are
    absent from the supplied arrays are omitted from the result.

    Args:
        scores: Countermeasure scores, one per clip.
        labels: Integer labels, 1 for bonafide and 0 for spoof.
        attack_ids: Attack identifier per clip; bonafide clips carry a
            placeholder that is ignored.

    Returns:
        Mapping of attack identifier to equal error rate, as a percentage.
    """
    bonafide = scores[labels == 1]
    if bonafide.size == 0:
        return {}

    attacks = np.asarray(attack_ids)
    per_attack: Dict[str, float] = {}
    for attack in sorted(set(attacks[labels == 0].tolist())):
        spoof = scores[(labels == 0) & (attacks == attack)]
        if spoof.size == 0:
            continue
        eer, _threshold = compute_eer(bonafide, spoof)
        per_attack[attack] = eer
    return per_attack
