"""
Compute per-attack-system WER/CER/NISQA/SIM statistics for the
partial-spoof tier.

WHY
---
main.tex already reports full-spoof quality metrics broken down by TTS
system (tab:fullspoof) and partial-spoof quality metrics broken down by
replacement tier (tab:pstierquality, compute_partial_spoof_quality_stats.py),
but partial-spoof has never been broken down by TTS system the way
full-spoof has. This script closes that gap, using the same source file
and the same metric set as the per-tier script, grouped by the
``attack`` column instead of ``tier``, so the new table is directly
comparable to tab:fullspoof.

SOURCE
------
``data/partial_spoof_output/corpus_samples.csv``, written by
``OutputFormatter`` and concatenated across all six attacks by
``PartialSpoofOrchestrator.aggregate`` (or the lightweight
``aggregate_partial_spoof_corpus`` script). The ``attack`` column holds
the lowercase attack slug (omnivoice, qwen, fishgram, openvoice,
chatterbox, outetts).

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    python -m app.scripts.compute_partial_spoof_quality_by_system
    deactivate
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from app.schemas.partial_spoof_system_quality_report import (
    PartialSpoofSystemQualityReport,
    SystemQualityStats,
)

CORPUS_SAMPLES_CSV = Path("data/partial_spoof_output/corpus_samples.csv")
SYSTEM_DISPLAY_NAMES: Dict[str, str] = {
    "omnivoice": "OmniVoice",
    "qwen3tts": "Qwen3-TTS",
    "chatterbox": "Chatterbox",
    "fishgram": "Fish-Speech (FishGram)",
    "outetts": "OuteTTS",
    "openvoice": "OpenVoice v2",
}
# NOTE: partial-spoof's corpus_samples.csv 'attack' column still carries
# 'qwen3tts' for Qwen, unlike the ASVspoof-style protocol files, which
# were normalized to 'qwen' corpus-wide (normalize_attack_ids.py,
# app/scripts, and docs/thesis-wiki/decisions/decision-log.md,
# 2026-08-25 entry). Confirmed directly against the real CSV on
# ml-server03 (2026-09-13) rather than assumed from the protocol-level
# fix. This is a real, separate inconsistency worth closing at the
# source at some point; this script works around it rather than fixes it.


def _parse_float(raw: str) -> Optional[float]:
    """Parse a CSV field into a float, tolerating blanks and junk.

    Args:
        raw: Raw CSV cell value.

    Returns:
        The parsed float, or None if the cell is blank or not numeric.
    """
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _mean_std(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    """Compute a rounded mean and sample standard deviation.

    Args:
        values: Numeric values to summarise.

    Returns:
        A (mean, std) tuple, both None if ``values`` is empty and std
        None if there is only one value.
    """
    if not values:
        return None, None
    mean = round(statistics.mean(values), 4)
    std = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
    return mean, std


def _load_sample_metrics() -> tuple[Dict[str, Dict[str, List[float]]], int]:
    """Load WER/CER/NISQA/SIM per attack system from corpus_samples.csv.

    Returns:
        A tuple of (attack slug -> metric name -> values, unmatched row count).

    Raises:
        FileNotFoundError: If the corpus-level samples CSV is missing.
    """
    if not CORPUS_SAMPLES_CSV.exists():
        raise FileNotFoundError(
            f"{CORPUS_SAMPLES_CSV} not found; run "
            "`python -m app.scripts.aggregate_partial_spoof_corpus` first."
        )

    metrics: Dict[str, Dict[str, List[float]]] = {
        slug: {"wer": [], "cer": [], "nisqa": [], "sim": []}
        for slug in SYSTEM_DISPLAY_NAMES
    }
    unmatched = 0

    with open(CORPUS_SAMPLES_CSV, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            attack = row.get("attack", "")
            if attack not in metrics:
                unmatched += 1
                continue

            wer = _parse_float(row.get("wer", ""))
            cer = _parse_float(row.get("cer", ""))
            nisqa = _parse_float(row.get("nisqa", ""))
            sim = _parse_float(row.get("ecapa_sim_final", ""))

            if wer is None and cer is None and nisqa is None and sim is None:
                unmatched += 1
                continue

            if wer is not None:
                metrics[attack]["wer"].append(wer)
            if cer is not None:
                metrics[attack]["cer"].append(cer)
            if nisqa is not None:
                metrics[attack]["nisqa"].append(nisqa)
            if sim is not None:
                metrics[attack]["sim"].append(sim)

    return metrics, unmatched


def _build_system_stats(
    system: str, count: int, sample_metrics: Dict[str, List[float]]
) -> SystemQualityStats:
    """Assemble one system's SystemQualityStats from its raw value lists.

    Args:
        system: Display name for the attack system, or "ALL".
        count: Spliced samples contributing to the statistics.
        sample_metrics: wer/cer/nisqa/sim value lists for this system.

    Returns:
        The populated SystemQualityStats.
    """
    mean_wer, std_wer = _mean_std(sample_metrics["wer"])
    mean_cer, std_cer = _mean_std(sample_metrics["cer"])
    mean_nisqa, std_nisqa = _mean_std(sample_metrics["nisqa"])
    mean_sim, std_sim = _mean_std(sample_metrics["sim"])

    return SystemQualityStats(
        system=system,
        count=count,
        mean_wer=mean_wer,
        std_wer=std_wer,
        mean_cer=mean_cer,
        std_cer=std_cer,
        mean_nisqa=mean_nisqa,
        std_nisqa=std_nisqa,
        mean_sim=mean_sim,
        std_sim=std_sim,
    )


if __name__ == "__main__":
    sample_metrics_by_system, unmatched_samples = _load_sample_metrics()

    system_stats: List[SystemQualityStats] = []
    overall_sample: Dict[str, List[float]] = defaultdict(list)
    overall_count = 0

    for slug, display_name in SYSTEM_DISPLAY_NAMES.items():
        sm = sample_metrics_by_system[slug]
        count = len(sm["wer"]) or len(sm["cer"]) or len(sm["nisqa"]) or len(sm["sim"])
        system_stats.append(_build_system_stats(display_name, count, sm))
        for key, values in sm.items():
            overall_sample[key].extend(values)
        overall_count += count

    system_stats.append(_build_system_stats("ALL", overall_count, overall_sample))

    report = PartialSpoofSystemQualityReport(
        system_stats=system_stats,
        unmatched_samples=unmatched_samples,
    )

    output_path = Path("data/partial_spoof_system_quality_report.json")
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("SUMMARY (paste these into the paper)")
    logger.info("=" * 70)
    for stat in system_stats:
        logger.info(
            f"{stat.system}: n={stat.count} "
            f"WER={stat.mean_wer}({stat.std_wer}) "
            f"CER={stat.mean_cer}({stat.std_cer}) "
            f"NISQA={stat.mean_nisqa}({stat.std_nisqa}) "
            f"SIM={stat.mean_sim}({stat.std_sim})"
        )
    logger.info(f"Unmatched: {unmatched_samples} sample rows.")
    logger.info(f"Full report written: {output_path}")
