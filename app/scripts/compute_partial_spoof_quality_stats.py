"""
Compute per-tier WER/CER/NISQA/SIM and boundary-margin statistics for the
partial-spoof tier, the last \\PENDING{} data gap in main.tex's Technical
Validation section (the "per-tier WER/NISQA/SIM/boundary-drift statistics"
paragraph).

SOURCE FILES
------------
WER, CER, NISQA and ECAPA speaker-similarity statistics are read from
``data/partial_spoof_output/corpus_samples.csv`` (one row per spliced
sample, written by ``OutputFormatter`` and concatenated across all six
attacks by ``PartialSpoofOrchestrator.aggregate``), grouped by the
``tier`` column (W1, W2, W3).

BOUNDARY-DRIFT DEFINITION
-------------------------
The pipeline does not compute a field literally named "boundary drift".
The closest released per-word measurements are the available silence
margin on each side of the splice seam (``margin_before_ms``,
``margin_after_ms``) and the crossfade duration actually applied within
that margin (``effective_crossfade_ms``), all in
``data/partial_spoof_output/corpus_spoofed_words.csv``. This script
reports those three as the boundary statistic; confirm this matches what
"boundary-drift" is meant to convey in the paper before citing the
numbers, since the wording was written before this script existed and
may need to change to match what is actually measured.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    python -m app.scripts.compute_partial_spoof_quality_stats
    deactivate
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from app.schemas.partial_spoof_quality_report import (
    PartialSpoofQualityReport,
    TierQualityStats,
)

CORPUS_SAMPLES_CSV = Path("data/partial_spoof_output/corpus_samples.csv")
CORPUS_SPOOFED_WORDS_CSV = Path("data/partial_spoof_output/corpus_spoofed_words.csv")
TIERS = ("W1", "W2", "W3")


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
    """Load WER/CER/NISQA/SIM per tier from corpus_samples.csv.

    Returns:
        A tuple of (tier -> metric name -> values, unmatched row count).

    Raises:
        FileNotFoundError: If the corpus-level samples CSV is missing.
    """
    if not CORPUS_SAMPLES_CSV.exists():
        raise FileNotFoundError(
            f"{CORPUS_SAMPLES_CSV} not found; run "
            "`python -m app.runner.partial_spoof_orchestrator --mode aggregate` first."
        )

    metrics: Dict[str, Dict[str, List[float]]] = {
        tier: {"wer": [], "cer": [], "nisqa": [], "sim": []} for tier in TIERS
    }
    unmatched = 0

    with open(CORPUS_SAMPLES_CSV, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tier = row.get("tier", "")
            if tier not in metrics:
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
                metrics[tier]["wer"].append(wer)
            if cer is not None:
                metrics[tier]["cer"].append(cer)
            if nisqa is not None:
                metrics[tier]["nisqa"].append(nisqa)
            if sim is not None:
                metrics[tier]["sim"].append(sim)

    return metrics, unmatched


def _load_boundary_metrics() -> tuple[Dict[str, Dict[str, List[float]]], int]:
    """Load margin/crossfade boundary metrics per tier from spoofed_words.csv.

    Returns:
        A tuple of (tier -> metric name -> values, unmatched row count).

    Raises:
        FileNotFoundError: If the corpus-level spoofed-words CSV is missing.
    """
    if not CORPUS_SPOOFED_WORDS_CSV.exists():
        raise FileNotFoundError(
            f"{CORPUS_SPOOFED_WORDS_CSV} not found; run "
            "`python -m app.runner.partial_spoof_orchestrator --mode aggregate` first."
        )

    metrics: Dict[str, Dict[str, List[float]]] = {
        tier: {"margin_before_ms": [], "margin_after_ms": [], "crossfade_ms": []}
        for tier in TIERS
    }
    unmatched = 0

    with open(CORPUS_SPOOFED_WORDS_CSV, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tier = row.get("tier", "")
            if tier not in metrics:
                unmatched += 1
                continue

            margin_before = _parse_float(row.get("margin_before_ms", ""))
            margin_after = _parse_float(row.get("margin_after_ms", ""))
            crossfade = _parse_float(row.get("effective_crossfade_ms", ""))

            if margin_before is None and margin_after is None and crossfade is None:
                unmatched += 1
                continue

            if margin_before is not None:
                metrics[tier]["margin_before_ms"].append(margin_before)
            if margin_after is not None:
                metrics[tier]["margin_after_ms"].append(margin_after)
            if crossfade is not None:
                metrics[tier]["crossfade_ms"].append(crossfade)

    return metrics, unmatched


def _build_tier_stats(
    tier: str,
    sample_count: int,
    sample_metrics: Dict[str, List[float]],
    boundary_word_count: int,
    boundary_metrics: Dict[str, List[float]],
) -> TierQualityStats:
    """Assemble one tier's TierQualityStats from its raw value lists.

    Args:
        tier: Tier label.
        sample_count: Spliced samples contributing to the WER/CER/NISQA/SIM
            statistics.
        sample_metrics: wer/cer/nisqa/sim value lists for this tier.
        boundary_word_count: Spoofed-word rows contributing to the
            boundary statistics.
        boundary_metrics: margin/crossfade value lists for this tier.

    Returns:
        The populated TierQualityStats.
    """
    mean_wer, std_wer = _mean_std(sample_metrics["wer"])
    mean_cer, std_cer = _mean_std(sample_metrics["cer"])
    mean_nisqa, std_nisqa = _mean_std(sample_metrics["nisqa"])
    mean_sim, std_sim = _mean_std(sample_metrics["sim"])
    mean_mb, std_mb = _mean_std(boundary_metrics["margin_before_ms"])
    mean_ma, std_ma = _mean_std(boundary_metrics["margin_after_ms"])
    mean_cf, std_cf = _mean_std(boundary_metrics["crossfade_ms"])

    return TierQualityStats(
        tier=tier,
        count=sample_count,
        mean_wer=mean_wer,
        std_wer=std_wer,
        mean_cer=mean_cer,
        std_cer=std_cer,
        mean_nisqa=mean_nisqa,
        std_nisqa=std_nisqa,
        mean_sim=mean_sim,
        std_sim=std_sim,
        boundary_word_count=boundary_word_count,
        mean_margin_before_ms=mean_mb,
        std_margin_before_ms=std_mb,
        mean_margin_after_ms=mean_ma,
        std_margin_after_ms=std_ma,
        mean_crossfade_ms=mean_cf,
        std_crossfade_ms=std_cf,
    )


if __name__ == "__main__":
    sample_metrics_by_tier, unmatched_samples = _load_sample_metrics()
    boundary_metrics_by_tier, unmatched_words = _load_boundary_metrics()

    tier_stats: List[TierQualityStats] = []
    overall_sample: Dict[str, List[float]] = defaultdict(list)
    overall_boundary: Dict[str, List[float]] = defaultdict(list)
    overall_sample_count = 0
    overall_boundary_count = 0

    for tier in TIERS:
        sm = sample_metrics_by_tier[tier]
        bm = boundary_metrics_by_tier[tier]
        sample_count = len(sm["wer"]) or len(sm["cer"]) or len(sm["nisqa"]) or len(sm["sim"])
        boundary_count = (
            len(bm["margin_before_ms"]) or len(bm["margin_after_ms"]) or len(bm["crossfade_ms"])
        )
        tier_stats.append(
            _build_tier_stats(tier, sample_count, sm, boundary_count, bm)
        )
        for key, values in sm.items():
            overall_sample[key].extend(values)
        for key, values in bm.items():
            overall_boundary[key].extend(values)
        overall_sample_count += sample_count
        overall_boundary_count += boundary_count

    tier_stats.append(
        _build_tier_stats(
            "ALL", overall_sample_count, overall_sample, overall_boundary_count, overall_boundary
        )
    )

    report = PartialSpoofQualityReport(
        tier_stats=tier_stats,
        unmatched_samples=unmatched_samples,
        unmatched_boundary_words=unmatched_words,
    )

    output_path = Path("data/partial_spoof_quality_report.json")
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("SUMMARY (paste these into the paper)")
    logger.info("=" * 70)
    for stat in tier_stats:
        logger.info(
            f"{stat.tier}: n={stat.count} "
            f"WER={stat.mean_wer}({stat.std_wer}) "
            f"CER={stat.mean_cer}({stat.std_cer}) "
            f"NISQA={stat.mean_nisqa}({stat.std_nisqa}) "
            f"SIM={stat.mean_sim}({stat.std_sim}) | "
            f"boundary n={stat.boundary_word_count} "
            f"margin_before_ms={stat.mean_margin_before_ms}({stat.std_margin_before_ms}) "
            f"margin_after_ms={stat.mean_margin_after_ms}({stat.std_margin_after_ms}) "
            f"crossfade_ms={stat.mean_crossfade_ms}({stat.std_crossfade_ms})"
        )
    logger.info(
        f"Unmatched: {unmatched_samples} sample rows, "
        f"{unmatched_words} boundary-word rows."
    )
    logger.info(f"Full report written: {output_path}")
