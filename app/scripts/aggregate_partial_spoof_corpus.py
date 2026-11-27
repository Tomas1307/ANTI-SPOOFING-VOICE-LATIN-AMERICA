"""
Concatenate per-pipeline partial-spoof CSVs into the corpus master tables.

WHY THIS SCRIPT EXISTS SEPARATELY FROM PartialSpoofOrchestrator.aggregate
--------------------------------------------------------------------------
``PartialSpoofOrchestrator.aggregate`` does the identical, purely
CSV-level concatenation this script does, and its own docstring promises
it needs "no GPU, runs in any env". That promise does not hold in
practice: importing ``app.runner.partial_spoof_orchestrator`` also
imports ``app.runner`` (whose ``__init__.py`` eagerly imports
``ParallelLauncher``, which needs ``loguru``) and the orchestrator module
itself eagerly imports ``PartialSpoofPipeline`` (which needs ``librosa``
and the full TTS cloning stack) for its other modes. Every venv on
ml-server03 is deliberately narrow (one per attack, plus a lean
``dfarena_env`` for detector work), so no single environment satisfies
that combined import graph. This script duplicates only the aggregation
logic, with zero project-internal imports, so it runs in any of them.

The attack list below mirrors ``PartialSpoofSettings.ATTACK_WEIGHTS``
(recorded in docs/thesis-wiki/log.md, 2026-05-20 entry: 40% OmniVoice,
20% Qwen3-TTS, 10% each FishGram/OpenVoice/Chatterbox/OuteTTS) rather
than importing it, for the same reason: importing that settings module
would re-trigger the same eager package import.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    python -m app.scripts.aggregate_partial_spoof_corpus
    deactivate
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger

from app.schemas.partial_spoof_corpus_summary import PartialSpoofCorpusSummary

CORPUS_ROOT = Path("data/partial_spoof_output")
CORPUS_SAMPLES_CSV = CORPUS_ROOT / "corpus_samples.csv"
CORPUS_SPOOFED_WORDS_CSV = CORPUS_ROOT / "corpus_spoofed_words.csv"
CORPUS_SUMMARY_JSON = CORPUS_ROOT / "corpus_summary.json"
PARTITIONS: Tuple[str, str] = ("not_jittered", "jittered")
ATTACK_WEIGHTS_TARGET: Dict[str, float] = {
    "omnivoice": 0.40,
    "qwen": 0.20,
    "fishgram": 0.10,
    "openvoice": 0.10,
    "chatterbox": 0.10,
    "outetts": 0.10,
}
ATTACKS: List[str] = list(ATTACK_WEIGHTS_TARGET.keys())


def _find_source_csvs(filename: str) -> List[Tuple[str, str, Path]]:
    """Locate every per-pipeline CSV with the given filename.

    Args:
        filename: Either 'samples.csv' or 'spoofed_words.csv'.

    Returns:
        (attack, partition, path) triples for every cell that has the
        file, in a fixed attack/partition order.
    """
    found: List[Tuple[str, str, Path]] = []
    for attack in ATTACKS:
        for partition in PARTITIONS:
            path = CORPUS_ROOT / attack / partition / filename
            if path.exists():
                found.append((attack, partition, path))
    return found


def _concatenate_csv(
    sources: List[Tuple[str, str, Path]], destination: Path
) -> int:
    """Concatenate a list of CSVs sharing a header into one file.

    Args:
        sources: (attack, partition, path) triples to concatenate.
        destination: Output path.

    Returns:
        Number of data rows written.
    """
    if not sources:
        logger.warning(
            f"No source CSVs found for {destination.name}; "
            "writing an empty file with no header."
        )
        destination.write_text("", encoding="utf-8")
        return 0

    canonical_header: List[str] = []
    for _, _, path in sources:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
            if header:
                canonical_header = header
                break

    if not canonical_header:
        logger.warning(
            f"All source CSVs for {destination.name} are empty; "
            "writing an empty file with no header."
        )
        destination.write_text("", encoding="utf-8")
        return 0

    total_rows = 0
    with open(destination, "w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(
            out_handle, fieldnames=canonical_header, quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        for attack, partition, path in sources:
            with open(path, "r", encoding="utf-8", newline="") as in_handle:
                for row in csv.DictReader(in_handle):
                    writer.writerow({field: row.get(field, "") for field in canonical_header})
                    total_rows += 1
            logger.debug(f"  + {attack}/{partition}: rows so far={total_rows}")
    return total_rows


def _build_summary(
    samples_count: int,
    words_count: int,
    samples_paths: List[Tuple[str, str, Path]],
) -> PartialSpoofCorpusSummary:
    """Compute marginal counts over the aggregated samples.

    Args:
        samples_count: Total rows in corpus_samples.csv.
        words_count: Total rows in corpus_spoofed_words.csv.
        samples_paths: Source paths that were aggregated.

    Returns:
        The populated summary.
    """
    per_cell: Dict[str, Dict[str, int]] = {a: {p: 0 for p in PARTITIONS} for a in ATTACKS}
    per_attack_total = {a: 0 for a in ATTACKS}
    per_partition_total = {p: 0 for p in PARTITIONS}
    per_quality = {"high": 0, "medium": 0, "low": 0}

    for attack, partition, path in samples_paths:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                per_cell[attack][partition] += 1
                per_attack_total[attack] += 1
                per_partition_total[partition] += 1
                flag = row.get("quality_flag", "")
                if flag in per_quality:
                    per_quality[flag] += 1

    actual_weights: Dict[str, float] = {}
    if samples_count > 0:
        for attack, count in per_attack_total.items():
            actual_weights[attack] = round(count / samples_count, 4)

    return PartialSpoofCorpusSummary(
        generated_at=datetime.now(timezone.utc).isoformat(),
        samples_total=samples_count,
        spoofed_words_total=words_count,
        per_attack_total=per_attack_total,
        per_partition_total=per_partition_total,
        per_cell=per_cell,
        per_quality_flag=per_quality,
        attack_weights_target=ATTACK_WEIGHTS_TARGET,
        attack_weights_actual=actual_weights,
    )


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Aggregating partial-spoof corpus CSVs (lightweight, no GPU deps)")
    logger.info("=" * 70)

    samples_sources = _find_source_csvs("samples.csv")
    words_sources = _find_source_csvs("spoofed_words.csv")

    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    samples_rows = _concatenate_csv(samples_sources, CORPUS_SAMPLES_CSV)
    words_rows = _concatenate_csv(words_sources, CORPUS_SPOOFED_WORDS_CSV)
    summary = _build_summary(samples_rows, words_rows, samples_sources)

    CORPUS_SUMMARY_JSON.write_text(
        json.dumps(summary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"corpus_samples.csv       : {samples_rows} rows")
    logger.info(f"corpus_spoofed_words.csv : {words_rows} rows")
    logger.info(f"corpus_summary.json      : {CORPUS_SUMMARY_JSON}")
    logger.info("Per-attack totals: " + str(summary.per_attack_total))
    logger.info("Per-quality-flag totals: " + str(summary.per_quality_flag))
