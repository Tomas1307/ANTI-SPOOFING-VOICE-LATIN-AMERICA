"""
Compute the three statistics main.tex's Technical Validation pass is missing.

WHY
---
A round of advisor comments on the MARSA Data Descriptor asked for three
numbers that do not exist anywhere in the repository or the wiki, and cannot
be derived from published totals: per-accent bonafide utterance counts (only
per-accent speaker counts were ever recorded), the standard deviation of
utterance duration per corpus tier, and the standard deviation of duration
and target-text word count per full-spoof system. This script computes all
three in one pass, from the same sources the corresponding averages were
computed from, so the numbers are consistent with what the paper already
states.

Bonafide durations are read from FLAC/WAV/MP3 file headers (soundfile.info,
no decode). Full-spoof and partial-spoof durations and word counts are read
from each pipeline's own generation metadata, not from audio headers, since
that is where the already-published averages came from.

FIELD-NAME ROBUSTNESS
----------------------
An earlier version of this script guessed at the key names in
generation_metadata.json and matched nothing (confirmed on ml-server03,
2026-09-08: every one of 215,045 full-spoof records came back unmatched).
The real schema, verified directly against one record per attack, is
``{speaker_id, text_id, text, audio_path, duration_seconds,
generation_time_seconds, rtf, split}`` (Qwen3-TTS and OmniVoice also carry
a boolean ``skipped_existing``). There is no explicit word-count field in
any of the six pipelines; word count is derived from ``text`` by
whitespace tokenization. DURATION_KEYS and the word-count fallback below
reflect this confirmed schema; the multi-candidate lookup is kept as a
defensive fallback for any future pipeline with different field names, and
still reports unmatched records rather than silently skipping them.

USAGE
-----
    cd ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA
    source envs/dfarena_env/bin/activate
    python -m app.scripts.compute_paper_statistics
    deactivate
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
from loguru import logger

from app.schemas.paper_statistics_report import (
    PaperStatisticsReport,
    SystemDurationWordStats,
    TierDurationStats,
)

BONAFIDE_DIR = Path("data/bonafide_dataset_by_speaker_v2")
ACCENT_PREFIXES = ("ar", "cl", "co", "mx", "pe", "ve", "es")
FULLSPOOF_SYSTEMS = {
    "Fish-Speech (FishGram)": "data/fishgram_output",
    "Qwen3-TTS": "data/qwen_output",
    "OpenVoice v2": "data/openvoice_output",
    "Chatterbox": "data/chatterbox_output",
    "OuteTTS": "data/outetts_output",
    "OmniVoice": "data/omnivoice_output",
}
# Already-published per-system utterance counts (main.tex, tab:fullspoofdesc),
# computed independently over the same "passed validation" population this
# script targets. Any mismatch means this script is reading the wrong
# population again and its numbers must not be trusted until reconciled.
PUBLISHED_FULLSPOOF_COUNTS = {
    "Fish-Speech (FishGram)": 34197,
    "Qwen3-TTS": 31568,
    "OpenVoice v2": 29796,
    "Chatterbox": 31701,
    "OuteTTS": 25642,
    "OmniVoice": 33743,
}
PARTIAL_SPOOF_ROOT = Path("data/partial_spoof_output")

DURATION_KEYS = (
    "duration_seconds", "duration_s", "duration", "audio_duration_s", "dur", "duration_sec"
)
WORD_COUNT_KEYS = ("word_count", "n_words", "num_words", "target_words", "words")
TEXT_KEYS = ("text",)


def _accent_from_speaker_dir(name: str) -> Optional[str]:
    """Map a speaker directory name to its two-letter accent code.

    Args:
        name: Speaker directory basename, e.g. ``arf_00295``.

    Returns:
        The accent code, or None if the prefix does not match any known
        accent.
    """
    prefix = name[:2].lower()
    return prefix if prefix in ACCENT_PREFIXES else None


def compute_bonafide_stats() -> Tuple[Dict[str, int], TierDurationStats]:
    """Compute per-accent utterance counts and overall duration stats.

    Returns:
        A tuple of (accent -> utterance count, overall bonafide duration
        stats).

    Raises:
        FileNotFoundError: If the bonafide directory does not exist.
    """
    if not BONAFIDE_DIR.is_dir():
        raise FileNotFoundError(f"Bonafide directory not found: {BONAFIDE_DIR}")

    accent_counts: Dict[str, int] = {code: 0 for code in ACCENT_PREFIXES}
    durations: List[float] = []
    unmatched_accent = 0

    speaker_dirs = sorted(p for p in BONAFIDE_DIR.iterdir() if p.is_dir())
    logger.info(f"Scanning {len(speaker_dirs)} bonafide speaker directories...")

    for index, speaker_dir in enumerate(speaker_dirs, start=1):
        accent = _accent_from_speaker_dir(speaker_dir.name)
        audio_files = [
            f
            for ext in ("*.wav", "*.flac", "*.mp3")
            for f in speaker_dir.rglob(ext)
        ]
        if accent is None:
            unmatched_accent += len(audio_files)
        else:
            accent_counts[accent] += len(audio_files)

        for audio_file in audio_files:
            try:
                info = sf.info(str(audio_file))
                durations.append(info.frames / info.samplerate)
            except Exception as error:
                logger.warning(f"Could not read {audio_file}: {error}")

        if index % 200 == 0:
            logger.debug(f"  {index}/{len(speaker_dirs)} speakers scanned")

    if unmatched_accent:
        logger.warning(
            f"{unmatched_accent} bonafide files had an unrecognised accent "
            "prefix and are excluded from the per-accent counts."
        )

    stats = TierDurationStats(
        tier="bonafide",
        count=len(durations),
        mean_duration_s=round(statistics.mean(durations), 3) if durations else 0.0,
        std_duration_s=round(statistics.stdev(durations), 3) if len(durations) > 1 else 0.0,
    )
    logger.info(f"Bonafide: {stats.count} files, mean {stats.mean_duration_s}s, std {stats.std_duration_s}s")
    return accent_counts, stats


def _find_field(record: dict, candidates: Tuple[str, ...]) -> Optional[float]:
    """Return the first matching field's value from a record.

    Args:
        record: A single sample's metadata dict.
        candidates: Field names to try, in priority order.

    Returns:
        The value of the first matching field, or None if none match.
    """
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _word_count(record: dict) -> Optional[float]:
    """Return a record's target-text word count.

    No pipeline's generation_metadata.json carries an explicit word-count
    field; every record instead carries the full target sentence under
    ``text``. Word count is the whitespace-token count of that sentence,
    the same convention used elsewhere in this repo for reporting average
    words per system.

    Args:
        record: A single sample's metadata dict.

    Returns:
        The word count, or None if neither an explicit field nor a usable
        ``text`` field is present.
    """
    explicit = _find_field(record, WORD_COUNT_KEYS)
    if explicit is not None:
        return explicit
    text = _find_field(record, TEXT_KEYS)
    if isinstance(text, str) and text.strip():
        return float(len(text.split()))
    return None


def compute_fullspoof_stats() -> Tuple[List[SystemDurationWordStats], TierDurationStats]:
    """Compute per-system and overall full-spoof duration/word statistics.

    Reads ``validated_samples.json``, not ``generation_metadata.json``.
    The latter logs every generation attempt, including ones later
    rejected and regenerated by ``QualityValidator`` (see each attack's
    ``step_04_validate_quality.py``); the former holds only the samples
    that passed the WER/CER gate, the same population the already-published
    per-system utterance counts (for example FishGram's 34,197) were
    computed over. Each validated entry is a copy of its generation
    record plus ``wer``/``cer``/``nisqa_mos``/``speaker_similarity``, so
    it carries the same ``duration_seconds``/``text`` fields.

    Returns:
        A tuple of (per-system stats, overall full-spoof duration stats).
    """
    per_system: List[SystemDurationWordStats] = []
    all_durations: List[float] = []
    total_unmatched = 0

    for system_name, output_dir in FULLSPOOF_SYSTEMS.items():
        metadata_path = Path(output_dir) / "validated_samples.json"
        if not metadata_path.exists():
            logger.warning(f"Missing validated_samples.json for {system_name}: {metadata_path}")
            continue

        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else list(data.values())

        durations: List[float] = []
        word_counts: List[float] = []
        unmatched = 0

        for record in records:
            if not isinstance(record, dict):
                continue
            duration = _find_field(record, DURATION_KEYS)
            words = _word_count(record)
            if duration is None or words is None:
                unmatched += 1
                continue
            durations.append(float(duration))
            word_counts.append(float(words))

        if unmatched:
            logger.warning(f"{system_name}: {unmatched} records missing duration/word fields")
            total_unmatched += unmatched

        if not durations:
            logger.error(f"{system_name}: no usable records; check DURATION_KEYS/WORD_COUNT_KEYS")
            continue

        published_count = PUBLISHED_FULLSPOOF_COUNTS.get(system_name)
        if published_count is not None and len(durations) != published_count:
            logger.error(
                f"{system_name}: computed n={len(durations)} does not match "
                f"the published count {published_count}. Do NOT trust these "
                "stats until this is reconciled; validated_samples.json may "
                "have been regenerated since the published count was taken, "
                "or this script is still reading the wrong population."
            )

        per_system.append(
            SystemDurationWordStats(
                system=system_name,
                count=len(durations),
                mean_duration_s=round(statistics.mean(durations), 3),
                std_duration_s=round(statistics.stdev(durations), 3) if len(durations) > 1 else 0.0,
                mean_words=round(statistics.mean(word_counts), 2),
                std_words=round(statistics.stdev(word_counts), 2) if len(word_counts) > 1 else 0.0,
            )
        )
        all_durations.extend(durations)
        logger.info(
            f"{system_name}: n={len(durations)} dur={statistics.mean(durations):.2f}"
            f"({statistics.stdev(durations) if len(durations) > 1 else 0:.2f}) "
            f"words={statistics.mean(word_counts):.1f}"
            f"({statistics.stdev(word_counts) if len(word_counts) > 1 else 0:.1f})"
        )

    if total_unmatched:
        logger.warning(
            f"{total_unmatched} full-spoof records total had no matching duration/word "
            "field; check DURATION_KEYS and WORD_COUNT_KEYS against the real schema."
        )

    overall = TierDurationStats(
        tier="full_spoof",
        count=len(all_durations),
        mean_duration_s=round(statistics.mean(all_durations), 3) if all_durations else 0.0,
        std_duration_s=round(statistics.stdev(all_durations), 3) if len(all_durations) > 1 else 0.0,
    )
    return per_system, overall


def compute_partial_spoof_stats() -> TierDurationStats:
    """Compute overall partial-spoof duration statistics.

    Returns:
        Duration statistics across every partial-spoof metadata file found.
    """
    durations: List[float] = []
    metadata_files = sorted(PARTIAL_SPOOF_ROOT.glob("*/*/LA/partial_spoof_metadata.json"))
    logger.info(f"Found {len(metadata_files)} partial-spoof metadata files")

    for path in metadata_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for sample in data.values():
            duration = sample.get("total_duration_s")
            if duration is not None:
                durations.append(float(duration))

    stats = TierDurationStats(
        tier="partial_spoof",
        count=len(durations),
        mean_duration_s=round(statistics.mean(durations), 3) if durations else 0.0,
        std_duration_s=round(statistics.stdev(durations), 3) if len(durations) > 1 else 0.0,
    )
    logger.info(
        f"Partial spoof: {stats.count} clips, mean {stats.mean_duration_s}s, "
        f"std {stats.std_duration_s}s"
    )
    return stats


if __name__ == "__main__":
    accent_counts, bonafide_stats = compute_bonafide_stats()
    system_stats, fullspoof_stats = compute_fullspoof_stats()
    partial_stats = compute_partial_spoof_stats()

    report = PaperStatisticsReport(
        accent_utterance_counts=accent_counts,
        tier_duration_stats=[bonafide_stats, fullspoof_stats, partial_stats],
        system_stats=system_stats,
    )

    output_path = Path("data/paper_statistics_report.json")
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("SUMMARY (paste these into the paper tables)")
    logger.info("=" * 70)
    logger.info("Per-accent bonafide utterance counts:")
    for accent, count in accent_counts.items():
        logger.info(f"  {accent}: {count:,}")
    logger.info("Per-tier duration mean (std), seconds:")
    for tier_stat in [bonafide_stats, fullspoof_stats, partial_stats]:
        logger.info(
            f"  {tier_stat.tier}: {tier_stat.mean_duration_s} ({tier_stat.std_duration_s}), "
            f"n={tier_stat.count}"
        )
    logger.info("Per-system duration and word stats:")
    for stat in system_stats:
        logger.info(
            f"  {stat.system}: dur {stat.mean_duration_s} ({stat.std_duration_s}), "
            f"words {stat.mean_words} ({stat.std_words})"
        )
    logger.info(f"Full report written: {output_path}")
