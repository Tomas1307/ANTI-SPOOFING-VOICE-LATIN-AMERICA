"""
Compute corpus duration and text-length statistics for the paper.

Covers the three base tiers (bonafide, full-spoof, partial-spoof):
  - Bonafide: audio duration read directly from file headers (no decode);
    word counts from the cached Parakeet transcripts.
  - Full-spoof: duration and target-text word counts read from each of the
    six attack pipelines' validated_samples.json (falling back to
    generation_metadata.json), both of which already carry these fields
    per sample -- no re-computation needed.
  - Partial-spoof: duration read from the released samples.csv files
    (total_duration_s column), one per (attack, partition) cell.

Augmentation tiers are not scanned: augmentation (RIR/noise addition,
codec re-encoding) does not change utterance duration, so augmented-tier
hours can be derived analytically as base-tier hours times the
augmentation factor (x2, x3, x5, x10) rather than rescanned from disk.

Deliberately imports nothing from app.pipeline.*: every attack pipeline's
__init__.py eagerly imports its pipeline_facade, which cascades into its
steps and cloners (e.g. Chatterbox's cloner unconditionally imports the
`perth` watermarking package). Those dependencies live in each pipeline's
own isolated venv and are not all present together in any single venv, so
this script only ever touches plain files (JSON/CSV/audio headers) and
uses the pipelines' documented default OUTPUT_DIR paths as string
literals -- no pipeline package is imported.

Usage on ml-server03 (CPU-only; soundfile header reads, no GPU; any venv
with soundfile/loguru/tqdm/pydantic is sufficient, e.g. fishgram_env):
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
    python -m app.scripts.compute_corpus_duration_stats
    deactivate
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.schemas.corpus_duration_report import CorpusDurationReport
from app.schemas.corpus_tier_duration_stat import TierDurationStat
from app.schemas.fullspoof_system_description_stat import SystemDescriptionStat

CACHED_TRANSCRIPTS_FILENAME = "bonafide_transcripts_full.json"
DEFAULT_BONAFIDE_DIR = Path("data/bonafide_dataset_by_speaker_v2")
DEFAULT_MANIFEST_DIR = Path("data/manifests")
DEFAULT_PARTIAL_SPOOF_ROOT = Path("data/partial_spoof_output")

# (OUTPUT_DIR default from each attack pipeline's settings.py, display name
# matching the paper's system tables)
FULLSPOOF_SYSTEMS: List[Tuple[Path, str]] = [
    (Path("data/fishgram_output"), "Fish-Speech (FishGram)"),
    (Path("data/qwen_output"), "Qwen3-TTS"),
    (Path("data/openvoice_output"), "OpenVoice v2"),
    (Path("data/chatterbox_output"), "Chatterbox"),
    (Path("data/outetts_output"), "OuteTTS"),
    (Path("data/omnivoice_output"), "OmniVoice"),
]

PARTIAL_SPOOF_SAMPLES_GLOB = "*/*/samples.csv"


class CorpusDurationStatsComputer:
    """Computes duration and text-length statistics for the base corpus.

    Attributes:
        bonafide_dir: Root directory containing HABLA v2 bonafide speakers.
        cached_transcripts_path: Path to the cached full-corpus Parakeet
            transcripts JSON (for bonafide word counts).
        partial_spoof_root: Root directory under which per-(attack,
            partition) samples.csv files are discovered.
        report_output_path: Where the resulting report JSON is written.
    """

    def __init__(
        self,
        bonafide_dir: Optional[Path] = None,
        cached_transcripts_path: Optional[Path] = None,
        partial_spoof_root: Optional[Path] = None,
        report_output_path: Optional[Path] = None,
    ) -> None:
        """Initialize the computer with sensible default paths.

        Args:
            bonafide_dir: Override for the bonafide speaker root.
            cached_transcripts_path: Override for the cached transcripts JSON.
            partial_spoof_root: Override for the partial-spoof output root
                under which samples.csv files are discovered.
            report_output_path: Override for the output report JSON location.
        """
        self.bonafide_dir = bonafide_dir or DEFAULT_BONAFIDE_DIR
        self.cached_transcripts_path = (
            cached_transcripts_path or DEFAULT_MANIFEST_DIR / CACHED_TRANSCRIPTS_FILENAME
        )
        self.partial_spoof_root = partial_spoof_root or DEFAULT_PARTIAL_SPOOF_ROOT
        self.report_output_path = (
            report_output_path or DEFAULT_MANIFEST_DIR / "corpus_duration_report.json"
        )

    def run(self) -> CorpusDurationReport:
        """Compute all tier statistics and persist the report.

        Returns:
            The computed CorpusDurationReport.
        """
        bonafide_stat = self._compute_bonafide_stats()
        fullspoof_systems, fullspoof_tier = self._compute_fullspoof_stats()
        partial_spoof_stat = self._compute_partial_spoof_stats()

        per_tier = [bonafide_stat, fullspoof_tier, partial_spoof_stat]
        base_total_hours = sum(t.total_hours for t in per_tier)

        report = CorpusDurationReport(
            per_tier=per_tier,
            per_fullspoof_system=fullspoof_systems,
            base_corpus_total_hours=base_total_hours,
            notes=[
                "Augmentation tiers (x2, x3, x5, x10) are not scanned: "
                "augmentation preserves utterance duration, so augmented-tier "
                "hours = base-tier hours x factor.",
                "Bonafide avg_word_count is computed only over utterances "
                "present in the cached Parakeet transcripts (utterances "
                "below the four-word minimum are absent), which biases it "
                "slightly upward relative to the full bonafide set.",
            ],
        )
        self._save_report(report)
        self._log_summary(report)
        return report

    def _compute_bonafide_stats(self) -> TierDurationStat:
        """Compute duration (all files) and word-count (transcribed subset).

        Returns:
            TierDurationStat for the bonafide tier.
        """
        audio_paths = sorted(
            p for ext in ("*.wav", "*.flac", "*.mp3")
            for p in self.bonafide_dir.rglob(ext)
        )
        logger.info(f"Bonafide: reading duration headers for {len(audio_paths)} files...")

        total_seconds = 0.0
        read_count = 0
        for path in tqdm(audio_paths, desc="Bonafide durations"):
            try:
                total_seconds += sf.info(path).duration
                read_count += 1
            except Exception as e:
                logger.warning(f"Could not read duration for {path}: {e}")

        avg_word_count = None
        if self.cached_transcripts_path.exists():
            with open(self.cached_transcripts_path, "r", encoding="utf-8") as f:
                transcripts = json.load(f)
            word_counts = [entry["word_count"] for entry in transcripts.values()]
            if word_counts:
                avg_word_count = sum(word_counts) / len(word_counts)
        else:
            logger.warning(
                f"Cached transcripts not found at {self.cached_transcripts_path}; "
                "bonafide avg_word_count will be null."
            )

        return TierDurationStat(
            tier="Bonafide",
            utterance_count=read_count,
            total_hours=total_seconds / 3600.0,
            avg_duration_seconds=(total_seconds / read_count) if read_count else 0.0,
            avg_word_count=avg_word_count,
        )

    def _compute_fullspoof_stats(
        self,
    ) -> Tuple[List[SystemDescriptionStat], TierDurationStat]:
        """Compute per-system and aggregate full-spoof statistics.

        Reads each system's validated_samples.json (or generation_metadata.json
        fallback), which already carries duration_seconds and text per sample.

        Returns:
            Tuple of (per-system stats list, aggregate TierDurationStat).
        """
        per_system: List[SystemDescriptionStat] = []
        all_durations: List[float] = []
        all_word_counts: List[int] = []

        for output_dir, display_name in FULLSPOOF_SYSTEMS:
            samples = self._load_fullspoof_samples(output_dir)
            if samples is None:
                logger.warning(f"{display_name}: no sample metadata found under {output_dir}, skipping.")
                continue

            durations = [s["duration_seconds"] for s in samples.values() if "duration_seconds" in s]
            word_counts = [
                len(str(s["text"]).split()) for s in samples.values() if "text" in s
            ]
            all_durations.extend(durations)
            all_word_counts.extend(word_counts)

            per_system.append(
                SystemDescriptionStat(
                    system=display_name,
                    utterance_count=len(samples),
                    avg_duration_seconds=(sum(durations) / len(durations)) if durations else 0.0,
                    avg_word_count=(sum(word_counts) / len(word_counts)) if word_counts else 0.0,
                    total_hours=sum(durations) / 3600.0,
                )
            )

        tier_stat = TierDurationStat(
            tier="Full spoof",
            utterance_count=len(all_durations),
            total_hours=sum(all_durations) / 3600.0,
            avg_duration_seconds=(sum(all_durations) / len(all_durations)) if all_durations else 0.0,
            avg_word_count=(sum(all_word_counts) / len(all_word_counts)) if all_word_counts else None,
        )
        return per_system, tier_stat

    def _load_fullspoof_samples(self, output_dir: Path) -> Optional[dict]:
        """Load a full-spoof system's per-sample metadata dict.

        Args:
            output_dir: The attack pipeline's default OUTPUT_DIR.

        Returns:
            The sample_id -> sample_data dict, or None if neither file exists.
        """
        validated_path = output_dir / "validated_samples.json"
        generation_path = output_dir / "generation_metadata.json"
        path = validated_path if validated_path.exists() else generation_path
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compute_partial_spoof_stats(self) -> TierDurationStat:
        """Compute aggregate partial-spoof duration from released samples.csv files.

        Returns:
            TierDurationStat for the partial-spoof tier.
        """
        csv_paths = sorted(self.partial_spoof_root.glob(PARTIAL_SPOOF_SAMPLES_GLOB))
        logger.info(f"Partial spoof: found {len(csv_paths)} samples.csv files under {self.partial_spoof_root}.")

        durations: List[float] = []
        for csv_path in csv_paths:
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "total_duration_s" in row and row["total_duration_s"]:
                        durations.append(float(row["total_duration_s"]))

        return TierDurationStat(
            tier="Partial spoof",
            utterance_count=len(durations),
            total_hours=sum(durations) / 3600.0,
            avg_duration_seconds=(sum(durations) / len(durations)) if durations else 0.0,
            avg_word_count=None,
        )

    def _save_report(self, report: CorpusDurationReport) -> None:
        """Write the report to disk as JSON.

        Args:
            report: The computed report.
        """
        self.report_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_output_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

    def _log_summary(self, report: CorpusDurationReport) -> None:
        """Log a human-readable summary of the report.

        Args:
            report: The computed report.
        """
        logger.info("=" * 70)
        logger.info("CORPUS DURATION AND DESCRIPTION STATISTICS")
        logger.info("=" * 70)
        for stat in report.per_tier:
            wc = f"{stat.avg_word_count:.1f}" if stat.avg_word_count is not None else "n/a"
            logger.info(
                f"{stat.tier:15s} n={stat.utterance_count:7d}  "
                f"hours={stat.total_hours:8.2f}  "
                f"avg_dur_s={stat.avg_duration_seconds:6.2f}  "
                f"avg_words={wc}"
            )
        logger.info(f"Base corpus total hours: {report.base_corpus_total_hours:.2f}")
        logger.info("-" * 70)
        logger.info("Per full-spoof system:")
        for stat in report.per_fullspoof_system:
            logger.info(
                f"  {stat.system:24s} n={stat.utterance_count:7d}  "
                f"hours={stat.total_hours:7.2f}  "
                f"avg_dur_s={stat.avg_duration_seconds:6.2f}  "
                f"avg_words={stat.avg_word_count:.1f}"
            )
        logger.info(f"Report written to: {self.report_output_path}")
        logger.info("=" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compute corpus duration and text-length statistics."
    )
    parser.add_argument(
        "--bonafide-dir",
        type=Path,
        default=None,
        help="Override the bonafide speaker root (default: "
             "data/bonafide_dataset_by_speaker_v2).",
    )
    parser.add_argument(
        "--partial-spoof-root",
        type=Path,
        default=None,
        help="Root directory under which per-(attack,partition) samples.csv "
             "files are discovered (default: data/partial_spoof_output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output report JSON path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    computer = CorpusDurationStatsComputer(
        bonafide_dir=args.bonafide_dir,
        partial_spoof_root=args.partial_spoof_root,
        report_output_path=args.output,
    )
    computer.run()
