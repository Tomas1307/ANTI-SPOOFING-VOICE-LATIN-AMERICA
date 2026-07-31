"""
Build a speaker-disjoint train/dev/eval partition of the MARSA corpus.

MARSA's own per-file split (bonafide: train/val/test subfolders per speaker;
full-spoof and partial-spoof: train/dev/eval assigned per utterance) is
speaker-DEPENDENT -- the same speaker's audio can appear in more than one
split. That is intentional for the released corpus (Usage Notes explicitly
tells users to build their own speaker-disjoint split before training), but
it is unsuitable to feed directly into the augmentation pipeline, which
assumes -- and the anti-spoofing literature requires -- that a detector never
sees a speaker at training time that it will be evaluated on.

This script re-partitions by whole speaker (every file a speaker has, across
all three sources, goes to exactly one of train/dev/eval) and writes the
result as a tree of symlinks in the layout DatasetLoader/AugmentationPipeline
expect:

    data/marsa_speaker_disjoint_partition/
      train/<speaker_id>/bonafide_<speaker_id>_<n>.<ext>
      train/<speaker_id>/spoof_<system>_<speaker_id>_<n>.<ext>
      train/<speaker_id>/spoof_<system>-ps<tier>_<speaker_id>_<n>.<ext>
      dev/...
      eval/...

Sources (deliberately no app.pipeline.* imports -- every attack pipeline's
__init__.py eagerly imports its pipeline_facade, which cascades into
system-specific dependencies like Chatterbox's `perth` watermarker that are
not installed together in any single venv; this script only reads plain
files):
  - Bonafide: data/bonafide_dataset_by_speaker_v2/<speaker>/{train,val,test}/*
  - Full-spoof: data/<system>_output/LA/ASVspoof2019_LA_{train,dev,eval}/
    flac/*.flac, keyed by speaker via the ASVspoof protocol text file
    (glob'd rather than hardcoded, since the .trn/.trl suffix is not
    consistent across the six attack pipelines' own formatters).
  - Partial-spoof: data/partial_spoof_output/corpus_samples.csv
    (spliced_audio_path column), the confirmed authoritative aggregate
    (18,421 rows total, matches the paper's Table exactly).

Usage on ml-server03 (CPU-only, no GPU, symlinks only -- no audio decoded):
    source ~/ANTI-SPOOFING-VOICE-LATIN-AMERICA/envs/fishgram_env/bin/activate
    python -m app.scripts.build_speaker_disjoint_partition
    deactivate
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from app.schemas.partition_split_stat import PartitionSplitStat
from app.schemas.speaker_partition_report import SpeakerPartitionReport

DEFAULT_BONAFIDE_DIR = Path("data/bonafide_dataset_by_speaker_v2")
DEFAULT_PARTIAL_SPOOF_CSV = Path("data/partial_spoof_output/corpus_samples.csv")
DEFAULT_OUTPUT_DIR = Path("data/marsa_speaker_disjoint_partition")

FULLSPOOF_SYSTEMS: List[str] = [
    "fishgram", "qwen", "openvoice", "chatterbox", "outetts", "omnivoice",
]
FULLSPOOF_OUTPUT_DIRS: Dict[str, Path] = {
    system: Path(f"data/{system}_output") for system in FULLSPOOF_SYSTEMS
}

TRAIN_RATIO = 0.80
DEV_RATIO = 0.10
# EVAL_RATIO is the remainder.


class SpeakerDisjointPartitionBuilder:
    """Builds a speaker-disjoint train/dev/eval symlink partition.

    Attributes:
        bonafide_dir: Root directory of HABLA v2 bonafide speakers.
        partial_spoof_csv: Path to the aggregated partial-spoof manifest.
        fullspoof_dirs: Mapping from system name to its output directory.
        output_dir: Where the symlink partition is written.
        seed: Random seed for the speaker-level split.
        dry_run: When True, index and count everything but create no
            directories or symlinks; logs example link names instead.
        include_noisy_partial_spoof: When False (default), only the
            paper's recommended clean partial-spoof subset (WER <= 0.15,
            CER <= 0.10; 15,641 of 18,421 rows) is included, matching
            Usage Notes' training recommendation. Set True to include the
            full unfiltered partial-spoof tier instead.
    """

    CLEAN_WER_MAX = 0.15
    CLEAN_CER_MAX = 0.10

    def __init__(
        self,
        bonafide_dir: Optional[Path] = None,
        partial_spoof_csv: Optional[Path] = None,
        fullspoof_dirs: Optional[Dict[str, Path]] = None,
        output_dir: Optional[Path] = None,
        seed: int = 42,
        dry_run: bool = False,
        include_noisy_partial_spoof: bool = False,
    ) -> None:
        """Initialize the builder.

        Args:
            bonafide_dir: Override for the bonafide speaker root.
            partial_spoof_csv: Override for the partial-spoof manifest CSV.
            fullspoof_dirs: Override mapping of system name to output dir.
            output_dir: Override for the output partition root.
            seed: Random seed for the speaker-level train/dev/eval split.
            dry_run: If True, preview counts and example filenames without
                writing anything to disk.
            include_noisy_partial_spoof: If True, include all partial-spoof
                rows regardless of WER/CER instead of just the clean subset.
        """
        self.bonafide_dir = bonafide_dir or DEFAULT_BONAFIDE_DIR
        self.partial_spoof_csv = partial_spoof_csv or DEFAULT_PARTIAL_SPOOF_CSV
        self.fullspoof_dirs = fullspoof_dirs or FULLSPOOF_OUTPUT_DIRS
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.seed = seed
        self.dry_run = dry_run
        self.include_noisy_partial_spoof = include_noisy_partial_spoof
        self._preview_examples: List[str] = []

    def run(self) -> SpeakerPartitionReport:
        """Build the full partition and return a summary report.

        Returns:
            SpeakerPartitionReport with per-split counts and notes.
        """
        logger.info("Loading bonafide speaker list...")
        bonafide_speakers = self._list_bonafide_speakers()
        logger.info(f"  {len(bonafide_speakers)} bonafide speakers found.")

        logger.info("Indexing full-spoof protocol files by speaker...")
        fullspoof_index = self._index_fullspoof_files()
        fullspoof_speakers = set(fullspoof_index.keys())
        logger.info(f"  {len(fullspoof_speakers)} speakers with full-spoof audio.")

        logger.info("Indexing partial-spoof manifest by speaker...")
        partialspoof_index = self._index_partialspoof_files()
        partialspoof_speakers = set(partialspoof_index.keys())
        logger.info(f"  {len(partialspoof_speakers)} speakers with partial-spoof audio.")

        all_speakers = sorted(
            set(bonafide_speakers) | fullspoof_speakers | partialspoof_speakers
        )
        missing_bonafide = sorted(
            (fullspoof_speakers | partialspoof_speakers) - set(bonafide_speakers)
        )
        if missing_bonafide:
            logger.warning(
                f"{len(missing_bonafide)} speakers have spoof audio but no "
                f"bonafide directory (sample: {missing_bonafide[:5]})."
            )

        split_assignment = self._assign_speakers_to_splits(all_speakers)

        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        per_split: List[PartitionSplitStat] = []

        for split_name in ("train", "dev", "eval"):
            speakers_in_split = [s for s in all_speakers if split_assignment[s] == split_name]
            stat = self._build_split(
                split_name, speakers_in_split, fullspoof_index, partialspoof_index
            )
            per_split.append(stat)

        report = SpeakerPartitionReport(
            per_split=per_split,
            total_speakers=len(all_speakers),
            speakers_missing_bonafide=missing_bonafide,
            notes=[
                "Speaker-disjoint by construction: every file a speaker owns, "
                "across bonafide, full-spoof, and partial-spoof, is assigned "
                "to the same split.",
                "Symlinks only; no audio bytes were copied or decoded.",
                "Partial-spoof source is corpus_samples.csv's spliced_audio_path "
                "(includes all quality_flag tiers; filter by wer/cer/quality_flag "
                "downstream if only the clean subset is wanted).",
            ],
        )
        self._log_summary(report)
        return report

    def _list_bonafide_speakers(self) -> List[str]:
        """List speaker IDs from the bonafide directory.

        Returns:
            Sorted list of speaker_id directory names.
        """
        return sorted(
            d.name for d in self.bonafide_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _bonafide_files_for_speaker(self, speaker_id: str) -> List[Path]:
        """List every bonafide audio file for one speaker, across all subsplits.

        Args:
            speaker_id: HABLA v2 speaker identifier.

        Returns:
            List of audio file paths (may be empty if the speaker has no
            bonafide directory).
        """
        speaker_dir = self.bonafide_dir / speaker_id
        if not speaker_dir.exists():
            return []
        return sorted(
            p for ext in ("*.wav", "*.flac", "*.mp3")
            for p in speaker_dir.rglob(ext)
        )

    def _index_fullspoof_files(self) -> Dict[str, List[Tuple[str, Path]]]:
        """Parse every system's ASVspoof protocol files into a speaker index.

        Returns:
            Mapping speaker_id -> list of (system_name, audio_file_path).
        """
        index: Dict[str, List[Tuple[str, Path]]] = {}

        for system, output_dir in self.fullspoof_dirs.items():
            la_dir = output_dir / "LA"
            if not la_dir.exists():
                logger.warning(f"{system}: no LA/ directory under {output_dir}, skipping.")
                continue

            for split_dir in sorted(la_dir.glob("ASVspoof2019_LA_*")):
                flac_dir = split_dir / "flac"
                protocol_candidates = sorted(split_dir.glob("ASVspoof2019.LA.cm.*.txt"))
                if not protocol_candidates:
                    logger.warning(f"{system}: no protocol file found in {split_dir}.")
                    continue
                protocol_path = protocol_candidates[0]

                with open(protocol_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        speaker_id, audio_id = parts[0], parts[1]
                        audio_path = flac_dir / f"{audio_id}.flac"
                        index.setdefault(speaker_id, []).append((system, audio_path))

        return index

    def _index_partialspoof_files(self) -> Dict[str, List[Tuple[str, str, Path]]]:
        """Parse the partial-spoof manifest into a speaker index.

        By default, keeps only the "clean" subset the paper recommends for
        training (WER <= 0.15, CER <= 0.10; 15,641 of 18,421 rows) -- set
        include_noisy_partial_spoof=True on the builder to keep all rows.

        Returns:
            Mapping speaker_id -> list of (system_name, tier, spliced_audio_path).
        """
        index: Dict[str, List[Tuple[str, str, Path]]] = {}
        total_rows = 0
        kept_rows = 0

        with open(self.partial_spoof_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total_rows += 1
                if not self.include_noisy_partial_spoof:
                    if float(row["wer"]) > self.CLEAN_WER_MAX or float(row["cer"]) > self.CLEAN_CER_MAX:
                        continue
                kept_rows += 1

                speaker_id = row["speaker_id"]
                system = row["attack"]
                tier = row["tier"]
                audio_path = Path(row["spliced_audio_path"])
                index.setdefault(speaker_id, []).append((system, tier, audio_path))

        if not self.include_noisy_partial_spoof:
            logger.info(
                f"Partial-spoof: kept clean subset {kept_rows}/{total_rows} rows "
                f"(WER <= {self.CLEAN_WER_MAX}, CER <= {self.CLEAN_CER_MAX})."
            )

        return index

    def _assign_speakers_to_splits(self, speakers: List[str]) -> Dict[str, str]:
        """Randomly assign each speaker to train/dev/eval.

        Args:
            speakers: Sorted list of all unique speaker IDs.

        Returns:
            Mapping speaker_id -> split name.
        """
        rng = random.Random(self.seed)
        shuffled = speakers.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_dev = int(n * DEV_RATIO)

        assignment: Dict[str, str] = {}
        for speaker_id in shuffled[:n_train]:
            assignment[speaker_id] = "train"
        for speaker_id in shuffled[n_train:n_train + n_dev]:
            assignment[speaker_id] = "dev"
        for speaker_id in shuffled[n_train + n_dev:]:
            assignment[speaker_id] = "eval"

        return assignment

    def _build_split(
        self,
        split_name: str,
        speakers: List[str],
        fullspoof_index: Dict[str, List[Tuple[str, Path]]],
        partialspoof_index: Dict[str, List[Tuple[str, str, Path]]],
    ) -> PartitionSplitStat:
        """Create symlinks for every speaker assigned to one split.

        Args:
            split_name: One of 'train', 'dev', 'eval'.
            speakers: Speaker IDs assigned to this split.
            fullspoof_index: Output of _index_fullspoof_files.
            partialspoof_index: Output of _index_partialspoof_files.

        Returns:
            PartitionSplitStat with the resulting counts.
        """
        split_dir = self.output_dir / split_name
        bonafide_count = 0
        fullspoof_count = 0
        partialspoof_count = 0
        preview_speakers_left = 2

        for speaker_id in tqdm(speakers, desc=f"  {split_name}"):
            speaker_dir = split_dir / speaker_id
            capture_preview = preview_speakers_left > 0
            if capture_preview:
                preview_speakers_left -= 1
                self._preview_examples.append(f"[{split_name}] {speaker_dir}/")
            if not self.dry_run:
                speaker_dir.mkdir(parents=True, exist_ok=True)

            for i, path in enumerate(self._bonafide_files_for_speaker(speaker_id)):
                link_name = f"bonafide_{speaker_id}_{i:04d}{path.suffix}"
                self._symlink(path, speaker_dir / link_name, capture_preview and i < 2)
                bonafide_count += 1

            for i, (system, path) in enumerate(fullspoof_index.get(speaker_id, [])):
                link_name = f"spoof_{system}_{speaker_id}_{i:04d}{path.suffix}"
                self._symlink(path, speaker_dir / link_name, capture_preview and i < 2)
                fullspoof_count += 1

            for i, (system, tier, path) in enumerate(partialspoof_index.get(speaker_id, [])):
                token = f"{system}-ps{tier.lower()}"
                link_name = f"spoof_{token}_{speaker_id}_{i:04d}{path.suffix}"
                self._symlink(path, speaker_dir / link_name, capture_preview and i < 2)
                partialspoof_count += 1

        return PartitionSplitStat(
            split=split_name,
            speaker_count=len(speakers),
            bonafide_count=bonafide_count,
            fullspoof_count=fullspoof_count,
            partialspoof_count=partialspoof_count,
        )

    def _symlink(self, source: Path, link_path: Path, capture_preview: bool = False) -> None:
        """Create a symlink (or, in dry-run mode, just validate the source).

        Args:
            source: The real audio file the symlink should point to.
            link_path: Where to create the symlink.
            capture_preview: If True, record this (source -> link) pair for
                the preview summary, regardless of dry_run.
        """
        source_exists = source.exists()
        if not source_exists:
            logger.warning(f"Source file missing, skipping: {source}")
        if capture_preview:
            status = "OK" if source_exists else "MISSING"
            self._preview_examples.append(f"    {link_path.name}  -> {source}  [{status}]")
        if self.dry_run or not source_exists:
            return
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(source.resolve())

    def _log_summary(self, report: SpeakerPartitionReport) -> None:
        """Log a human-readable summary of the report.

        Args:
            report: The computed report.
        """
        logger.info("=" * 70)
        header = "SPEAKER-DISJOINT PARTITION -- DRY RUN (nothing written)" if self.dry_run \
            else "SPEAKER-DISJOINT PARTITION COMPLETE"
        logger.info(header)
        logger.info("=" * 70)
        if self._preview_examples:
            logger.info("Example structure (first 2 speakers per split):")
            for line in self._preview_examples:
                logger.info(line)
            logger.info("-" * 70)
        for stat in report.per_split:
            logger.info(
                f"  {stat.split:6s} speakers={stat.speaker_count:5d}  "
                f"bonafide={stat.bonafide_count:7d}  "
                f"fullspoof={stat.fullspoof_count:7d}  "
                f"partialspoof={stat.partialspoof_count:6d}"
            )
        logger.info(f"Total speakers: {report.total_speakers}")
        if report.speakers_missing_bonafide:
            logger.warning(
                f"{len(report.speakers_missing_bonafide)} speakers have spoof "
                f"audio but no bonafide directory."
            )
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 70)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build a speaker-disjoint train/dev/eval symlink partition of MARSA."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output partition root (default: "
             "data/marsa_speaker_disjoint_partition).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the speaker-level split (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Index and count everything, print example filenames, but "
             "write nothing to disk.",
    )
    parser.add_argument(
        "--include-noisy-partial-spoof",
        action="store_true",
        help="Include all 18,421 partial-spoof rows instead of just the "
             "paper's recommended clean subset (WER<=0.15, CER<=0.10, "
             "15,641 rows).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    builder = SpeakerDisjointPartitionBuilder(
        output_dir=args.output,
        seed=args.seed,
        dry_run=args.dry_run,
        include_noisy_partial_spoof=args.include_noisy_partial_spoof,
    )
    builder.run()
