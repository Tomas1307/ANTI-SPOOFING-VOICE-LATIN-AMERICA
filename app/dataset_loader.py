"""
Dataset Loader

Handles loading of original voice files for the augmentation pipeline and
provides organized access to the required audio resources.

Discovery for every split (train/dev/eval) goes through a single code path so
that a change to the accepted-extension set or to the directory-layout logic
cannot be applied to one split and forgotten on another. Every regular file
encountered is either ingested or explicitly accounted for in a
DatasetDiscoveryReport, so files can never be dropped silently.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from app.schemas.dataset_discovery_report import DatasetDiscoveryReport


class DatasetLoader:
    """
    Loader for the speaker-partitioned voice dataset.

    Supports two directory layouts:
    1. Split-first: ``voices_root/<split>/<speaker_id>/files`` (preferred)
    2. Speaker-first: ``voices_root/<speaker_id>/<split>/files`` (legacy)

    Attributes:
        voices_root: Path to the partitioned dataset root.
        musan_root: Path to the MUSAN noise dataset.
        rir_root: Path to the RIR dataset.
        reports: Discovery report per split name, populated on each load.
    """

    AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")
    SPLIT_NAMES = ("train", "dev", "eval")
    BONAFIDE_PREFIX = "bonafide_"
    SPOOF_PREFIX = "spoof_"

    def __init__(
        self,
        voices_root: str = "data/marsa_speaker_disjoint_partition",
        musan_root: str = "data/noise_dataset/musan",
        rir_root: str = "data/noise_dataset/RIR"
    ):
        """
        Initialize dataset loader.

        Args:
            voices_root: Root directory for original voice files.
            musan_root: Root directory for the MUSAN noise dataset.
            rir_root: Root directory for RIR files.

        Raises:
            FileNotFoundError: If voices_root does not exist.
        """
        self.voices_root = Path(voices_root)
        self.musan_root = Path(musan_root)
        self.rir_root = Path(rir_root)
        self.reports: Dict[str, DatasetDiscoveryReport] = {}
        self._cache: Dict[str, List[Dict[str, str]]] = {}

        self._validate_paths()

    def _validate_paths(self) -> None:
        """
        Validate that all required paths exist.

        Raises:
            FileNotFoundError: If the voices root is missing.
        """
        if not self.voices_root.exists():
            raise FileNotFoundError(f"Voices root not found: {self.voices_root}")

        if not self.musan_root.exists():
            print(f"Warning: MUSAN root not found: {self.musan_root}")

        if not self.rir_root.exists():
            print(f"Warning: RIR root not found: {self.rir_root}")

    def _scan_directory(
        self,
        directory: Path,
        skipped: Dict[str, int]
    ) -> List[str]:
        """
        Collect audio files under a directory, accounting for every skip.

        Args:
            directory: Directory to walk recursively.
            skipped: Mutable tally of ignored files keyed by lowercase
                extension; updated in place.

        Returns:
            Sorted list of absolute paths to accepted audio files.
        """
        audio_files: List[str] = []

        for root, _dirs, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(self.AUDIO_EXTENSIONS):
                    audio_files.append(os.path.join(root, filename))
                else:
                    extension = os.path.splitext(filename)[1].lower() or "<none>"
                    skipped[extension] = skipped.get(extension, 0) + 1

        return sorted(audio_files)

    def _classify(self, filename: str) -> Optional[str]:
        """
        Classify a filename as bonafide or spoof by its prefix.

        Args:
            filename: Basename of the audio file.

        Returns:
            'bonafide', 'spoof', or None when the name matches neither
            prefix convention.
        """
        if filename.startswith(self.BONAFIDE_PREFIX):
            return "bonafide"
        if filename.startswith(self.SPOOF_PREFIX):
            return "spoof"
        return None

    def _attack_id(self, filename: str, file_type: str) -> str:
        """
        Extract the attack-system identifier from a partition filename.

        Partition filenames encode the generating system as the token after
        the ``spoof_`` prefix: ``spoof_fishgram_esm_00030_0001.flac`` yields
        ``fishgram`` and the partial-spoof variant
        ``spoof_omnivoice-psw2_esm_00030_0001.flac`` yields
        ``omnivoice-psw2``. Bonafide files carry no system and yield ``-``,
        matching the ASVspoof convention for the system column.

        Args:
            filename: Basename of the audio file.
            file_type: 'bonafide' or 'spoof' as returned by _classify.

        Returns:
            The attack-system identifier, '-' for bonafide, or 'unknown'
            when a spoof filename does not follow the naming convention.
        """
        if file_type == "bonafide":
            return "-"
        parts = filename.split("_")
        if len(parts) >= 2 and parts[0] == "spoof" and parts[1]:
            return parts[1]
        return "unknown"

    def _resolve_speaker_dirs(self, split: str) -> tuple:
        """
        Resolve the speaker directories and their audio roots for a split.

        Args:
            split: One of 'train', 'dev', or 'eval'.

        Returns:
            Tuple of (structure_name, list of (speaker_id, audio_root) pairs).
            Speakers are sorted by identifier so that discovery order is
            deterministic across runs and machines, which the seeded
            augmentation depends on for reproducibility.
        """
        split_dir = self.voices_root / split

        if split_dir.is_dir():
            speaker_dirs = sorted(
                (d for d in split_dir.iterdir() if d.is_dir()),
                key=lambda d: d.name
            )
            return "split-first", [(d.name, d) for d in speaker_dirs]

        speaker_dirs = sorted(
            (d for d in self.voices_root.iterdir() if d.is_dir()),
            key=lambda d: d.name
        )
        pairs = [
            (d.name, d / split) for d in speaker_dirs if (d / split).is_dir()
        ]
        return "speaker-first", pairs

    def _discover_split_files(self, split: str) -> List[Dict[str, str]]:
        """
        Discover every audio file belonging to one split.

        This is the single discovery path shared by all splits. It records a
        DatasetDiscoveryReport in ``self.reports[split]`` and prints a warning
        whenever files on disk were ignored, so that a mismatch between files
        present and files ingested can never pass unnoticed. Results are
        memoized per split because a single run queries each split several
        times and walking the partition is I/O bound.

        Args:
            split: One of 'train', 'dev', or 'eval'.

        Returns:
            List of dictionaries with keys ``filepath``, ``speaker_id``,
            ``split``, ``filename`` and ``file_type``.

        Raises:
            ValueError: If split is not a recognized split name.
        """
        if split not in self.SPLIT_NAMES:
            raise ValueError(
                f"Unknown split '{split}'; expected one of {self.SPLIT_NAMES}"
            )

        if split in self._cache:
            return self._cache[split]

        structure, speaker_pairs = self._resolve_speaker_dirs(split)
        print(
            f"\nLoading {split} files from {len(speaker_pairs)} speakers "
            f"({structure} structure)..."
        )

        discovered: List[Dict[str, str]] = []
        skipped: Dict[str, int] = {}
        unknown_prefix = 0

        for speaker_id, audio_root in speaker_pairs:
            for audio_file in self._scan_directory(audio_root, skipped):
                filename = Path(audio_file).name
                file_type = self._classify(filename)

                if file_type is None:
                    unknown_prefix += 1
                    file_type = "spoof"

                discovered.append({
                    "filepath": audio_file,
                    "speaker_id": speaker_id,
                    "split": split,
                    "filename": filename,
                    "file_type": file_type,
                    "attack_id": self._attack_id(filename, file_type)
                })

        report = DatasetDiscoveryReport(
            split=split,
            structure=structure,
            speaker_count=len(speaker_pairs),
            audio_file_count=len(discovered),
            bonafide_count=sum(
                1 for f in discovered if f["file_type"] == "bonafide"
            ),
            spoof_count=sum(1 for f in discovered if f["file_type"] == "spoof"),
            unknown_prefix_count=unknown_prefix,
            skipped_by_extension=skipped
        )
        self.reports[split] = report

        print(f"Loaded {report.audio_file_count} {split} files")

        if report.skipped_total:
            print(
                f"WARNING: {report.skipped_total} file(s) under {split} were "
                f"ignored because their extension is not in "
                f"{list(self.AUDIO_EXTENSIONS)}: {report.skipped_by_extension}"
            )
        if report.unknown_prefix_count:
            print(
                f"WARNING: {report.unknown_prefix_count} {split} file(s) match "
                f"neither '{self.BONAFIDE_PREFIX}' nor '{self.SPOOF_PREFIX}' "
                f"and were classified as spoof by default."
            )

        self._cache[split] = discovered
        return discovered

    def load_train_files(self) -> List[Dict[str, str]]:
        """
        Load all training audio files.

        Returns:
            List of file metadata dictionaries for the train split.
        """
        return self._discover_split_files("train")

    def load_dev_files(self) -> List[Dict[str, str]]:
        """
        Load all dev (validation) audio files.

        Returns:
            List of file metadata dictionaries for the dev split.
        """
        return self._discover_split_files("dev")

    def load_eval_files(self) -> List[Dict[str, str]]:
        """
        Load all eval (test) audio files.

        Returns:
            List of file metadata dictionaries for the eval split.
        """
        return self._discover_split_files("eval")

    def load_bonafide_train_files(self) -> List[Dict[str, str]]:
        """
        Load only bonafide training files.

        Returns:
            List of bonafide file metadata dictionaries.
        """
        return self._filter_by_type(self.load_train_files(), "bonafide", "train")

    def load_spoof_train_files(self) -> List[Dict[str, str]]:
        """
        Load only spoof training files.

        Returns:
            List of spoof file metadata dictionaries.
        """
        return self._filter_by_type(self.load_train_files(), "spoof", "train")

    def load_bonafide_dev_files(self) -> List[Dict[str, str]]:
        """
        Load only bonafide dev files.

        Returns:
            List of bonafide file metadata dictionaries.
        """
        return self._filter_by_type(self.load_dev_files(), "bonafide", "dev")

    def load_spoof_dev_files(self) -> List[Dict[str, str]]:
        """
        Load only spoof dev files.

        Returns:
            List of spoof file metadata dictionaries.
        """
        return self._filter_by_type(self.load_dev_files(), "spoof", "dev")

    def load_bonafide_eval_files(self) -> List[Dict[str, str]]:
        """
        Load only bonafide eval files.

        Returns:
            List of bonafide file metadata dictionaries.
        """
        return self._filter_by_type(self.load_eval_files(), "bonafide", "eval")

    def load_spoof_eval_files(self) -> List[Dict[str, str]]:
        """
        Load only spoof eval files.

        Returns:
            List of spoof file metadata dictionaries.
        """
        return self._filter_by_type(self.load_eval_files(), "spoof", "eval")

    def _filter_by_type(
        self,
        files: List[Dict[str, str]],
        file_type: str,
        split: str
    ) -> List[Dict[str, str]]:
        """
        Filter discovered files by class.

        Args:
            files: Discovered file metadata dictionaries.
            file_type: Either 'bonafide' or 'spoof'.
            split: Split name, used only for the progress message.

        Returns:
            The subset of files matching file_type.
        """
        selected = [f for f in files if f["file_type"] == file_type]
        print(f"Filtered {len(selected)} {file_type} {split} files")
        return selected

    def get_dataset_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get per-split counts for the whole partition.

        Returns:
            Mapping of split name to a dictionary with 'total', 'bonafide'
            and 'spoof' counts.
        """
        statistics: Dict[str, Dict[str, int]] = {}

        for split in self.SPLIT_NAMES:
            files = self._discover_split_files(split)
            statistics[split] = {
                "total": len(files),
                "bonafide": sum(
                    1 for f in files if f["file_type"] == "bonafide"
                ),
                "spoof": sum(1 for f in files if f["file_type"] == "spoof")
            }

        return statistics

    def print_summary(self) -> None:
        """Print a dataset summary with the bonafide/spoof breakdown."""
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)

        statistics = self.get_dataset_statistics()

        for split, counts in statistics.items():
            total = counts["total"]
            if total == 0:
                print(f"\n  {split.upper()}: empty")
                continue

            print(f"\n  {split.upper()}: {total:,} total")
            print(
                f"    - Bonafide: {counts['bonafide']:,} "
                f"({counts['bonafide'] / total * 100:.1f}%)"
            )
            print(
                f"    - Spoof:    {counts['spoof']:,} "
                f"({counts['spoof'] / total * 100:.1f}%)"
            )

        print("=" * 70 + "\n")
