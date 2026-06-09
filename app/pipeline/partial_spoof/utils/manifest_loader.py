"""
Manifest CSV reader/writer plus slice helpers for the dispatch plan.

The manifest is a flat CSV where each row is one ManifestEntry. This
util handles serialisation (List[ManifestEntry] -> CSV), deserialisation
(CSV -> List[ManifestEntry]), and the slice operations consumed by per-
attack pipeline runs (filter by attack, by partition, by both).
"""
import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from loguru import logger

from app.pipeline.partial_spoof.schemas.manifest_entry import ManifestEntry
from app.pipeline.partial_spoof.schemas.manifest_summary import ManifestSummary


class ManifestLoader:
    """Read, write, and slice the partial spoof dispatch manifest.

    The on-disk CSV uses '|' to delimit the planned_tiers list so
    Excel and pandas readers see flat columns. The bonafide_transcript
    column is double-quoted by the standard csv writer to handle
    embedded commas and newlines.

    Attributes:
        entries: Loaded ManifestEntry list (empty until load() is called).
    """

    PLANNED_TIERS_DELIMITER = "|"
    FIELD_NAMES = [
        "sample_key",
        "speaker_id",
        "audio_path",
        "split",
        "partition",
        "attack",
        "planned_tiers",
        "word_count",
        "bonafide_transcript",
    ]

    def __init__(self) -> None:
        """Initialise an empty loader. Use load() or set entries directly."""
        self.entries: List[ManifestEntry] = []

    def load(self, manifest_path: Path) -> List[ManifestEntry]:
        """Read the manifest CSV from disk into ManifestEntry instances.

        Args:
            manifest_path: Path to the manifest CSV file.

        Returns:
            List of ManifestEntry. Also stored on the instance.

        Raises:
            FileNotFoundError: If the manifest CSV does not exist.
            ValueError: If a row is missing required fields or has an
                unparseable planned_tiers / word_count.
        """
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest CSV not found at {manifest_path}. "
                "Run the manifest generation script first."
            )

        entries: List[ManifestEntry] = []
        with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader, start=2):
                entries.append(self._row_to_entry(row, row_idx))

        self.entries = entries
        logger.info(
            f"Loaded manifest from {manifest_path}: {len(entries)} entries."
        )
        return entries

    def save(
        self,
        entries: List[ManifestEntry],
        manifest_path: Path,
    ) -> None:
        """Serialise ManifestEntry list to a CSV file.

        Args:
            entries: Manifest rows to write.
            manifest_path: Output CSV path; parent directory is created.
        """
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.FIELD_NAMES,
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(self._entry_to_row(entry))
        logger.info(
            f"Wrote manifest to {manifest_path}: {len(entries)} entries."
        )

    def save_summary(
        self,
        summary: ManifestSummary,
        summary_path: Path,
    ) -> None:
        """Serialise the ManifestSummary to a JSON sidecar file.

        Args:
            summary: ManifestSummary instance to persist.
            summary_path: Output JSON path; parent directory is created.
        """
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(summary.model_dump_json(indent=2))
        logger.info(f"Wrote manifest summary to {summary_path}.")

    def by_attack(self, attack: str) -> List[ManifestEntry]:
        """Slice entries to a single attack system.

        Args:
            attack: Attack identifier (e.g. 'omnivoice').

        Returns:
            List of entries with entry.attack == attack.
        """
        return [e for e in self.entries if e.attack == attack]

    def by_partition(self, partition: str) -> List[ManifestEntry]:
        """Slice entries to a single partition.

        Args:
            partition: 'not_jittered' or 'jittered'.

        Returns:
            List of entries with entry.partition == partition.
        """
        return [e for e in self.entries if e.partition == partition]

    def by_attack_and_partition(
        self,
        attack: str,
        partition: str,
    ) -> List[ManifestEntry]:
        """Slice entries to one (attack, partition) cell.

        Args:
            attack: Attack identifier.
            partition: Partition identifier.

        Returns:
            List of entries matching both filters.
        """
        return [
            e for e in self.entries
            if e.attack == attack and e.partition == partition
        ]

    def sample_keys(
        self,
        attack: Optional[str] = None,
        partition: Optional[str] = None,
    ) -> List[str]:
        """Return sample_keys, optionally filtered by attack and/or partition.

        Args:
            attack: Optional attack filter.
            partition: Optional partition filter.

        Returns:
            Sorted list of sample_keys matching the filters.
        """
        selected = self.entries
        if attack is not None:
            selected = [e for e in selected if e.attack == attack]
        if partition is not None:
            selected = [e for e in selected if e.partition == partition]
        return sorted(e.sample_key for e in selected)

    def index_by_sample_key(
        self,
        attack: Optional[str] = None,
        partition: Optional[str] = None,
    ) -> Dict[str, ManifestEntry]:
        """Build a sample_key -> ManifestEntry index.

        Args:
            attack: Optional attack filter applied before indexing.
            partition: Optional partition filter applied before indexing.

        Returns:
            Dict mapping sample_key to its ManifestEntry.
        """
        selected = self.entries
        if attack is not None:
            selected = [e for e in selected if e.attack == attack]
        if partition is not None:
            selected = [e for e in selected if e.partition == partition]
        return {e.sample_key: e for e in selected}

    def iter_entries(self) -> Iterator[ManifestEntry]:
        """Iterate all loaded entries (no filtering).

        Yields:
            Each ManifestEntry in load order.
        """
        for entry in self.entries:
            yield entry

    def _entry_to_row(self, entry: ManifestEntry) -> Dict[str, str]:
        """Flatten one ManifestEntry into a CSV-writable dict.

        Args:
            entry: ManifestEntry instance.

        Returns:
            Dict suitable for csv.DictWriter.writerow.
        """
        return {
            "sample_key": entry.sample_key,
            "speaker_id": entry.speaker_id,
            "audio_path": str(entry.audio_path),
            "split": entry.split,
            "partition": entry.partition,
            "attack": entry.attack,
            "planned_tiers": self.PLANNED_TIERS_DELIMITER.join(
                entry.planned_tiers
            ),
            "word_count": str(entry.word_count),
            "bonafide_transcript": entry.bonafide_transcript or "",
        }

    def _row_to_entry(
        self,
        row: Dict[str, str],
        row_idx: int,
    ) -> ManifestEntry:
        """Parse one CSV row into a ManifestEntry instance.

        Args:
            row: Mapping from column name to string value.
            row_idx: 1-indexed CSV row number, for error messages.

        Returns:
            Fully-typed ManifestEntry.

        Raises:
            ValueError: If required fields are missing or unparseable.
        """
        missing = [f for f in self.FIELD_NAMES if f not in row]
        if missing:
            raise ValueError(
                f"Manifest row {row_idx} is missing columns: {missing}"
            )
        try:
            word_count = int(row["word_count"])
        except ValueError as exc:
            raise ValueError(
                f"Manifest row {row_idx} has unparseable word_count "
                f"'{row['word_count']}': {exc}"
            ) from exc

        tiers_raw = row["planned_tiers"].strip()
        if tiers_raw:
            planned_tiers = tiers_raw.split(self.PLANNED_TIERS_DELIMITER)
        else:
            planned_tiers = []

        transcript = row.get("bonafide_transcript") or None

        return ManifestEntry(
            speaker_id=row["speaker_id"],
            sample_key=row["sample_key"],
            audio_path=Path(row["audio_path"]),
            split=row["split"],
            partition=row["partition"],
            attack=row["attack"],
            planned_tiers=planned_tiers,
            word_count=word_count,
            bonafide_transcript=transcript,
        )

    def load_summary(
        self,
        summary_path: Path,
    ) -> ManifestSummary:
        """Read a previously-saved ManifestSummary JSON.

        Args:
            summary_path: Path to the summary JSON sidecar.

        Returns:
            ManifestSummary instance.

        Raises:
            FileNotFoundError: If the summary file is missing.
        """
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Manifest summary not found at {summary_path}."
            )
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ManifestSummary(**payload)
