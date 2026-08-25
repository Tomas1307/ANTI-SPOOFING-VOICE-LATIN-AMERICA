"""
Readers and writers for MARSA protocol, metadata and score files.

The corpus declares every clip twice: once in a byte-faithful ASVspoof2019 LA
protocol file, and once in a companion metadata CSV that carries the
augmentation label and the source utterance basename. These helpers read both,
join them and expose the result as typed protocol entries.
"""
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

from app.pipeline.training.schemas.protocol_entry import ProtocolEntry
from app.pipeline.training.settings import settings


def split_dir(corpus_root: Path, split: str) -> Path:
    """Return the directory holding one split of the corpus.

    Args:
        corpus_root: Corpus directory containing the LA tree.
        split: Split name, one of train, dev or eval.

    Returns:
        Path of the split directory.
    """
    return corpus_root / "LA" / settings.SPLIT_DIR_TEMPLATE.format(split=split)


def protocol_path(corpus_root: Path, split: str) -> Path:
    """Return the ASVspoof2019 LA protocol file path for a split.

    Args:
        corpus_root: Corpus directory containing the LA tree.
        split: Split name, one of train, dev or eval.

    Returns:
        Path of the protocol file.
    """
    return split_dir(corpus_root, split) / settings.PROTOCOL_FILENAMES[split]


def metadata_path(corpus_root: Path, split: str) -> Path:
    """Return the MARSA metadata CSV path for a split.

    Args:
        corpus_root: Corpus directory containing the LA tree.
        split: Split name, one of train, dev or eval.

    Returns:
        Path of the metadata CSV.
    """
    return split_dir(corpus_root, split) / settings.METADATA_TEMPLATE.format(split=split)


def flac_dir(corpus_root: Path, split: str) -> Path:
    """Return the FLAC directory for a split.

    Args:
        corpus_root: Corpus directory containing the LA tree.
        split: Split name, one of train, dev or eval.

    Returns:
        Path of the directory holding the split clips.
    """
    return split_dir(corpus_root, split) / "flac"


def read_protocol(path: Path) -> List[Tuple[str, str, str, str]]:
    """Read an ASVspoof2019 LA protocol file.

    The format is five whitespace-separated fields, the third being a literal
    placeholder used only in the physical access scenario.

    Args:
        path: Protocol file path.

    Returns:
        One tuple of (speaker_id, audio_id, attack_id, key) per row, in file
        order.

    Raises:
        FileNotFoundError: If the protocol file does not exist.
        ValueError: If any row does not carry exactly five fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    rows: List[Tuple[str, str, str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 5:
                raise ValueError(
                    f"{path}:{line_number} has {len(fields)} fields, expected 5"
                )
            speaker_id, audio_id, _placeholder, attack_id, key = fields
            rows.append((speaker_id, audio_id, attack_id, key))
    return rows


def read_metadata(path: Path) -> List[ProtocolEntry]:
    """Read a MARSA metadata CSV into typed entries.

    Args:
        path: Metadata CSV path.

    Returns:
        One entry per row, in file order.

    Raises:
        FileNotFoundError: If the metadata CSV does not exist.
        ValueError: If the header does not carry the expected columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    expected = ["audio_id", "speaker_id", "key", "attack_id", "aug_id", "source_file"]
    entries: List[ProtocolEntry] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in expected if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for row in reader:
            entries.append(
                ProtocolEntry(
                    audio_id=row["audio_id"],
                    speaker_id=row["speaker_id"],
                    key=row["key"],
                    attack_id=row["attack_id"],
                    aug_id=row["aug_id"],
                    source_file=row["source_file"],
                )
            )
    return entries


def read_strict_filter(path: Path) -> Dict[str, Set[str]]:
    """Read the strict sentence-disjoint filter table.

    The table marks, for each dev and eval source utterance, whether its
    underlying sentence is unseen in training. Clips join to it through the
    source_file column, so the strict subset propagates to every augmentation
    tier without recomputation.

    Args:
        path: Filter CSV path.

    Returns:
        Mapping of split name to the set of strict source basenames.

    Raises:
        FileNotFoundError: If the filter table does not exist.
        ValueError: If the header does not carry the expected columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Strict filter table not found: {path}")

    strict: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for column in ("source_file", "split", "strict"):
            if column not in (reader.fieldnames or []):
                raise ValueError(f"{path} is missing column: {column}")
        for row in reader:
            if str(row["strict"]).strip().lower() in ("true", "1", "yes"):
                strict.setdefault(row["split"], set()).add(row["source_file"])
    return strict


def write_scores(path: Path, rows: List[Tuple[str, str, str, float]]) -> None:
    """Write a per-clip countermeasure score file.

    The layout mirrors the ASVspoof score-file convention so the official
    evaluation package can consume it directly.

    Args:
        path: Destination path.
        rows: One tuple of (audio_id, attack_id, key, score) per clip.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=" ")
        for audio_id, attack_id, key, score in rows:
            writer.writerow([audio_id, attack_id, key, f"{score:.6f}"])
