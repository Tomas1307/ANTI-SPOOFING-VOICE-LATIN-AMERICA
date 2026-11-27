"""
Step 2: resolve the corpus protocol into materialised dataset splits.
"""
from pathlib import Path
from typing import Dict, List

from loguru import logger

from app.pipeline.training.schemas.dataset_split import DatasetSplit
from app.pipeline.training.schemas.protocol_entry import ProtocolEntry
from app.pipeline.training.utils import protocol_io


class ProtocolDatasetBuilder:
    """Turn protocol and metadata files into typed, ordered dataset splits.

    Ordering is made explicit here rather than left to the filesystem: rows
    are sorted by clip identifier so that a resumed run replays the same
    sequence, and any sampling cap takes a deterministic stratified slice
    rather than the head of the file.

    Attributes:
        corpus_root: Corpus directory containing the LA tree.
        splits: Split names to build.
        max_train_items: Cap on training clips; zero means no cap.
        seed: Seed used when a cap forces subsampling.
    """

    def __init__(
        self,
        corpus_root: Path,
        splits: List[str],
        max_train_items: int = 0,
        seed: int = 42,
    ) -> None:
        """Initialize the builder.

        Args:
            corpus_root: Corpus directory containing the LA tree.
            splits: Split names to build.
            max_train_items: Cap on training clips, for smoke tests. Zero
                means no cap.
            seed: Seed used when a cap forces subsampling.
        """
        self.corpus_root = Path(corpus_root)
        self.splits = list(splits)
        self.max_train_items = max_train_items
        self.seed = seed

    def execute(self) -> Dict[str, DatasetSplit]:
        """Build every requested split.

        Returns:
            Mapping of split name to its resolved description.

        Raises:
            FileNotFoundError: If a split's metadata or FLAC directory is
                absent.
            ValueError: If a split resolves to no clips.
        """
        logger.info(f"Step {self.__class__.__name__}: Starting")
        built: Dict[str, DatasetSplit] = {}

        for split in self.splits:
            entries = protocol_io.read_metadata(
                protocol_io.metadata_path(self.corpus_root, split)
            )
            entries.sort(key=lambda row: row.audio_id)

            if split == "train" and self.max_train_items:
                entries = self._subsample(entries, self.max_train_items)

            directory = protocol_io.flac_dir(self.corpus_root, split)
            if not directory.is_dir():
                raise FileNotFoundError(f"FLAC directory not found: {directory}")
            if not entries:
                raise ValueError(f"Split '{split}' resolved to zero clips")

            label_counts: Dict[str, int] = {}
            for entry in entries:
                label_counts[entry.key] = label_counts.get(entry.key, 0) + 1

            built[split] = DatasetSplit(
                name=split,
                flac_dir=str(directory),
                entries=entries,
                speaker_count=len({entry.speaker_id for entry in entries}),
                attack_ids=sorted(
                    {entry.attack_id for entry in entries if entry.key == "spoof"}
                ),
                label_counts=label_counts,
            )
            logger.info(
                f"  {split}: {len(entries):,} clips, "
                f"{built[split].speaker_count:,} speakers, "
                f"{label_counts.get('bonafide', 0):,} bonafide / "
                f"{label_counts.get('spoof', 0):,} spoof, "
                f"{len(built[split].attack_ids)} attack ids"
            )

        logger.info(f"Step {self.__class__.__name__}: Complete")
        return built

    def _subsample(
        self, entries: List[ProtocolEntry], limit: int
    ) -> List[ProtocolEntry]:
        """Take a deterministic, class-stratified slice of the entries.

        Args:
            entries: Full ordered entry list.
            limit: Maximum entries to retain.

        Returns:
            At most ``limit`` entries, preserving the class ratio and the
            original ordering.
        """
        if len(entries) <= limit:
            return entries

        stride = len(entries) / limit
        sampled = [entries[int(index * stride)] for index in range(limit)]
        logger.warning(
            f"  train capped at {limit:,} of {len(entries):,} clips "
            "(smoke-test mode; the reported EER is not a corpus result)"
        )
        return sampled
