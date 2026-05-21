"""
Step 1: Transcribe Bonafide Audio

Transcribes each bonafide HABLA audio file using Parakeet TDT 0.6b-v3
to obtain text transcripts and word-level timestamps. Filters out
utterances with fewer than MIN_WORDS_W1 words (below W1 tier minimum).

The transcripts serve two purposes:
1. Input text for the voice cloning system (Step 2).
2. Word-level timestamps for forced alignment on the bonafide side (Step 3).

Two operating modes:

  Manifest mode (preferred for production). When a ManifestLoader is
  supplied, Step 1 short-circuits Parakeet: it loads the cached full
  transcripts produced by the manifest pre-flight, filters to the
  sample_keys assigned to this (attack, partition) cell, and writes
  the slice to bonafide_transcripts.json. No GPU work, no
  re-transcription, no partition shuffle (the manifest already encodes
  the partition).

  Legacy mode. When no ManifestLoader is supplied, Step 1 walks the
  bonafide root, applies the per-speaker not_jittered/jittered partition
  shuffle, and transcribes the resulting slice with Parakeet. Kept for
  ad-hoc runs and backward compatibility.
"""
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.transcription_result import TranscriptionResult
from app.pipeline.partial_spoof.utils.manifest_loader import ManifestLoader
from app.utils.parakeet_transcriber import ParakeetTranscriber


class BonafideTranscriber:
    """Transcribes bonafide HABLA audio using Parakeet TDT ASR.

    Two operating modes (see module docstring):
      Manifest mode: filter a pre-transcribed cache by the (attack,
        partition) slice. No GPU work.
      Legacy mode: walk the bonafide root, apply per-speaker partition,
        run Parakeet on the slice.

    Attributes:
        bonafide_dir: Root directory containing HABLA speaker subfolders.
        output_dir: Directory where bonafide_transcripts.json is saved.
        min_words: Minimum word count to include an utterance.
        manifest_loader: Optional ManifestLoader for manifest mode.
        manifest_attack: Attack key for slicing in manifest mode.
        manifest_partition: Partition key for slicing in manifest mode.
        cached_transcripts_path: Where to read the full-corpus transcripts
            JSON in manifest mode.
    """

    CACHED_TRANSCRIPTS_FILENAME = "bonafide_transcripts_full.json"

    def __init__(
        self,
        bonafide_dir: Path | None = None,
        output_dir: Path | None = None,
        min_words: int | None = None,
        manifest_loader: Optional[ManifestLoader] = None,
        manifest_attack: Optional[str] = None,
        manifest_partition: Optional[str] = None,
        cached_transcripts_path: Optional[Path] = None,
    ) -> None:
        """Initialize bonafide transcriber.

        Args:
            bonafide_dir: Directory with bonafide speakers (default: from settings).
            output_dir: Output directory (default: from settings).
            min_words: Minimum word count threshold (default: MIN_WORDS_W1 from settings).
            manifest_loader: Optional ManifestLoader. When provided, Step 1
                operates in manifest mode and short-circuits Parakeet.
            manifest_attack: Attack slice key (manifest mode only).
            manifest_partition: Partition slice key (manifest mode only).
            cached_transcripts_path: Path to the full-corpus transcripts JSON
                produced by the pre-flight (manifest mode only). Defaults
                to settings.MANIFEST_PATH.parent / CACHED_TRANSCRIPTS_FILENAME.
        """
        self.bonafide_dir = bonafide_dir or settings.BONAFIDE_DIR
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.min_words = min_words or settings.MIN_WORDS_W1
        self.manifest_loader = manifest_loader
        self.manifest_attack = manifest_attack
        self.manifest_partition = manifest_partition
        self.cached_transcripts_path = (
            cached_transcripts_path
            or settings.MANIFEST_PATH.parent / self.CACHED_TRANSCRIPTS_FILENAME
        )

    def execute(self) -> TranscriptionResult:
        """Transcribe (or slice cached) bonafide audio and save results.

        Branches on manifest_loader: in manifest mode, loads the cached
        full-corpus transcripts produced by the pre-flight and filters
        to the (manifest_attack, manifest_partition) slice without
        running Parakeet. In legacy mode, walks the bonafide root and
        runs Parakeet on the per-speaker not_jittered/jittered slice.

        Returns:
            TranscriptionResult with path to transcripts JSON and statistics.
        """
        if self.manifest_loader is not None:
            return self._execute_manifest_mode()
        return self._execute_legacy_mode()

    def _execute_manifest_mode(self) -> TranscriptionResult:
        """Filter the cached full transcripts by the manifest slice.

        Skips Parakeet entirely. Reads the cached transcripts JSON,
        intersects keys with the manifest slice for this (attack,
        partition), and writes the filtered subset to
        output_dir/bonafide_transcripts.json.

        Returns:
            TranscriptionResult populated from the slice.

        Raises:
            FileNotFoundError: If the cached transcripts JSON is missing.
            ValueError: If manifest_attack or manifest_partition is unset.
        """
        if self.manifest_attack is None or self.manifest_partition is None:
            raise ValueError(
                "Manifest mode requires both manifest_attack and "
                "manifest_partition to be set."
            )
        if not self.cached_transcripts_path.exists():
            raise FileNotFoundError(
                f"Cached transcripts not found at {self.cached_transcripts_path}. "
                "Run app/scripts/generate_partial_spoof_manifest.py first."
            )

        logger.info(
            f"Step 1 (manifest mode): slicing cached transcripts for "
            f"{self.manifest_attack}/{self.manifest_partition}."
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.cached_transcripts_path, "r", encoding="utf-8") as f:
            full_transcripts = json.load(f)

        slice_keys = set(
            self.manifest_loader.sample_keys(
                attack=self.manifest_attack,
                partition=self.manifest_partition,
            )
        )
        if not slice_keys:
            logger.warning(
                f"Manifest slice {self.manifest_attack}/{self.manifest_partition} "
                "is empty. Step 1 will write an empty transcripts file."
            )

        sliced = {
            key: entry for key, entry in full_transcripts.items()
            if key in slice_keys
        }
        missing = slice_keys - sliced.keys()
        if missing:
            logger.warning(
                f"{len(missing)} manifest keys missing from cached "
                f"transcripts (sample: {sorted(missing)[:5]})."
            )

        output_path = self.output_dir / "bonafide_transcripts.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sliced, f, ensure_ascii=False, indent=2)

        word_count_dist = self._compute_word_count_distribution(sliced)
        logger.info(
            f"Step 1 (manifest mode) complete: {len(sliced)} kept, "
            f"{len(slice_keys) - len(sliced)} missing from cache."
        )
        return TranscriptionResult(
            transcripts_path=output_path,
            total_transcribed=len(sliced),
            skipped_short=0,
            word_count_distribution=word_count_dist,
        )

    def _execute_legacy_mode(self) -> TranscriptionResult:
        """Walk the bonafide root and run Parakeet on the partition slice.

        Used when no ManifestLoader is supplied (ad-hoc runs, legacy
        callers). Applies the per-speaker not_jittered/jittered shuffle
        and filters utterances under MIN_WORDS_W1 words.

        Returns:
            TranscriptionResult with path to transcripts JSON and statistics.
        """
        logger.info("Step 1 (legacy mode): Transcribing bonafide audio with Parakeet TDT...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        transcriber = ParakeetTranscriber()
        transcriber.load(model_id=settings.PARAKEET_MODEL_ID, device=settings.DEVICE)

        speaker_dirs = self._get_speaker_dirs()
        logger.info(f"Processing {len(speaker_dirs)} speakers.")

        transcripts = {}
        total_transcribed = 0
        skipped_short = 0

        max_reached = False
        for speaker_dir in tqdm(speaker_dirs, desc="Transcribing speakers"):
            if max_reached:
                break
            speaker_id = speaker_dir.name
            for split in ["train", "val", "test"]:
                if max_reached:
                    break
                split_dir = speaker_dir / split
                if not split_dir.exists():
                    continue

                audio_files = sorted(
                    f for ext in ("*.wav", "*.flac", "*.mp3")
                    for f in split_dir.glob(ext)
                )
                audio_files = self._apply_partition(audio_files, speaker_id, split)
                for audio_path in audio_files:
                    if settings.MAX_SAMPLES > 0 and total_transcribed >= settings.MAX_SAMPLES:
                        max_reached = True
                        break

                    text, word_timestamps = transcriber.transcribe_with_timestamps(audio_path)
                    word_count = len(text.split())

                    if word_count < self.min_words:
                        skipped_short += 1
                        continue

                    sample_key = f"{speaker_id}_{audio_path.stem}"
                    transcripts[sample_key] = {
                        "speaker_id": speaker_id,
                        "split": split,
                        "audio_path": str(audio_path),
                        "transcript": text,
                        "word_count": word_count,
                        "word_timestamps": [
                            {"word": wt.word, "start": wt.start, "end": wt.end}
                            for wt in word_timestamps
                        ],
                    }
                    total_transcribed += 1

        output_path = self.output_dir / "bonafide_transcripts.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2)

        word_count_dist = self._compute_word_count_distribution(transcripts)

        logger.info(
            f"Step 1 complete: {total_transcribed} transcribed, "
            f"{skipped_short} skipped (< {self.min_words} words)."
        )

        return TranscriptionResult(
            transcripts_path=output_path,
            total_transcribed=total_transcribed,
            skipped_short=skipped_short,
            word_count_distribution=word_count_dist,
        )

    def _get_speaker_dirs(self) -> list:
        """Retrieve speaker directories, filtered by validation mode.

        Returns:
            List of Path objects for each speaker directory to process.
        """
        if settings.VALIDATION_MODE:
            dirs = []
            for speaker_id in settings.VALIDATION_SPEAKERS:
                speaker_dir = self.bonafide_dir / speaker_id
                if speaker_dir.exists():
                    dirs.append(speaker_dir)
                else:
                    logger.warning(f"Validation speaker not found: {speaker_dir}")
            return dirs

        return sorted([
            d for d in self.bonafide_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    def _apply_partition(
        self,
        audio_files: List[Path],
        speaker_id: str,
        split: str,
    ) -> List[Path]:
        """Apply per-speaker bonafide file partition for not_jittered/jittered split.

        Shuffles the file list deterministically using a seed derived from
        BONAFIDE_PARTITION_SEED and a stable speaker hash, then returns the
        first half (partition='not_jittered') or second half
        (partition='jittered').

        Speakers with only one file contribute that file to whichever
        partition the shuffle assigned it to. No speakers are discarded
        for having too few files; the partition naturally yields fewer
        files per speaker as the file count decreases.

        Args:
            audio_files: Sorted list of audio file paths for this speaker/split.
            speaker_id: HABLA speaker identifier (used to seed the shuffle).
            split: Dataset split label (used only for logging).

        Returns:
            Subset of audio_files corresponding to the active partition.
        """
        partition = settings.BONAFIDE_FILE_PARTITION
        if partition not in ("not_jittered", "jittered"):
            raise ValueError(
                f"BONAFIDE_FILE_PARTITION must be 'not_jittered' or 'jittered', "
                f"got '{partition}'."
            )

        if not audio_files:
            return audio_files

        speaker_hash = int.from_bytes(
            hashlib.sha256(speaker_id.encode("utf-8")).digest()[:4],
            byteorder="little",
        )
        rng = np.random.RandomState(settings.BONAFIDE_PARTITION_SEED + speaker_hash)
        shuffled_indices = rng.permutation(len(audio_files))

        half = len(shuffled_indices) // 2
        if partition == "not_jittered":
            selected_indices = sorted(shuffled_indices[:half].tolist())
        else:
            selected_indices = sorted(shuffled_indices[half:].tolist())

        selected_files = [audio_files[i] for i in selected_indices]

        logger.debug(
            f"Partition '{partition}' for {speaker_id}/{split}: "
            f"{len(selected_files)}/{len(audio_files)} files"
        )

        return selected_files

    def _compute_word_count_distribution(self, transcripts: dict) -> dict:
        """Compute word count distribution across tier buckets.

        Args:
            transcripts: Dictionary of transcript entries.

        Returns:
            Distribution mapping bucket labels to counts.
        """
        buckets = {"4-7": 0, "8-11": 0, "12+": 0}
        for entry in transcripts.values():
            wc = entry["word_count"]
            if wc >= 12:
                buckets["12+"] += 1
            elif wc >= 8:
                buckets["8-11"] += 1
            else:
                buckets["4-7"] += 1
        return buckets
