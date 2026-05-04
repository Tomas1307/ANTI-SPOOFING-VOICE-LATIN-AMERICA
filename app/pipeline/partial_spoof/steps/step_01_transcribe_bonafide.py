"""
Step 1: Transcribe Bonafide Audio

Transcribes each bonafide HABLA audio file using Parakeet TDT 0.6b-v3
to obtain text transcripts and word-level timestamps. Filters out
utterances with fewer than MIN_WORDS_W1 words (below W1 tier minimum).

The transcripts serve two purposes:
1. Input text for the voice cloning system (Step 2).
2. Word-level timestamps for forced alignment on the bonafide side (Step 3).

Supports per-speaker bonafide file partition for the boundary-jitter dataset.
When BONAFIDE_FILE_PARTITION is 'main' or 'jitter', the file list per
speaker is shuffled deterministically and split 50/50, ensuring main and
jitter datasets use disjoint utterances.
"""
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.transcription_result import TranscriptionResult
from app.utils.parakeet_transcriber import ParakeetTranscriber


class BonafideTranscriber:
    """Transcribes bonafide HABLA audio using Parakeet TDT ASR.

    Iterates over all speakers (or validation subset), transcribes each
    audio file, records word-level timestamps from the native TDT output,
    and filters utterances by minimum word count for tier eligibility.

    Attributes:
        bonafide_dir: Root directory containing HABLA speaker subfolders.
        output_dir: Directory where bonafide_transcripts.json is saved.
        min_words: Minimum word count to include an utterance.
    """

    def __init__(
        self,
        bonafide_dir: Path | None = None,
        output_dir: Path | None = None,
        min_words: int | None = None,
    ) -> None:
        """Initialize bonafide transcriber.

        Args:
            bonafide_dir: Directory with bonafide speakers (default: from settings).
            output_dir: Output directory (default: from settings).
            min_words: Minimum word count threshold (default: MIN_WORDS_W1 from settings).
        """
        self.bonafide_dir = bonafide_dir or settings.BONAFIDE_DIR
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.min_words = min_words or settings.MIN_WORDS_W1

    def execute(self) -> TranscriptionResult:
        """Transcribe all bonafide audio files and save results.

        Loads the Parakeet TDT model, iterates over speaker directories
        and their train/val/test splits, transcribes each WAV file, and
        saves transcripts with word timestamps to JSON.

        Returns:
            TranscriptionResult with path to transcripts JSON and statistics.
        """
        logger.info("Step 1: Transcribing bonafide audio with Parakeet TDT...")
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
        """Apply per-speaker bonafide file partition for main/jitter split.

        Shuffles the file list deterministically using a seed derived from
        BONAFIDE_PARTITION_SEED and a stable speaker hash, then returns the
        first half (partition='main') or second half (partition='jitter').

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
        if partition not in ("main", "jitter"):
            raise ValueError(
                f"BONAFIDE_FILE_PARTITION must be 'main' or 'jitter', got '{partition}'."
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
        if partition == "main":
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
