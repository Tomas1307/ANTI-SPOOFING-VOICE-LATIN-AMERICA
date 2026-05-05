"""
Step 1: Prepare Reference Audio with Parakeet Transcription

Creates 10-second reference audio clips for each speaker by concatenating
training samples, then transcribes them using NVIDIA Parakeet TDT for use
as ref_text in OmniVoice voice cloning.

OmniVoice docs recommend 3-10s reference audio; longer clips degrade
cloning quality. We target 10s as the upper bound to capture sufficient
speaker characteristics.
"""
import json
from pathlib import Path

import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.pipeline.omnivoice_attack.settings import settings
from app.pipeline.omnivoice_attack.schemas.reference_result import ReferenceResult
from app.pipeline.omnivoice_attack.utils.audio_concatenation import concatenate_with_padding
from app.pipeline.omnivoice_attack.utils.reference_transcriber import transcribe_audio


class ReferenceAudioPreparator:
    """Prepares reference audio clips with Parakeet transcription for OmniVoice cloning.

    Concatenates multiple training samples per speaker to create 10-second
    reference clips, then transcribes them with NVIDIA Parakeet TDT. The
    transcripts are stored in metadata and used by Step 3 as the ref_text
    argument to OmniVoice.generate(), which improves cloning quality versus
    auto-transcription.

    Attributes:
        bonafide_dir: Directory containing HABLA bonafide speaker folders.
        output_dir: Directory where reference clips and metadata are saved.
        target_duration: Target duration in seconds for concatenated reference clips.
    """

    def __init__(
        self,
        bonafide_dir: Path | None = None,
        output_dir: Path | None = None,
        target_duration: float | None = None,
    ):
        """Initialize reference audio preparator.

        Args:
            bonafide_dir: Directory with bonafide speakers (default: from settings).
            output_dir: Output directory (default: from settings).
            target_duration: Target reference duration in seconds (default: from settings).
        """
        self.bonafide_dir = bonafide_dir or settings.BONAFIDE_DIR
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.target_duration = target_duration or settings.REFERENCE_DURATION_TARGET

    def execute(self) -> ReferenceResult:
        """Prepare reference audio with transcription for all speakers.

        For each speaker:
            1. Concatenates up to 5 training audio files into a 10s reference clip.
            2. Transcribes the reference clip using Parakeet TDT (Spanish).
            3. Stores metadata including the transcript for voice cloning.

        Returns:
            ReferenceResult with metadata path, speaker counts, and transcription stats.

        Raises:
            Exception: If reference preparation fails.
        """
        logger.info("Preparing reference audio clips with Parakeet transcription...")

        ref_dir = self.output_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)

        if settings.VALIDATION_MODE:
            speaker_dirs = [
                self.bonafide_dir / speaker_id
                for speaker_id in settings.VALIDATION_SPEAKERS
                if (self.bonafide_dir / speaker_id).exists()
            ]
            logger.info(f"VALIDATION MODE: Processing {len(speaker_dirs)} speakers")
        else:
            speaker_dirs = sorted([d for d in self.bonafide_dir.iterdir() if d.is_dir()])
            logger.info(f"PRODUCTION MODE: Processing {len(speaker_dirs)} speakers")

        references = {}
        split_counts = {"train": 0, "val": 0, "test": 0}

        for speaker_dir in tqdm(speaker_dirs, desc="Preparing references"):
            speaker_id = speaker_dir.name

            split = self._determine_split(speaker_dir)
            split_counts[split] += 1

            train_dir = speaker_dir / "train"
            if not train_dir.exists():
                logger.warning(f"No train directory for {speaker_id}, skipping")
                continue

            audio_files = sorted(
                f for ext in ("*.wav", "*.flac", "*.mp3")
                for f in train_dir.glob(ext)
            )[:5]

            if not audio_files:
                logger.warning(f"No audio files for {speaker_id}, skipping")
                continue

            try:
                reference_audio = concatenate_with_padding(
                    audio_files,
                    target_duration=self.target_duration,
                    sample_rate=settings.SAMPLE_RATE,
                )
            except Exception as e:
                logger.error(f"Failed to concatenate audio for {speaker_id}: {e}")
                continue

            ref_path = ref_dir / f"{speaker_id}_ref.wav"
            sf.write(ref_path, reference_audio, settings.SAMPLE_RATE)

            try:
                reference_text = transcribe_audio(ref_path, language="es")
                logger.debug(f"Transcribed {speaker_id}: '{reference_text[:80]}...'")
            except Exception as e:
                logger.error(f"Transcription failed for {speaker_id}: {e}")
                reference_text = ""

            bonafide_count = sum(
                len(list(speaker_dir.rglob(f"*.{ext}")))
                for ext in ("wav", "flac", "mp3")
            )

            references[speaker_id] = {
                "speaker_id": speaker_id,
                "reference_path": str(ref_path),
                "reference_text": reference_text,
                "duration_seconds": self.target_duration,
                "split": split,
                "source_files": [f.name for f in audio_files],
                "bonafide_count": bonafide_count,
            }

        metadata_path = self.output_dir / "reference_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(references, f, indent=2, ensure_ascii=False)

        transcribed_count = sum(
            1 for ref in references.values() if ref["reference_text"]
        )

        logger.info(f"Prepared {len(references)} reference clips")
        logger.info(f"  Transcribed: {transcribed_count}/{len(references)} speakers")
        logger.info(
            f"  Split breakdown: train={split_counts['train']}, "
            f"val={split_counts['val']}, test={split_counts['test']}"
        )
        logger.info(f"  Metadata saved to: {metadata_path}")

        return ReferenceResult(
            reference_metadata_path=metadata_path,
            reference_count=len(references),
            split_breakdown=split_counts,
            transcribed_count=transcribed_count,
        )

    def _determine_split(self, speaker_dir: Path) -> str:
        """Determine speaker's primary split based on file counts.

        Counts wav, flac, and mp3 files (mxm_* speakers from Mozilla
        Common Voice store audio as mp3, not wav).

        Args:
            speaker_dir: Speaker directory path.

        Returns:
            Split name ('train', 'val', or 'test').
        """
        train_count = self._count_audio_files(speaker_dir / "train")
        val_count = self._count_audio_files(speaker_dir / "val")
        test_count = self._count_audio_files(speaker_dir / "test")

        if train_count >= val_count and train_count >= test_count:
            return "train"
        elif val_count >= test_count:
            return "val"
        else:
            return "test"

    def _count_audio_files(self, directory: Path) -> int:
        """Count audio files (wav, flac, mp3) in a directory.

        Args:
            directory: Path to the directory to count files in.

        Returns:
            Total number of audio files found.
        """
        if not directory.exists():
            return 0
        return sum(
            len(list(directory.glob(f"*.{ext}")))
            for ext in ("wav", "flac", "mp3")
        )
