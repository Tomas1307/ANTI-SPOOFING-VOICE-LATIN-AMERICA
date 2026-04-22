"""
Step 2: Generate Cloned Speech

For each transcribed bonafide utterance, generates the same sentence
using the configured voice cloning attack strategy with the speaker's
reference audio. The cloned audio will be used in Step 3 for forced
alignment and in Step 5 for word-level extraction and splicing.
"""
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.cloned_generation_result import ClonedGenerationResult
from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.partial_spoof.utils.audio_concatenation import (
    concatenate_with_padding,
)


class ClonedSpeechGenerator:
    """Generates voice-cloned speech for bonafide utterances.

    Delegates generation to an AttackStrategy instance and manages
    reference audio preparation, output directory creation, and
    metadata recording.

    Attributes:
        output_dir: Base output directory for pipeline artifacts.
        strategy: The voice cloning attack strategy to use.
        reference_duration: Target duration for speaker reference clips.
        skip_existing: Skip generation for samples with existing output files.
    """

    def __init__(
        self,
        strategy: AttackStrategy,
        output_dir: Path | None = None,
        reference_duration: float | None = None,
        skip_existing: bool = False,
        seed_offset: int = 0,
        regenerate_keys: list | None = None,
    ) -> None:
        """Initialize cloned speech generator.

        Args:
            strategy: Voice cloning attack strategy instance.
            output_dir: Output directory (default: from settings).
            reference_duration: Target reference clip duration (default: from settings).
            skip_existing: Skip samples with existing output files.
            seed_offset: Offset added to TTS seed for regeneration rounds.
            regenerate_keys: If set, only regenerate these sample keys
                (deletes existing cloned files for these keys first).
        """
        self.strategy = strategy
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.reference_duration = reference_duration or settings.REFERENCE_DURATION_TARGET
        self.skip_existing = skip_existing
        self.seed_offset = seed_offset
        self.regenerate_keys = set(regenerate_keys) if regenerate_keys else None

    def execute(self) -> ClonedGenerationResult:
        """Generate cloned speech for all transcribed bonafide utterances.

        Loads the voice cloning model via the strategy, iterates over
        bonafide transcripts, generates cloned audio, and saves metadata.

        Returns:
            ClonedGenerationResult with generation statistics.
        """
        logger.info(f"Step 2: Generating cloned speech via {self.strategy.name()}...")

        cloned_dir = self.output_dir / "cloned"
        cloned_dir.mkdir(parents=True, exist_ok=True)
        refs_dir = self.output_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)

        transcripts_path = self.output_dir / "bonafide_transcripts.json"
        with open(transcripts_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        self.strategy.load_model(device=settings.DEVICE)

        generation_metadata = {}
        failed_generations = []
        total_rtf = 0.0
        total_generated = 0

        speakers = set(entry["speaker_id"] for entry in transcripts.values())
        reference_cache = {}

        for speaker_id in tqdm(sorted(speakers), desc="Preparing references"):
            ref_path = self._prepare_reference(speaker_id, refs_dir)
            if ref_path is not None:
                reference_cache[speaker_id] = ref_path

        for sample_key, entry in tqdm(transcripts.items(), desc="Generating cloned speech"):
            speaker_id = entry["speaker_id"]
            text = entry["transcript"]

            if self.regenerate_keys is not None and sample_key not in self.regenerate_keys:
                if sample_key in generation_metadata:
                    total_generated += 1
                continue

            if speaker_id not in reference_cache:
                failed_generations.append(sample_key)
                continue

            output_path = cloned_dir / f"{self.strategy.name()}_{sample_key}.wav"

            if self.regenerate_keys is not None and output_path.exists():
                output_path.unlink()
                logger.debug(f"Deleted old clone for regeneration: {sample_key}")

            if self.skip_existing and output_path.exists():
                info = sf.info(str(output_path))
                generation_metadata[sample_key] = {
                    "speaker_id": speaker_id,
                    "split": entry["split"],
                    "text": text,
                    "audio_path": str(output_path),
                    "duration_seconds": info.duration,
                    "generation_time_seconds": 0.0,
                    "rtf": 0.0,
                    "skipped_existing": True,
                }
                total_generated += 1
                continue

            try:
                ref_text = ""
                if self.strategy.needs_reference_transcript():
                    ref_text = self._get_reference_transcript(speaker_id)

                gen_time = self.strategy.generate(
                    text=text,
                    reference_audio_path=reference_cache[speaker_id],
                    output_path=output_path,
                    reference_text=ref_text,
                    seed=self.seed_offset + hash(sample_key) % (2**31) if self.seed_offset > 0 else None,
                )

                info = sf.info(str(output_path))
                rtf = gen_time / info.duration if info.duration > 0 else 0.0
                total_rtf += rtf
                total_generated += 1

                generation_metadata[sample_key] = {
                    "speaker_id": speaker_id,
                    "split": entry["split"],
                    "text": text,
                    "audio_path": str(output_path),
                    "duration_seconds": info.duration,
                    "generation_time_seconds": gen_time,
                    "rtf": rtf,
                    "skipped_existing": False,
                }

            except Exception as exc:
                logger.error(f"Generation failed for {sample_key}: {exc}")
                failed_generations.append(sample_key)

        self.strategy.cleanup()

        metadata_path = self.output_dir / "cloned_generation_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(generation_metadata, f, ensure_ascii=False, indent=2)

        avg_rtf = total_rtf / total_generated if total_generated > 0 else 0.0

        logger.info(
            f"Step 2 complete: {total_generated} generated, "
            f"{len(failed_generations)} failed, avg RTF={avg_rtf:.3f}"
        )

        return ClonedGenerationResult(
            metadata_path=metadata_path,
            total_generated=total_generated,
            failed_generations=failed_generations,
            avg_rtf=avg_rtf,
        )

    def _prepare_reference(self, speaker_id: str, refs_dir: Path) -> Path | None:
        """Prepare a reference audio clip for a speaker.

        Concatenates training samples to target duration using the shared
        audio concatenation utility from existing pipelines.

        Args:
            speaker_id: HABLA speaker identifier.
            refs_dir: Directory where reference clips are saved.

        Returns:
            Path to the reference WAV file, or None if no audio found.
        """
        ref_path = refs_dir / f"{speaker_id}_ref.wav"
        if ref_path.exists():
            return ref_path

        speaker_dir = settings.BONAFIDE_DIR / speaker_id / "train"
        if not speaker_dir.exists():
            logger.warning(f"No training data for speaker {speaker_id}")
            return None

        audio_files = sorted(
            f for ext in ("*.wav", "*.flac", "*.mp3")
            for f in speaker_dir.glob(ext)
        )
        if not audio_files:
            logger.warning(f"No audio files in {speaker_dir}")
            return None

        reference_audio = concatenate_with_padding(
            audio_files=audio_files,
            target_duration=self.reference_duration,
            sample_rate=settings.SAMPLE_RATE,
        )

        sf.write(str(ref_path), reference_audio, settings.SAMPLE_RATE)
        return ref_path

    def _get_reference_transcript(self, speaker_id: str) -> str:
        """Retrieve a reference transcript for systems that require it.

        Reads the first available transcript from the bonafide transcripts
        for the given speaker to serve as the reference text for voice
        cloning systems like Qwen and CosyVoice.

        Args:
            speaker_id: HABLA speaker identifier.

        Returns:
            Reference transcript text, or empty string if not found.
        """
        transcripts_path = self.output_dir / "bonafide_transcripts.json"
        with open(transcripts_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        for entry in transcripts.values():
            if entry["speaker_id"] == speaker_id:
                return entry["transcript"]
        return ""
