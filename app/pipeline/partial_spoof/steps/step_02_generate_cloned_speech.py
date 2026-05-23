"""
Step 2: Generate Cloned Speech

For each transcribed bonafide utterance, generates the same sentence
using the configured voice cloning attack via the shared per-attack
Cloner class (the same class consumed by ``<attack>_attack/steps/
step_03_generate_speech.py``). The cloned audio will be used in Step 3
for forced alignment and in Step 5 for word-level extraction and
splicing.

Resumable via the optional CheckpointManager: every successful clone
commit calls checkpoint.mark_cloned(sample_key) before moving on, so a
killed run loses at most the in-flight sample. Generation errors are
recorded as recoverable failures and retried up to MAX_GENERATION_RETRIES
on subsequent runs; quality failures are NEVER filtered at this layer
(keep-bad-stuff principle).
"""
import json
from pathlib import Path
from typing import Optional

import soundfile as sf
from loguru import logger
from tqdm import tqdm

from app.pipeline.partial_spoof.settings import settings
from app.pipeline.partial_spoof.schemas.cloned_generation_result import ClonedGenerationResult
from app.pipeline.partial_spoof.utils.audio_concatenation import (
    concatenate_with_padding,
)
from app.pipeline.partial_spoof.utils.checkpoint_manager import CheckpointManager
from app.pipeline.partial_spoof.utils.cloner_dispatcher import get_cloner_class


class ClonedSpeechGenerator:
    """Generates voice-cloned speech for bonafide utterances.

    Resolves the per-attack Cloner class via the dispatcher (one source
    of truth shared with the standalone attack pipeline), prepares
    per-speaker reference audio, then loops over bonafide transcripts
    invoking ``cloner.prepare_speaker`` and ``cloner.clone_single`` per
    the documented contract.

    Attributes:
        output_dir: Base output directory for pipeline artifacts.
        cloner: BaseCloner subclass instance resolved at construction time.
        reference_duration: Target duration for speaker reference clips.
        skip_existing: Skip generation for samples with existing output files.
        checkpoint: Optional CheckpointManager for per-clone resume.
    """

    def __init__(
        self,
        attack_system: str,
        output_dir: Path | None = None,
        reference_duration: float | None = None,
        skip_existing: bool = False,
        seed_offset: int = 0,
        regenerate_keys: list | None = None,
        checkpoint: Optional[CheckpointManager] = None,
    ) -> None:
        """Initialize cloned speech generator.

        Args:
            attack_system: Attack identifier ('chatterbox', 'qwen',
                'fishgram', 'openvoice', 'outetts', 'omnivoice'). The
                matching Cloner class is resolved via the dispatcher.
            output_dir: Output directory (default: from settings).
            reference_duration: Target reference clip duration (default: from settings).
            skip_existing: Skip samples with existing output files.
            seed_offset: Offset added to TTS seed for regeneration rounds.
            regenerate_keys: If set, only regenerate these sample keys
                (deletes existing cloned files for these keys first).
            checkpoint: Optional CheckpointManager. When provided, Step 2
                marks every successful clone for resume and records
                recoverable failures with retry counters.

        Raises:
            ValueError: If attack_system is not recognised by the dispatcher.
        """
        cloner_cls = get_cloner_class(attack_system)
        self.cloner = cloner_cls()
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.reference_duration = reference_duration or settings.REFERENCE_DURATION_TARGET
        self.skip_existing = skip_existing
        self.seed_offset = seed_offset
        self.regenerate_keys = set(regenerate_keys) if regenerate_keys else None
        self.checkpoint = checkpoint

    def execute(self) -> ClonedGenerationResult:
        """Generate cloned speech for all transcribed bonafide utterances.

        Loads the voice cloning model via the Cloner, iterates over
        bonafide transcripts, generates cloned audio, and saves metadata.

        Returns:
            ClonedGenerationResult with generation statistics.
        """
        logger.info(
            f"Step 2: Generating cloned speech via {self.cloner.SYSTEM_ID}..."
        )

        cloned_dir = self.output_dir / "cloned"
        cloned_dir.mkdir(parents=True, exist_ok=True)
        refs_dir = self.output_dir / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)

        transcripts_path = self.output_dir / "bonafide_transcripts.json"
        with open(transcripts_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        self.cloner.load(device=settings.DEVICE)

        generation_metadata = {}
        failed_generations = []
        total_rtf = 0.0
        total_generated = 0

        speakers = set(entry["speaker_id"] for entry in transcripts.values())
        reference_cache = {}
        prepared_speakers: set = set()

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

            if (
                self.checkpoint is not None
                and self.checkpoint.is_abandoned(
                    sample_key, settings.MAX_GENERATION_RETRIES
                )
            ):
                logger.debug(
                    f"Skipping abandoned sample (retries exhausted): {sample_key}"
                )
                failed_generations.append(sample_key)
                continue

            if speaker_id not in reference_cache:
                failed_generations.append(sample_key)
                continue

            ref_path = reference_cache[speaker_id]
            output_path = (
                cloned_dir / f"{self.cloner.SYSTEM_ID}_{sample_key}.wav"
            )

            if self.regenerate_keys is not None and output_path.exists():
                output_path.unlink()
                logger.debug(f"Deleted old clone for regeneration: {sample_key}")

            checkpoint_says_done = (
                self.checkpoint is not None
                and self.checkpoint.is_cloned(sample_key)
            )
            if (
                self.skip_existing
                and (output_path.exists() or checkpoint_says_done)
                and output_path.exists()
            ):
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

            if speaker_id not in prepared_speakers:
                try:
                    ref_text_for_prep = ""
                    if self.cloner.NEEDS_REFERENCE_TRANSCRIPT:
                        ref_text_for_prep = self._get_reference_transcript(speaker_id)
                    self.cloner.prepare_speaker(
                        speaker_id=speaker_id,
                        reference_audio_path=ref_path,
                        reference_text=ref_text_for_prep,
                    )
                    prepared_speakers.add(speaker_id)
                except Exception as exc:
                    logger.error(
                        f"Failed to prepare speaker {speaker_id}: {exc}"
                    )
                    failed_generations.append(sample_key)
                    continue

            try:
                ref_text = ""
                if self.cloner.NEEDS_REFERENCE_TRANSCRIPT:
                    ref_text = self._get_reference_transcript(speaker_id)

                gen_time, gen_duration = self.cloner.clone_single(
                    text=text,
                    reference_audio_path=ref_path,
                    output_path=output_path,
                    reference_text=ref_text,
                    seed=self.seed_offset + hash(sample_key) % (2**31) if self.seed_offset > 0 else None,
                )

                if gen_duration < settings.MIN_CLONE_DURATION_S:
                    # Diffusion-based TTS (e.g., OmniVoice) occasionally
                    # terminates generation at zero or near-zero length.
                    # Empty / sub-half-second clones break downstream
                    # ECAPA and alignment. Delete the WAV and raise so the
                    # outer except branch routes this through the
                    # recoverable-retry path with a bumped seed.
                    try:
                        output_path.unlink()
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"Clone duration {gen_duration:.3f}s below "
                        f"MIN_CLONE_DURATION_S={settings.MIN_CLONE_DURATION_S}s "
                        "-- degenerate generation"
                    )

                rtf = gen_time / gen_duration if gen_duration > 0 else 0.0
                total_rtf += rtf
                total_generated += 1

                generation_metadata[sample_key] = {
                    "speaker_id": speaker_id,
                    "split": entry["split"],
                    "text": text,
                    "audio_path": str(output_path),
                    "duration_seconds": gen_duration,
                    "generation_time_seconds": gen_time,
                    "rtf": rtf,
                    "skipped_existing": False,
                }

                if self.checkpoint is not None:
                    self.checkpoint.mark_cloned(sample_key)

            except Exception as exc:
                logger.error(f"Generation failed for {sample_key}: {exc}")
                failed_generations.append(sample_key)
                if self.checkpoint is not None:
                    retries = self.checkpoint.record_failure(
                        sample_key,
                        CheckpointManager.truncate_error(exc),
                    )
                    if retries > settings.MAX_GENERATION_RETRIES:
                        logger.warning(
                            f"Sample {sample_key} exceeded "
                            f"MAX_GENERATION_RETRIES={settings.MAX_GENERATION_RETRIES}; "
                            "marked abandoned."
                        )

        self.cloner.cleanup()

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
        audio concatenation utility (now sourced from app/utils/).

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
        cloning systems like Qwen and OmniVoice.

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
