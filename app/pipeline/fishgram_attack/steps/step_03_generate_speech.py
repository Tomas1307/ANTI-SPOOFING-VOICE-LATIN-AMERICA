"""
Step 3: Generate Synthetic Speech

Generates synthetic Spanish speech using the Fish Speech HTTP API. The
per-sample cloning is delegated to the shared Cloner class in
``fishgram_attack/utils/cloner.py`` so the standalone attack pipeline
and ``partial_spoof/steps/step_02_generate_cloned_speech.py`` exercise
identical request/response code.

The Fish Speech server must be running on ml-server03 before executing
this step. See guide/how_to_run_fishgram/README.md for server startup
instructions.
"""
import json
from pathlib import Path

import librosa
from loguru import logger
from tqdm import tqdm

from app.pipeline.fishgram_attack.settings import settings
from app.pipeline.fishgram_attack.schemas.generation_result import GenerationResult
from app.pipeline.fishgram_attack.utils.cloner import Cloner


class SpeechGenerator:
    """Generates synthetic speech via the shared FishGram Cloner.

    Owns the outer loop (per-speaker, per-text iteration, resume,
    metadata recording, RTF stats). Delegates the per-sample HTTP request
    to ``Cloner.clone_single`` so the standalone attack pipeline and
    partial_spoof exercise the same Fish Speech API integration.

    Attributes:
        output_dir: Directory where generated audio files are saved.
        skip_existing: When True, skip WAVs that already exist (resume mode).
        cloner: FishGram Cloner instance, initialised during execute().
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        skip_existing: bool = False,
    ):
        """Initialize speech generator.

        Args:
            output_dir: Output directory (default: from settings).
            skip_existing: When True, skip WAV files that already exist on disk (for resume).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.skip_existing = skip_existing
        self.cloner = Cloner()

    def execute(self) -> GenerationResult:
        """Generate synthetic speech for all speaker-text pairs.

        Returns:
            GenerationResult with metadata path, counts, and statistics.

        Raises:
            ConnectionError: If the Fish Speech API server is not reachable.
        """
        logger.info("Generating synthetic speech...")
        if self.skip_existing:
            logger.info("  Resume mode: skip_existing=True (will skip already-generated WAVs)")

        # Verifies server health internally; raises ConnectionError on failure.
        self.cloner.load(device=settings.DEVICE)

        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        ref_metadata_path = self.output_dir / "reference_metadata.json"
        prompts_path = self.output_dir / "text_prompts.json"

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        generated = {}
        failed = []
        rtf_values = []

        total_pairs = sum(len(texts) for texts in prompts.values())
        logger.info(f"Generating {total_pairs} synthetic samples...")

        with tqdm(total=total_pairs, desc="Generating") as pbar:
            for speaker_id in sorted(references.keys()):
                ref_path = Path(references[speaker_id]["reference_path"])
                split = references[speaker_id]["split"]

                if not ref_path.exists():
                    logger.error(f"Reference audio not found for {speaker_id}: {ref_path}")
                    failed.extend([p["text_id"] for p in prompts[speaker_id]])
                    pbar.update(len(prompts[speaker_id]))
                    continue

                self.cloner.prepare_speaker(speaker_id, ref_path)

                for prompt_data in prompts[speaker_id]:
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"
                    output_path = (
                        gen_dir / f"{self.cloner.SYSTEM_ID}_{speaker_id}_{text_id}.wav"
                    )

                    if self.skip_existing and output_path.exists():
                        try:
                            synthetic_audio, _ = librosa.load(
                                output_path, sr=settings.SAMPLE_RATE
                            )
                            audio_duration = len(synthetic_audio) / settings.SAMPLE_RATE
                            generated[sample_id] = {
                                "speaker_id": speaker_id,
                                "text_id": text_id,
                                "text": text,
                                "audio_path": str(output_path),
                                "duration_seconds": audio_duration,
                                "generation_time_seconds": 0.0,
                                "rtf": 0.0,
                                "split": split,
                                "skipped_existing": True,
                            }
                            logger.debug(f"Skipping existing: {output_path.name}")
                        except Exception as e:
                            logger.warning(f"Existing file unreadable {output_path.name}: {e}")
                        pbar.update(1)
                        continue

                    try:
                        generation_time, audio_duration = self.cloner.clone_single(
                            text=text,
                            reference_audio_path=ref_path,
                            output_path=output_path,
                            reference_text="",
                        )

                        rtf = generation_time / audio_duration if audio_duration > 0 else 0.0
                        rtf_values.append(rtf)

                        generated[sample_id] = {
                            "speaker_id": speaker_id,
                            "text_id": text_id,
                            "text": text,
                            "audio_path": str(output_path),
                            "duration_seconds": audio_duration,
                            "generation_time_seconds": generation_time,
                            "rtf": rtf,
                            "split": split,
                        }

                        logger.debug(
                            f"Generated {sample_id}: {audio_duration:.1f}s "
                            f"in {generation_time:.1f}s (RTF={rtf:.2f})"
                        )

                    except Exception as e:
                        logger.error(f"Generation failed for {sample_id}: {e}")
                        failed.append(sample_id)

                    pbar.update(1)

        gen_metadata_path = self.output_dir / "generation_metadata.json"
        with open(gen_metadata_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2, ensure_ascii=False)

        avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0

        logger.info(f"Generated {len(generated)} samples")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Average RTF: {avg_rtf:.2f}")
        logger.info(f"  Metadata saved to: {gen_metadata_path}")

        self.cloner.cleanup()

        return GenerationResult(
            generated_samples_path=gen_metadata_path,
            total_generated=len(generated),
            failed_generations=failed,
            avg_rtf=avg_rtf,
        )
