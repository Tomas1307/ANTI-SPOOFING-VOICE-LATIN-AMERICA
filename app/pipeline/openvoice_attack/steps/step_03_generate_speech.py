"""
Step 3: Generate Synthetic Speech with OpenVoice

Generates synthetic Spanish voice cloning attacks using OpenVoice
(MeloTTS + ToneColorConverter). The per-sample cloning is delegated to
the shared Cloner class in ``openvoice_attack/utils/cloner.py`` so the
exact same logic is reused by ``partial_spoof/steps/step_02_generate_cloned_speech.py``.

Output audio is resampled to SAMPLE_RATE (16 kHz) for consistency with
other pipeline stages.
"""
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from app.pipeline.openvoice_attack.settings import settings
from app.pipeline.openvoice_attack.schemas.generation_result import GenerationResult
from app.pipeline.openvoice_attack.utils.cloner import Cloner


class SpeechGenerator:
    """Generates synthetic speech via the shared OpenVoice Cloner.

    Owns the outer loop (per-speaker, per-text iteration, resume from
    on-disk generation_metadata.json, metadata recording, RTF stats).
    Delegates per-speaker target_se extraction and per-sample two-stage
    generation to the Cloner.

    Attributes:
        output_dir: Directory where generated audio files are saved.
        cloner: OpenVoice Cloner instance, initialised during execute().
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize speech generator.

        Args:
            output_dir: Output directory (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.cloner = Cloner()

    def execute(self) -> GenerationResult:
        """Generate synthetic speech for all speaker-text pairs.

        Returns:
            GenerationResult with metadata path, counts, and statistics.

        Raises:
            RuntimeError: If model loading fails.
            FileNotFoundError: If required checkpoint files are missing.
        """
        logger.info("Step 3: Generating synthetic speech with OpenVoice...")

        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        ref_metadata_path = self.output_dir / "reference_metadata.json"
        prompts_path = self.output_dir / "text_prompts.json"

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        self.cloner.load(device=settings.DEVICE)

        gen_metadata_path = self.output_dir / "generation_metadata.json"
        if gen_metadata_path.exists():
            with open(gen_metadata_path, "r", encoding="utf-8") as f:
                generated = json.load(f)
            logger.info(f"Resuming from checkpoint: {len(generated)} samples already generated")
        else:
            generated = {}

        failed = []
        rtf_values = []

        total_pairs = sum(len(texts) for texts in prompts.values())
        logger.info(f"Generating {total_pairs} synthetic samples ({len(generated)} cached)...")

        with tqdm(total=total_pairs, desc="Generating") as pbar:
            for speaker_id in sorted(references.keys()):
                ref_data = references[speaker_id]
                ref_path = Path(ref_data["reference_path"])
                split = ref_data["split"]

                if not ref_path.exists():
                    logger.error(f"Reference audio not found for {speaker_id}: {ref_path}")
                    failed.extend([p["text_id"] for p in prompts.get(speaker_id, [])])
                    pbar.update(len(prompts.get(speaker_id, [])))
                    continue

                speaker_prompts = prompts.get(speaker_id, [])
                all_cached = all(
                    f"{speaker_id}_{p['text_id']}" in generated
                    and (gen_dir / f"{self.cloner.SYSTEM_ID}_{speaker_id}_{p['text_id']}.wav").exists()
                    for p in speaker_prompts
                )
                if all_cached:
                    pbar.update(len(speaker_prompts))
                    continue

                try:
                    self.cloner.prepare_speaker(speaker_id, ref_path)
                except Exception as e:
                    logger.error(f"Failed to extract tone color for {speaker_id}: {e}")
                    failed.extend([p["text_id"] for p in speaker_prompts])
                    pbar.update(len(speaker_prompts))
                    continue

                for prompt_data in speaker_prompts:
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"
                    output_path = (
                        gen_dir / f"{self.cloner.SYSTEM_ID}_{speaker_id}_{text_id}.wav"
                    )

                    if sample_id in generated and output_path.exists():
                        pbar.update(1)
                        continue

                    try:
                        generation_time, audio_duration = self.cloner.clone_single(
                            text=text,
                            reference_audio_path=ref_path,
                            output_path=output_path,
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

                        with open(gen_metadata_path, "w", encoding="utf-8") as f:
                            json.dump(generated, f, indent=2, ensure_ascii=False)

                        logger.debug(
                            f"Generated {sample_id}: {audio_duration:.1f}s "
                            f"in {generation_time:.1f}s (RTF={rtf:.2f})"
                        )

                    except Exception as e:
                        logger.error(f"Generation failed for {sample_id}: {e}")
                        failed.append(sample_id)

                    pbar.update(1)

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
