"""
Step 4: Generate Synthetic Speech

Generates synthetic Spanish speech using Fish Speech TTS model.
"""
import json
import time
import librosa
import soundfile as sf
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import Any
from app.pipeline.fishgram_attack.settings import settings
from app.pipeline.fishgram_attack.schemas.generation_result import GenerationResult


class SpeechGenerator:
    """Generates synthetic speech using Fish Speech TTS.

    Processes all speaker-text pairs and generates synthetic audio
    with voice cloning from reference samples.
    """

    def __init__(
        self,
        model: Any,
        output_dir: Path | None = None
    ):
        """Initialize speech generator.

        Args:
            model: Loaded Fish Speech model instance (from Step 1)
            output_dir: Output directory (default: from settings)
        """
        self.model = model
        self.output_dir = output_dir or settings.OUTPUT_DIR

    def execute(self) -> GenerationResult:
        """Generate synthetic speech for all speaker-text pairs.

        Returns:
            GenerationResult with metadata path, counts, and statistics

        Raises:
            Exception: If generation fails
        """
        logger.info("Generating synthetic speech...")

        # Create output directory
        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        ref_metadata_path = self.output_dir / "reference_metadata.json"
        prompts_path = self.output_dir / "text_prompts.json"

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        # Generate speech for each speaker-text pair
        generated = {}
        failed = []
        rtf_values = []

        total_pairs = sum(len(texts) for texts in prompts.values())
        logger.info(f"Generating {total_pairs} synthetic samples...")

        with tqdm(total=total_pairs, desc="Generating") as pbar:
            for speaker_id in sorted(references.keys()):
                ref_path = Path(references[speaker_id]["reference_path"])
                split = references[speaker_id]["split"]

                # Load reference audio
                try:
                    ref_audio, sr = librosa.load(ref_path, sr=settings.SAMPLE_RATE)
                except Exception as e:
                    logger.error(f"Failed to load reference for {speaker_id}: {e}")
                    failed.extend([p["text_id"] for p in prompts[speaker_id]])
                    pbar.update(len(prompts[speaker_id]))
                    continue

                # Generate for each text prompt
                for prompt_data in prompts[speaker_id]:
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"

                    try:
                        # Generate speech
                        start_time = time.time()

                        # TODO: Replace with actual Fish Speech API
                        # synthetic_audio = self.model.generate(
                        #     reference_audio=ref_audio,
                        #     text=text,
                        #     sample_rate=settings.SAMPLE_RATE,
                        #     language="es"
                        # )

                        # Placeholder: Return silence
                        import numpy as np
                        synthetic_audio = np.zeros(int(5.0 * settings.SAMPLE_RATE))  # 5 seconds

                        generation_time = time.time() - start_time
                        audio_duration = len(synthetic_audio) / settings.SAMPLE_RATE
                        rtf = generation_time / audio_duration if audio_duration > 0 else 0.0
                        rtf_values.append(rtf)

                        # Save synthetic audio
                        output_path = gen_dir / f"FISHGRAM_{speaker_id}_{text_id}.wav"
                        sf.write(output_path, synthetic_audio, settings.SAMPLE_RATE)

                        # Store metadata
                        generated[sample_id] = {
                            "speaker_id": speaker_id,
                            "text_id": text_id,
                            "text": text,
                            "audio_path": str(output_path),
                            "duration_seconds": audio_duration,
                            "generation_time_seconds": generation_time,
                            "rtf": rtf,
                            "split": split
                        }

                    except Exception as e:
                        logger.error(f"Generation failed for {sample_id}: {e}")
                        failed.append(sample_id)

                    pbar.update(1)

        # Save generation metadata
        gen_metadata_path = self.output_dir / "generation_metadata.json"
        with open(gen_metadata_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2, ensure_ascii=False)

        avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0

        logger.info(f"✓ Generated {len(generated)} samples")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Average RTF: {avg_rtf:.2f}")
        logger.info(f"  Metadata saved to: {gen_metadata_path}")

        return GenerationResult(
            generated_samples_path=gen_metadata_path,
            total_generated=len(generated),
            failed_generations=failed,
            avg_rtf=avg_rtf
        )
