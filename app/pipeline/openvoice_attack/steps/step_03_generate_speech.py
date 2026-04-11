"""
Step 3: Generate Synthetic Speech with OpenVoice

Generates synthetic Spanish voice cloning attacks using OpenVoice:
  1. MeloTTS (language='ES') synthesises text into a base Spanish voice.
  2. ToneColorConverter transfers the target speaker's tone colour onto the
     base voice using the speaker's reference audio embedding.

Both models are loaded once and reused across all samples. The ToneColorConverter
processes each speaker's reference audio once to extract target_se, then this
embedding is reused for all texts assigned to that speaker.

Output audio is resampled to SAMPLE_RATE (16 kHz) for consistency with other
pipeline stages.
"""
import json
import os
import tempfile
import time
import librosa
import soundfile as sf
import torch
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

from app.pipeline.openvoice_attack.settings import settings
from app.pipeline.openvoice_attack.schemas.generation_result import GenerationResult


class SpeechGenerator:
    """Generates synthetic speech using OpenVoice (MeloTTS + ToneColorConverter).

    Loads both models once at the beginning of execute() and releases GPU
    memory on completion. Per speaker, extracts the tone color embedding once
    from the reference audio. Per text, synthesises with MeloTTS and then
    applies tone color conversion.

    Attributes:
        output_dir: Directory where generated audio files are saved.
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize speech generator.

        Args:
            output_dir: Output directory (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR

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

        converter_config = str(
            settings.OPENVOICE_CHECKPOINT_DIR / "converter" / "config.json"
        )
        converter_ckpt = str(
            settings.OPENVOICE_CHECKPOINT_DIR / "converter" / "checkpoint.pth"
        )
        source_se_path = str(
            settings.OPENVOICE_CHECKPOINT_DIR / "base_speakers" / "ses" / "es.pth"
        )

        logger.info("Loading ToneColorConverter...")
        tone_color_converter = ToneColorConverter(converter_config, device=settings.DEVICE)
        tone_color_converter.load_ckpt(converter_ckpt)
        logger.info("Loading MeloTTS (ES)...")
        tts_model = TTS(language=settings.MELO_LANGUAGE, device=settings.DEVICE)
        speaker_ids = tts_model.hps.data.spk2id
        melo_speaker_id = speaker_ids[settings.MELO_LANGUAGE]

        source_se = torch.load(source_se_path, map_location=settings.DEVICE)

        logger.info("Models loaded successfully")

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
                    and (gen_dir / f"OPENVOICE_{speaker_id}_{p['text_id']}.wav").exists()
                    for p in speaker_prompts
                )
                if all_cached:
                    pbar.update(len(speaker_prompts))
                    continue

                try:
                    target_se, _ = se_extractor.get_se(
                        str(ref_path),
                        tone_color_converter,
                        vad=True,
                    )
                    logger.debug(f"Extracted tone color embedding for {speaker_id}")
                except Exception as e:
                    logger.error(f"Failed to extract tone color for {speaker_id}: {e}")
                    failed.extend([p["text_id"] for p in speaker_prompts])
                    pbar.update(len(speaker_prompts))
                    continue

                for prompt_data in speaker_prompts:
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"
                    output_path = gen_dir / f"OPENVOICE_{speaker_id}_{text_id}.wav"

                    if sample_id in generated and output_path.exists():
                        pbar.update(1)
                        continue

                    try:
                        generation_time, audio_duration = self._generate_single(
                            text=text,
                            tts_model=tts_model,
                            melo_speaker_id=melo_speaker_id,
                            tone_color_converter=tone_color_converter,
                            source_se=source_se,
                            target_se=target_se,
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

        del tts_model
        del tone_color_converter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory released")

        return GenerationResult(
            generated_samples_path=gen_metadata_path,
            total_generated=len(generated),
            failed_generations=failed,
            avg_rtf=avg_rtf,
        )

    def _generate_single(
        self,
        text: str,
        tts_model: "TTS",
        melo_speaker_id: int,
        tone_color_converter: "ToneColorConverter",
        source_se: torch.Tensor,
        target_se: torch.Tensor,
        output_path: Path,
    ) -> tuple:
        """Generate a single synthetic sample via MeloTTS + ToneColorConverter.

        Args:
            text: Spanish text to synthesise.
            tts_model: Loaded MeloTTS model instance.
            melo_speaker_id: Integer speaker ID for MeloTTS ES voice.
            tone_color_converter: Loaded ToneColorConverter instance.
            source_se: Base ES speaker tone color embedding (from checkpoint).
            target_se: Target speaker tone color embedding (from reference audio).
            output_path: Path where the final 16 kHz WAV file will be saved.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If TTS or conversion fails.
        """
        start_time = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_base:
            tmp_base_path = tmp_base.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_converted:
            tmp_converted_path = tmp_converted.name

        try:
            tts_model.tts_to_file(
                text,
                melo_speaker_id,
                tmp_base_path,
                speed=settings.MELO_SPEED,
            )

            tone_color_converter.convert(
                audio_src_path=tmp_base_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=tmp_converted_path,
                tau=settings.CONVERSION_TAU,
                message="@MyShell",
            )

            audio, _ = librosa.load(tmp_converted_path, sr=settings.SAMPLE_RATE)
            sf.write(str(output_path), audio, settings.SAMPLE_RATE)

        finally:
            if os.path.exists(tmp_base_path):
                os.unlink(tmp_base_path)
            if os.path.exists(tmp_converted_path):
                os.unlink(tmp_converted_path)

        generation_time = time.time() - start_time
        audio_duration = len(audio) / settings.SAMPLE_RATE

        return generation_time, audio_duration
