"""
Step 3: Generate Synthetic Speech with OuteTTS

Generates synthetic Spanish voice cloning attacks using OuteTTS 0.6B
(Qwen-based LLM backbone with DAC codec for audio tokenization).

Key implementation details:
  - OuteTTS uses a fundamentally different approach from other TTS systems:
    it treats speech synthesis as a language modeling task, encoding audio
    as discrete tokens via the Descript Audio Codec (DAC).
  - Speaker cloning works by creating a speaker profile (JSON) from a
    reference audio file. This profile captures tempo, energy, pitch,
    and spectral centroid characteristics.
  - Generation is autoregressive and SLOW (~2-3 min per 10s audio on A40).
    The 8,192 token context window limits effective output to ~32s when
    the speaker profile is included.
  - Output is saved by the outetts library at its native sample rate,
    then resampled to 16 kHz for consistency with all other pipeline stages.
  - The model is loaded once via InterfaceHF, speaker profiles are cached
    per speaker, and GPU memory is released after all generation completes.
"""
import json
import time
import torch
import torchaudio
import outetts
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from app.pipeline.outetts_attack.settings import settings
from app.pipeline.outetts_attack.schemas.generation_result import GenerationResult


class SpeechGenerator:
    """Generates synthetic speech using OuteTTS 0.6B.

    Loads the OuteTTS model via InterfaceHF once, creates speaker profiles
    from reference audio per speaker, then generates all samples autoregressively.
    Speaker profiles are cached to avoid redundant extraction.

    OuteTTS is notably slower than other TTS systems in the attack suite due
    to its autoregressive LLM backbone (~2-3 min per 10s on A40).

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
        """
        logger.info("Step 3: Generating synthetic speech with OuteTTS...")
        logger.info(f"  Device          : {settings.DEVICE}")
        logger.info(f"  Model version   : {settings.OUTETTS_MODEL_VERSION}")
        logger.info(f"  Model ID        : {settings.OUTETTS_MODEL_ID}")
        logger.info(f"  Temperature     : {settings.TEMPERATURE}")
        logger.info(f"  Top-K           : {settings.TOP_K}")
        logger.info(f"  Top-P           : {settings.TOP_P}")
        logger.info(f"  Rep. penalty    : {settings.REPETITION_PENALTY}")
        logger.info(f"  Max length      : {settings.MAX_LENGTH}")

        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        ref_metadata_path = self.output_dir / "reference_metadata.json"
        prompts_path = self.output_dir / "text_prompts.json"

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        logger.info("Loading OuteTTS model via InterfaceHF...")
        interface = outetts.Interface(
            model_version=settings.OUTETTS_MODEL_VERSION,
            cfg=outetts.HFModelConfig_v2(
                model_path=settings.OUTETTS_MODEL_ID,
                tokenizer_path=settings.OUTETTS_MODEL_ID,
            ),
        )
        logger.info("OuteTTS model loaded successfully")

        generated = {}
        failed = []
        rtf_values = []
        speaker_profiles = {}

        total_pairs = sum(len(texts) for texts in prompts.values())
        logger.info(f"Generating {total_pairs} synthetic samples...")
        logger.info(
            "NOTE: OuteTTS is slow (~2-3 min per 10s audio). "
            "This will take a while."
        )

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

                if speaker_id not in speaker_profiles:
                    try:
                        speaker_profiles[speaker_id] = interface.create_speaker(
                            str(ref_path)
                        )
                        logger.debug(f"Created speaker profile for {speaker_id}")
                    except Exception as e:
                        logger.error(
                            f"Failed to create speaker profile for {speaker_id}: {e}"
                        )
                        failed.extend([p["text_id"] for p in prompts.get(speaker_id, [])])
                        pbar.update(len(prompts.get(speaker_id, [])))
                        continue

                speaker = speaker_profiles[speaker_id]

                for prompt_data in prompts.get(speaker_id, []):
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"

                    try:
                        output_path = gen_dir / f"OUTETTS_{speaker_id}_{text_id}.wav"

                        generation_time, audio_duration = self._generate_single(
                            text=text,
                            interface=interface,
                            speaker=speaker,
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

        del interface
        del speaker_profiles
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
        interface: outetts.Interface,
        speaker: object,
        output_path: Path,
    ) -> tuple:
        """Generate a single synthetic sample using OuteTTS.

        The outetts library saves audio at its native sample rate (determined
        by the DAC codec). After saving, the audio is resampled to 16 kHz
        for consistency with the rest of the pipeline.

        Args:
            text: Spanish text to synthesise.
            interface: Loaded OuteTTS InterfaceHF instance.
            speaker: Speaker profile object created via interface.create_speaker().
            output_path: Path where the final 16 kHz WAV file will be saved.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If generation or saving fails.
        """
        start_time = time.time()

        output = interface.generate(
            config=outetts.GenerationConfig(
                text=text,
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(
                    top_k=settings.TOP_K,
                    top_p=settings.TOP_P,
                    temperature=settings.TEMPERATURE,
                    repetition_penalty=settings.REPETITION_PENALTY,
                ),
                max_length=settings.MAX_LENGTH,
            )
        )

        temp_path = output_path.with_suffix(".tmp.wav")
        output.save(str(temp_path))

        wav, native_sr = torchaudio.load(str(temp_path))

        if native_sr != settings.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, native_sr, settings.SAMPLE_RATE)

        torchaudio.save(str(output_path), wav, settings.SAMPLE_RATE)

        temp_path.unlink(missing_ok=True)

        generation_time = time.time() - start_time
        audio_duration = wav.shape[-1] / settings.SAMPLE_RATE

        return generation_time, audio_duration
