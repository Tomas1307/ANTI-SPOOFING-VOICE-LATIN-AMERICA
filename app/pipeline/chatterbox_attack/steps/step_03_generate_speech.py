"""
Step 3: Generate Synthetic Speech with Chatterbox Multilingual TTS

Generates synthetic Spanish voice cloning attacks using ChatterboxMultilingualTTS
(500M parameter flow-matching TTS with zero-shot voice cloning).

Key implementation decisions:
  - The Resemble Perth watermark is bypassed via NoOpWatermarker for research validity.
    Without this, every generated sample carries a neural steganographic watermark
    that could give anti-spoofing detectors an artificial advantage.
  - The perth_patcher module MUST be imported before chatterbox.mtl_tts because
    ChatterboxMultilingualTTS.__init__ calls perth.PerthImplicitWatermarker()
    unconditionally. The native Perth binary frequently fails to load (returns None),
    so we replace it with NoOpWatermarker at module level before the import.
  - Output is a torch.Tensor at 24 kHz (model.sr). It is resampled to 16 kHz
    before saving for consistency with all other pipeline stages.
  - The model is loaded once with from_pretrained(), then reused for all samples.
    GPU memory is released after generation completes.
"""
import json
import time
import torch
import torchaudio
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from app.pipeline.chatterbox_attack.settings import settings
from app.pipeline.chatterbox_attack.schemas.generation_result import GenerationResult
from app.pipeline.chatterbox_attack.utils.perth_patcher import ensure_patched  # noqa: F401 — patches perth on import
from app.pipeline.chatterbox_attack.utils.speech_trimmer import trim_trailing_noise
from app.pipeline.chatterbox_attack.utils.watermark_remover import NoOpWatermarker
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


class SpeechGenerator:
    """Generates synthetic speech using Chatterbox Multilingual TTS.

    Loads ChatterboxMultilingualTTS once, replaces the internal watermarker
    with a no-op implementation for research use, then generates all samples.
    Reuses the same model instance across all speakers and texts.

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
        logger.info("Step 3: Generating synthetic speech with Chatterbox Multilingual TTS...")
        logger.info(f"  Device: {settings.DEVICE}")
        logger.info(f"  Language: {settings.LANGUAGE_ID}")
        logger.info(f"  Exaggeration: {settings.EXAGGERATION}")
        logger.info(f"  CFG weight: {settings.CFG_WEIGHT}")
        logger.info(f"  Temperature: {settings.TEMPERATURE}")

        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        ref_metadata_path = self.output_dir / "reference_metadata.json"
        prompts_path = self.output_dir / "text_prompts.json"

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        logger.info("Loading ChatterboxMultilingualTTS from HuggingFace cache...")
        model = ChatterboxMultilingualTTS.from_pretrained(device=settings.DEVICE)
        model.watermarker = NoOpWatermarker()

        # Fix: transformers >= 4.47 rejects output_attentions=True with SDPA
        # attention. Chatterbox's internal GPT model uses output_attentions, so
        # force eager attention on all sub-modules that have SDPA configured.
        # ChatterboxMultilingualTTS is NOT an nn.Module, so walk its attributes
        # and recurse into any nn.Module children via named_modules().
        self._patch_sdpa_to_eager(model)

        logger.info("Model loaded; watermark bypassed for research use")

        generated = {}
        failed = []
        rtf_values = []

        total_pairs = sum(len(texts) for texts in prompts.values())
        logger.info(f"Generating {total_pairs} synthetic samples...")

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

                for prompt_data in prompts.get(speaker_id, []):
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"

                    try:
                        output_path = gen_dir / f"CHATTERBOX_{speaker_id}_{text_id}.wav"

                        generation_time, audio_duration = self._generate_single(
                            text=text,
                            model=model,
                            ref_path=ref_path,
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

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory released")

        return GenerationResult(
            generated_samples_path=gen_metadata_path,
            total_generated=len(generated),
            failed_generations=failed,
            avg_rtf=avg_rtf,
        )

    @staticmethod
    def _patch_sdpa_to_eager(model: ChatterboxMultilingualTTS) -> None:
        """Force eager attention on all internal transformers models.

        ChatterboxMultilingualTTS is a plain Python class (not nn.Module) that
        holds several nn.Module sub-models. This method walks all attributes,
        finds nn.Module instances, and patches any transformers config that uses
        SDPA attention to use eager instead. Required for transformers >= 4.47
        where output_attentions=True is incompatible with SDPA.

        Args:
            model: Loaded ChatterboxMultilingualTTS instance to patch in-place.
        """
        patched = 0
        for attr_name in vars(model):
            attr = getattr(model, attr_name)
            if not isinstance(attr, torch.nn.Module):
                continue
            for name, submodule in attr.named_modules():
                config = getattr(submodule, "config", None)
                if config is None:
                    continue
                if getattr(config, "_attn_implementation", None) == "sdpa":
                    config._attn_implementation = "eager"
                    config._attn_implementation_internal = "eager"
                    patched += 1
        if patched > 0:
            logger.info(f"Patched {patched} sub-module(s) from SDPA to eager attention")

    def _generate_single(
        self,
        text: str,
        model: ChatterboxMultilingualTTS,
        ref_path: Path,
        output_path: Path,
    ) -> tuple:
        """Generate a single synthetic sample using Chatterbox.

        Args:
            text: Spanish text to synthesise.
            model: Loaded ChatterboxMultilingualTTS instance with NoOpWatermarker.
            ref_path: Path to the speaker's reference audio file.
            output_path: Path where the 16 kHz WAV file will be saved.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If generation fails.
        """
        start_time = time.time()

        wav = model.generate(
            text=text,
            language_id=settings.LANGUAGE_ID,
            audio_prompt_path=str(ref_path),
            exaggeration=settings.EXAGGERATION,
            cfg_weight=settings.CFG_WEIGHT,
            temperature=settings.TEMPERATURE,
            repetition_penalty=settings.REPETITION_PENALTY,
        )

        wav_resampled = torchaudio.functional.resample(
            wav, model.sr, settings.SAMPLE_RATE
        )

        wav_trimmed = trim_trailing_noise(
            wav_resampled,
            settings.SAMPLE_RATE,
            margin_ms=settings.VAD_MARGIN_MS,
        )

        torchaudio.save(str(output_path), wav_trimmed, settings.SAMPLE_RATE)

        generation_time = time.time() - start_time
        audio_duration = wav_trimmed.shape[-1] / settings.SAMPLE_RATE

        return generation_time, audio_duration
