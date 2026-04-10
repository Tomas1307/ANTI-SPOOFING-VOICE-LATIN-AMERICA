"""
Step 3: Generate Synthetic Speech with CosyVoice 3.0

Generates synthetic Spanish voice cloning attacks using CosyVoice2
(Alibaba FunAudioLLM, Conditional Flow Matching, trained on 1M hours).

Key implementation decisions:
  - CosyVoice is not pip-installable. The cloned repo path and its
    Matcha-TTS submodule are added to sys.path BEFORE importing any
    cosyvoice modules. This is an intentional exception to the standard
    import pattern required by the project's non-pip dependency.
  - Zero-shot voice cloning mode is used: inference_zero_shot(text,
    prompt_text, prompt_speech, stream=False). This requires both a
    reference audio waveform AND a text transcription of that reference.
  - The inference returns a generator; we iterate it to collect the
    final tts_speech tensor.
  - Output is a torch.Tensor at the model's native sample rate. It is
    resampled to 16 kHz before saving for consistency with all other
    pipeline stages.
  - The model is loaded once with CosyVoice2(), then reused for all
    samples. GPU memory is released after generation completes.
"""
import json
import sys
import time
import torch
import torchaudio
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from app.pipeline.cosyvoice_attack.settings import settings
from app.pipeline.cosyvoice_attack.schemas.generation_result import GenerationResult

# CosyVoice is not pip-installable; it must be imported from the cloned repo.
# The repo path and its Matcha-TTS submodule are prepended to sys.path so that
# the cosyvoice package and its transitive dependencies resolve correctly.
sys.path.insert(0, str(settings.COSYVOICE_REPO_PATH))
sys.path.insert(0, str(settings.COSYVOICE_REPO_PATH / "third_party" / "Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice2


class SpeechGenerator:
    """Generates synthetic speech using CosyVoice 3.0 zero-shot voice cloning.

    Loads CosyVoice2 once from the local pretrained model directory, then
    generates all samples using inference_zero_shot() which requires both
    the reference audio waveform and its text transcription.

    The model directory is resolved relative to COSYVOICE_REPO_PATH so that
    CosyVoice can locate its config files and checkpoint weights.

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
            RuntimeError: If model loading fails or CosyVoice repo is not found.
        """
        logger.info("Step 3: Generating synthetic speech with CosyVoice 3.0...")
        logger.info(f"  Device: {settings.DEVICE}")
        logger.info(f"  CosyVoice repo: {settings.COSYVOICE_REPO_PATH}")
        logger.info(f"  Model dir: {settings.COSYVOICE_MODEL_DIR}")
        logger.info(f"  Load JIT: {settings.COSYVOICE_LOAD_JIT}")
        logger.info(f"  Load TRT: {settings.COSYVOICE_LOAD_TRT}")

        gen_dir = self.output_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        ref_metadata_path = self.output_dir / "reference_metadata.json"
        prompts_path = self.output_dir / "text_prompts.json"

        with open(ref_metadata_path, "r", encoding="utf-8") as f:
            references = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        model_path = str(settings.COSYVOICE_REPO_PATH / settings.COSYVOICE_MODEL_DIR)
        logger.info(f"Loading CosyVoice2 from {model_path}...")
        model = CosyVoice2(
            model_path,
            load_jit=settings.COSYVOICE_LOAD_JIT,
            load_trt=settings.COSYVOICE_LOAD_TRT,
        )
        logger.info("CosyVoice2 model loaded successfully")

        generated = {}
        failed = []
        rtf_values = []

        total_pairs = sum(len(texts) for texts in prompts.values())
        logger.info(f"Generating {total_pairs} synthetic samples...")

        with tqdm(total=total_pairs, desc="Generating") as pbar:
            for speaker_id in sorted(references.keys()):
                ref_data = references[speaker_id]
                ref_path = Path(ref_data["reference_path"])
                ref_text = ref_data.get("reference_text", "")
                split = ref_data["split"]

                if not ref_path.exists():
                    logger.error(f"Reference audio not found for {speaker_id}: {ref_path}")
                    failed.extend([p["text_id"] for p in prompts.get(speaker_id, [])])
                    pbar.update(len(prompts.get(speaker_id, [])))
                    continue

                if not ref_text:
                    logger.warning(
                        f"No reference transcription for {speaker_id}; "
                        f"zero-shot quality may be degraded"
                    )

                prompt_wav_path = str(ref_path)

                for prompt_data in prompts.get(speaker_id, []):
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"

                    try:
                        output_path = gen_dir / f"COSYVOICE_{speaker_id}_{text_id}.wav"

                        generation_time, audio_duration = self._generate_single(
                            text=text,
                            prompt_text=ref_text,
                            prompt_wav_path=prompt_wav_path,
                            model=model,
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

    def _generate_single(
        self,
        text: str,
        prompt_text: str,
        prompt_wav_path: str,
        model: CosyVoice2,
        output_path: Path,
    ) -> tuple:
        """Generate a single synthetic sample using CosyVoice zero-shot cloning.

        The inference_zero_shot method returns a generator. We iterate through it
        to collect the final tts_speech tensor, then resample from the model's
        native sample rate to the target 16 kHz.

        CosyVoice2.inference_zero_shot expects a file path string for prompt_wav,
        not a pre-loaded tensor. It calls load_wav internally.

        Args:
            text: Spanish text to synthesise.
            prompt_text: Transcription of the reference audio (for zero-shot conditioning).
            prompt_wav_path: File path to reference audio WAV (CosyVoice loads it internally).
            model: Loaded CosyVoice2 instance.
            output_path: Path where the 16 kHz WAV file will be saved.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If generation fails or produces no output.
        """
        start_time = time.time()

        tts_audio = None
        for result in model.inference_zero_shot(
            text, prompt_text, prompt_wav_path, stream=False
        ):
            tts_audio = result["tts_speech"]

        if tts_audio is None:
            raise RuntimeError(
                f"CosyVoice inference_zero_shot returned no results for text: "
                f"'{text[:60]}...'"
            )

        if tts_audio.dim() == 1:
            tts_audio = tts_audio.unsqueeze(0)

        model_sr = getattr(model, "sample_rate", 22050)
        if model_sr != settings.SAMPLE_RATE:
            tts_audio = torchaudio.functional.resample(
                tts_audio, model_sr, settings.SAMPLE_RATE
            )

        torchaudio.save(str(output_path), tts_audio.cpu(), settings.SAMPLE_RATE)

        generation_time = time.time() - start_time
        audio_duration = tts_audio.shape[-1] / settings.SAMPLE_RATE

        return generation_time, audio_duration
