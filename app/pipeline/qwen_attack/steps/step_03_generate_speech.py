"""
Step 3: Generate Synthetic Speech

Generates synthetic Spanish speech using Qwen3-TTS local model inference.
Unlike FishGram (HTTP API), this step loads the model directly and generates
audio in-process. Uses speaker prompt reuse optimization to avoid redundant
computation when generating multiple utterances per speaker.
"""
import json
import time
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from qwen_tts import Qwen3TTSModel
from app.pipeline.qwen_attack.settings import settings
from app.pipeline.qwen_attack.schemas.generation_result import GenerationResult


class SpeechGenerator:
    """Generates synthetic speech using Qwen3-TTS local model.

    Loads the Qwen3-TTS model directly into GPU memory and generates
    synthetic speech for each speaker-text pair. Uses the speaker prompt
    reuse pattern: create_voice_clone_prompt() is called once per speaker,
    then the prompt is reused for all N utterances from that speaker.

    Attributes:
        output_dir: Directory where generated audio files are saved.
        model: Qwen3TTSModel instance, loaded during execute().
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize speech generator.

        Args:
            output_dir: Output directory (default: from settings).
        """
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.model = None

    def _load_model(self) -> Qwen3TTSModel:
        """Load Qwen3-TTS model to GPU.

        Loads the 1.7B Base model with the configured attention implementation
        (flash_attention_2 for speed, sdpa as fallback). The 0.6B model is
        explicitly avoided due to known dimension mismatch bugs.

        Returns:
            Initialized Qwen3TTSModel ready for inference.

        Raises:
            RuntimeError: If model loading fails (VRAM, CUDA, or download issues).
        """
        logger.info(f"Loading Qwen3-TTS model: {settings.QWEN_MODEL_ID}")
        logger.info(f"  Device: {settings.DEVICE}")
        logger.info(f"  Dtype: {settings.DTYPE}")
        logger.info(f"  Attention: {settings.QWEN_ATTN_IMPLEMENTATION}")

        start_time = time.time()

        dtype = getattr(torch, settings.DTYPE)
        model = Qwen3TTSModel.from_pretrained(
            settings.QWEN_MODEL_ID,
            device_map=settings.DEVICE,
            dtype=dtype,
            attn_implementation=settings.QWEN_ATTN_IMPLEMENTATION,
        )

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.1f}s")

        return model

    def _build_speaker_prompt(
        self,
        ref_audio_path: Path,
        ref_text: str
    ) -> dict:
        """Build reusable voice clone prompt for a speaker.

        Pre-computes speaker features from reference audio. This prompt
        is then reused for all utterances from the same speaker, avoiding
        redundant feature extraction.

        Args:
            ref_audio_path: Path to the speaker's reference audio file.
            ref_text: Transcript of the reference audio (from Whisper STT).

        Returns:
            Dictionary containing pre-computed speaker prompt features.

        Raises:
            RuntimeError: If prompt creation fails.
        """
        return self.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
            x_vector_only_mode=settings.X_VECTOR_ONLY_MODE,
        )

    def _generate_single(
        self,
        text: str,
        voice_clone_prompt: dict,
        output_path: Path
    ) -> tuple:
        """Generate a single synthetic audio sample using pre-built speaker prompt.

        Args:
            text: The Spanish text to synthesize.
            voice_clone_prompt: Pre-computed speaker prompt from _build_speaker_prompt.
            output_path: Path where the generated WAV file will be saved.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If generation fails.
        """
        start_time = time.time()

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=settings.QWEN_LANGUAGE,
            voice_clone_prompt=voice_clone_prompt,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            do_sample=True,
            top_k=settings.TOP_K,
            top_p=settings.TOP_P,
            temperature=settings.TEMPERATURE,
            repetition_penalty=settings.REPETITION_PENALTY,
            subtalker_dosample=True,
            subtalker_top_k=settings.SUBTALKER_TOP_K,
            subtalker_top_p=settings.SUBTALKER_TOP_P,
            subtalker_temperature=settings.SUBTALKER_TEMPERATURE,
        )

        generation_time = time.time() - start_time

        # Save generated audio
        audio = wavs[0]
        sf.write(str(output_path), audio, sr)

        audio_duration = len(audio) / sr

        return generation_time, audio_duration

    def execute(self) -> GenerationResult:
        """Generate synthetic speech for all speaker-text pairs.

        Loads the Qwen3-TTS model, then for each speaker builds a voice
        clone prompt once and generates all assigned utterances. Saves
        individual WAV files and a generation metadata JSON.

        Returns:
            GenerationResult with metadata path, counts, and statistics.

        Raises:
            RuntimeError: If model loading fails.
        """
        logger.info("Generating synthetic speech with Qwen3-TTS...")

        # Load model
        self.model = self._load_model()

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
                ref_data = references[speaker_id]
                ref_path = Path(ref_data["reference_path"])
                ref_text = ref_data.get("reference_text", "")
                split = ref_data["split"]

                # Verify reference audio exists
                if not ref_path.exists():
                    logger.error(
                        f"Reference audio not found for {speaker_id}: {ref_path}"
                    )
                    failed.extend([p["text_id"] for p in prompts.get(speaker_id, [])])
                    pbar.update(len(prompts.get(speaker_id, [])))
                    continue

                # Build speaker prompt once (reused for all utterances)
                try:
                    voice_clone_prompt = self._build_speaker_prompt(
                        ref_path, ref_text
                    )
                    logger.debug(f"Built voice clone prompt for {speaker_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to build speaker prompt for {speaker_id}: {e}"
                    )
                    failed.extend([p["text_id"] for p in prompts.get(speaker_id, [])])
                    pbar.update(len(prompts.get(speaker_id, [])))
                    continue

                # Generate for each text prompt
                for prompt_data in prompts.get(speaker_id, []):
                    text = prompt_data["text"]
                    text_id = prompt_data["text_id"]
                    sample_id = f"{speaker_id}_{text_id}"

                    try:
                        output_path = gen_dir / f"QWEN3TTS_{speaker_id}_{text_id}.wav"

                        generation_time, audio_duration = self._generate_single(
                            text=text,
                            voice_clone_prompt=voice_clone_prompt,
                            output_path=output_path,
                        )

                        rtf = generation_time / audio_duration if audio_duration > 0 else 0.0
                        rtf_values.append(rtf)

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

                        logger.debug(
                            f"Generated {sample_id}: {audio_duration:.1f}s "
                            f"in {generation_time:.1f}s (RTF={rtf:.2f})"
                        )

                    except Exception as e:
                        logger.error(f"Generation failed for {sample_id}: {e}")
                        failed.append(sample_id)

                    pbar.update(1)

        # Save generation metadata
        gen_metadata_path = self.output_dir / "generation_metadata.json"
        with open(gen_metadata_path, "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2, ensure_ascii=False)

        avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0.0

        logger.info(f"Generated {len(generated)} samples")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Average RTF: {avg_rtf:.2f}")
        logger.info(f"  Metadata saved to: {gen_metadata_path}")

        # Cleanup: release model from GPU
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory released")

        return GenerationResult(
            generated_samples_path=gen_metadata_path,
            total_generated=len(generated),
            failed_generations=failed,
            avg_rtf=avg_rtf
        )
