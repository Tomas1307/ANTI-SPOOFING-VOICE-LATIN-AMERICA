"""
Qwen3-TTS voice cloning unit -- single source of truth.

This module exposes a Cloner class that encapsulates EXACTLY the per-sample
cloning logic for Qwen3-TTS. It is consumed by both:

    1. qwen_attack/steps/step_03_generate_speech.py (the standalone
       full-utterance attack pipeline).
    2. partial_spoof/steps/step_02_generate_cloned_speech.py (via the
       Cloner dispatcher).

Qwen3-TTS uses a per-speaker prompt-reuse optimisation: the call to
``model.create_voice_clone_prompt`` is expensive (extracts speaker
features from the reference audio), so we do it once per speaker via
``prepare_speaker`` and reuse the resulting dict for every utterance
assigned to that speaker.
"""
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from loguru import logger
from qwen_tts import Qwen3TTSModel

from app.pipeline.qwen_attack.settings import settings


class Cloner:
    """Qwen3-TTS cloning unit: load, per-speaker prompt cache, clone_single.

    Attributes:
        SYSTEM_ID: Uppercase attack identifier used in output filenames.
        NEEDS_REFERENCE_TRANSCRIPT: Qwen3-TTS requires the reference
            transcript at prompt-build time, so partial_spoof Step 2
            must fetch the bonafide transcript for each speaker before
            calling prepare_speaker.
        model: Loaded Qwen3TTSModel (None before load() is called).
        _speaker_prompts: Per-reference-path voice_clone_prompt cache.
    """

    SYSTEM_ID: str = "QWEN3TTS"
    NEEDS_REFERENCE_TRANSCRIPT: bool = True

    def __init__(self) -> None:
        """Initialise an empty cloner. Call load() before clone_single()."""
        self.model: Optional[Qwen3TTSModel] = None
        self._speaker_prompts: dict = {}

    def load(self, device: str) -> None:
        """Load the Qwen3-TTS model onto the target device.

        Args:
            device: PyTorch device string passed via ``device_map`` to
                ``Qwen3TTSModel.from_pretrained``.

        Raises:
            RuntimeError: If model loading fails (VRAM, CUDA, download).
        """
        logger.info(f"Qwen Cloner: loading model {settings.QWEN_MODEL_ID}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {settings.DTYPE}")
        logger.info(f"  Attention: {settings.QWEN_ATTN_IMPLEMENTATION}")

        start_time = time.time()
        dtype = getattr(torch, settings.DTYPE)
        self.model = Qwen3TTSModel.from_pretrained(
            settings.QWEN_MODEL_ID,
            device_map=device,
            dtype=dtype,
            attn_implementation=settings.QWEN_ATTN_IMPLEMENTATION,
        )
        load_time = time.time() - start_time
        logger.info(f"Qwen Cloner: model loaded in {load_time:.1f}s")

    def prepare_speaker(
        self,
        speaker_id: str,
        reference_audio_path: Path,
        reference_text: str = "",
    ) -> None:
        """Build and cache the per-speaker voice_clone_prompt.

        Args:
            speaker_id: HABLA speaker identifier (used in log messages).
            reference_audio_path: Speaker reference audio.
            reference_text: Transcript of the reference audio. Required
                for the Qwen prompt builder; pass an empty string only
                if the standalone pipeline omits the transcript.

        Raises:
            RuntimeError: If load() was not called first.
        """
        if self.model is None:
            raise RuntimeError(
                "Qwen Cloner: load() must be called before prepare_speaker()"
            )

        ref_key = str(reference_audio_path)
        if ref_key in self._speaker_prompts:
            return

        self._speaker_prompts[ref_key] = self.model.create_voice_clone_prompt(
            ref_audio=ref_key,
            ref_text=reference_text,
            x_vector_only_mode=settings.X_VECTOR_ONLY_MODE,
        )
        logger.debug(f"Qwen Cloner: built voice_clone_prompt for {speaker_id}")

    def clone_single(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: Optional[int] = None,
    ) -> tuple:
        """Generate one Qwen3-TTS clone using the cached speaker prompt.

        Args:
            text: Spanish text to synthesise.
            reference_audio_path: Speaker reference audio path. Used as
                the cache key for the previously-built voice_clone_prompt.
            output_path: Destination WAV path. Written at the sample rate
                returned by ``generate_voice_clone``.
            reference_text: Optional; not used per-call (Qwen uses the
                speaker prompt built in prepare_speaker).
            seed: Accepted for interface compatibility; Qwen does not
                expose a deterministic sampling seed via this API.

        Returns:
            Tuple of (generation_time_seconds, audio_duration_seconds).

        Raises:
            RuntimeError: If load() / prepare_speaker() were not called.
        """
        if self.model is None:
            raise RuntimeError(
                "Qwen Cloner: load() must be called before clone_single()"
            )

        ref_key = str(reference_audio_path)
        if ref_key not in self._speaker_prompts:
            raise RuntimeError(
                f"Qwen Cloner: prepare_speaker() must be called before "
                f"clone_single() for reference '{ref_key}'"
            )
        voice_clone_prompt = self._speaker_prompts[ref_key]

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

        audio = wavs[0]
        sf.write(str(output_path), audio, sr)
        audio_duration = len(audio) / sr

        return generation_time, audio_duration

    def cleanup(self) -> None:
        """Release the Qwen model and clear CUDA memory."""
        self.model = None
        self._speaker_prompts.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Qwen Cloner: GPU memory released")
