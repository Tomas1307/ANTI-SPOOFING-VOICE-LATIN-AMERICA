"""Qwen3-TTS attack strategy for the Partial Spoof pipeline.

Wraps the Qwen3-TTS local model for voice cloning. Requires a reference
transcript for optimal cloning quality via the x-vector prompt mechanism.
"""
import time
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.qwen_attack.settings import settings as qwen_settings


class QwenStrategy(AttackStrategy):
    """Voice cloning via Qwen3-TTS local model.

    Loads the Qwen3-TTS model once, builds a speaker prompt per speaker,
    and reuses it across utterances for efficient generation.

    Attributes:
        model: Loaded Qwen3TTSModel instance (None until load_model()).
    """

    def __init__(self) -> None:
        """Initialize Qwen strategy."""
        self.model = None
        self._speaker_prompts = {}

    def load_model(self, device: str) -> None:
        """Load Qwen3-TTS model onto the specified device.

        Args:
            device: PyTorch device string.
        """
        from qwen_tts import Qwen3TTSModel

        dtype = getattr(torch, qwen_settings.DTYPE)
        self.model = Qwen3TTSModel.from_pretrained(
            qwen_settings.QWEN_MODEL_ID,
            device_map=device,
            dtype=dtype,
            attn_implementation=qwen_settings.QWEN_ATTN_IMPLEMENTATION,
        )
        logger.info(f"QwenStrategy: Model loaded on {device}")

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech using Qwen3-TTS.

        Args:
            text: Text to synthesize.
            reference_audio_path: Speaker reference audio path.
            output_path: Output WAV path.
            reference_text: Transcript of reference audio (recommended).
            seed: Optional random seed.

        Returns:
            Generation time in seconds.
        """
        start_time = time.time()

        ref_key = str(reference_audio_path)
        if ref_key not in self._speaker_prompts:
            self._speaker_prompts[ref_key] = self.model.create_voice_clone_prompt(
                ref_audio=str(reference_audio_path),
                ref_text=reference_text,
                x_vector_only_mode=qwen_settings.X_VECTOR_ONLY_MODE,
            )

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=qwen_settings.QWEN_LANGUAGE,
            voice_clone_prompt=self._speaker_prompts[ref_key],
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wavs[0], sr)

        return time.time() - start_time

    def cleanup(self) -> None:
        """Release model and clear GPU memory."""
        self.model = None
        self._speaker_prompts.clear()
        torch.cuda.empty_cache()
        logger.info("QwenStrategy: Cleanup complete.")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'QWEN3TTS' for protocol file entries.
        """
        return "QWEN3TTS"

    def needs_reference_transcript(self) -> bool:
        """Qwen3-TTS benefits from reference transcripts.

        Returns:
            True.
        """
        return True
