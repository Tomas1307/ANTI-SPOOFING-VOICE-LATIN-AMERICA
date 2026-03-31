"""CosyVoice attack strategy for the Partial Spoof pipeline.

Wraps the CosyVoice 2 local model for zero-shot voice cloning.
Requires a reference transcript for the prompt_text parameter.
"""
import time
from pathlib import Path

import torch
import torchaudio
from loguru import logger

from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.cosyvoice_attack.settings import settings as cosyvoice_settings


class CosyVoiceStrategy(AttackStrategy):
    """Voice cloning via CosyVoice 2 local model.

    Attributes:
        model: Loaded CosyVoice2 model instance.
    """

    def __init__(self) -> None:
        """Initialize CosyVoice strategy."""
        self.model = None

    def load_model(self, device: str) -> None:
        """Load CosyVoice 2 model.

        Args:
            device: PyTorch device string.
        """
        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.model = CosyVoice2(
            cosyvoice_settings.COSYVOICE_MODEL_DIR,
            load_jit=False,
            load_trt=False,
            fp16=cosyvoice_settings.COSYVOICE_FP16,
        )
        logger.info(f"CosyVoiceStrategy: Model loaded from {cosyvoice_settings.COSYVOICE_MODEL_DIR}")

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech using CosyVoice 2.

        Args:
            text: Text to synthesize.
            reference_audio_path: Speaker reference audio path.
            output_path: Output WAV path.
            reference_text: Transcript of reference audio (required).
            seed: Optional random seed.

        Returns:
            Generation time in seconds.
        """
        start_time = time.time()

        prompt_speech, prompt_sr = torchaudio.load(str(reference_audio_path))
        if prompt_sr != 16000:
            prompt_speech = torchaudio.functional.resample(prompt_speech, prompt_sr, 16000)

        full_audio = []
        for chunk in self.model.inference_zero_shot(
            tts_text=text,
            prompt_text=reference_text,
            prompt_speech_16k=prompt_speech,
            stream=False,
        ):
            full_audio.append(chunk["tts_speech"])

        if full_audio:
            audio_tensor = torch.cat(full_audio, dim=-1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(
                str(output_path),
                audio_tensor,
                cosyvoice_settings.COSYVOICE_SAMPLE_RATE,
            )

        return time.time() - start_time

    def cleanup(self) -> None:
        """Release model and clear GPU memory."""
        self.model = None
        torch.cuda.empty_cache()
        logger.info("CosyVoiceStrategy: Cleanup complete.")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'COSYVOICE' for protocol file entries.
        """
        return "COSYVOICE"

    def needs_reference_transcript(self) -> bool:
        """CosyVoice requires reference transcripts.

        Returns:
            True.
        """
        return True
