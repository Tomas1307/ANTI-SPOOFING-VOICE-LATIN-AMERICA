"""OmniVoice attack strategy for the Partial Spoof pipeline.

Wraps the OmniVoice (k2-fsa) local model for word-level voice cloning
splice attacks. OmniVoice is a 646-language zero-shot TTS based on a
diffusion language model architecture; Spanish is supported with 27,559
hours of training data.

The omnivoice import lives inside load_model() because the package is
installed only in envs/omnivoice_env. Other partial_spoof strategy modules
follow the same pragmatic pattern (e.g. qwen_strategy, outetts_strategy).
"""
import time
from pathlib import Path

import torch
import soundfile as sf
from loguru import logger

from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.omnivoice_attack.settings import settings as omnivoice_settings


class OmniVoiceStrategy(AttackStrategy):
    """Voice cloning via OmniVoice local model.

    Loads the OmniVoice diffusion language model once and reuses it for
    all generation calls. OmniVoice is benefit from a reference transcript
    (ref_text), which is supplied by the partial_spoof Step 1 (bonafide
    transcription via Parakeet TDT).

    Audio is generated at OmniVoice's native 24 kHz sample rate. The
    partial_spoof splicing engine resamples audio on load via librosa
    when needed.

    Attributes:
        model: Loaded OmniVoice instance (None until load_model()).
    """

    def __init__(self) -> None:
        """Initialize OmniVoice strategy."""
        self.model = None

    def load_model(self, device: str) -> None:
        """Load OmniVoice model onto the specified device.

        Args:
            device: PyTorch device string (e.g., 'cuda:0', 'cpu').

        Raises:
            RuntimeError: If model loading fails (VRAM, CUDA, or download issues).
        """
        from omnivoice import OmniVoice

        dtype = getattr(torch, omnivoice_settings.DTYPE)
        self.model = OmniVoice.from_pretrained(
            omnivoice_settings.OMNIVOICE_MODEL_ID,
            device_map=device,
            dtype=dtype,
        )
        logger.info(
            f"OmniVoiceStrategy: Model loaded on {device} "
            f"({omnivoice_settings.OMNIVOICE_MODEL_ID})"
        )

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech using OmniVoice.

        Args:
            text: Text to synthesize.
            reference_audio_path: Speaker reference audio path.
            output_path: Output WAV path.
            reference_text: Transcript of reference audio (Parakeet output, recommended).
            seed: Optional random seed for reproducible generation.

        Returns:
            Generation time in seconds.

        Raises:
            RuntimeError: If generation fails.
        """
        start_time = time.time()

        audios = self.model.generate(
            text=text,
            ref_audio=str(reference_audio_path),
            ref_text=reference_text,
            num_step=omnivoice_settings.OMNIVOICE_NUM_STEP,
            speed=omnivoice_settings.OMNIVOICE_SPEED,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(output_path),
            audios[0],
            omnivoice_settings.OMNIVOICE_NATIVE_SAMPLE_RATE,
        )

        return time.time() - start_time

    def cleanup(self) -> None:
        """Release model and clear GPU memory."""
        self.model = None
        torch.cuda.empty_cache()
        logger.info("OmniVoiceStrategy: Cleanup complete.")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'OMNIVOICE' for protocol file entries.
        """
        return "OMNIVOICE"

    def needs_reference_transcript(self) -> bool:
        """OmniVoice benefits from reference transcripts for higher cloning quality.

        Returns:
            True.
        """
        return True
