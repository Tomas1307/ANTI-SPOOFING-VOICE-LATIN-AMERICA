"""OuteTTS attack strategy for the Partial Spoof pipeline.

Wraps the OuteTTS local model for voice cloning. Does not require
a reference transcript; cloning is audio-based via DAC tokens.
"""
import time
from pathlib import Path

import torch
import soundfile as sf
from loguru import logger

from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.outetts_attack.settings import settings as outetts_settings


class OuteTTSStrategy(AttackStrategy):
    """Voice cloning via OuteTTS 0.6B local model.

    Attributes:
        model: Loaded OuteTTS interface instance.
        speaker_cache: Cache of speaker profiles per reference path.
    """

    def __init__(self) -> None:
        """Initialize OuteTTS strategy."""
        self.model = None
        self.speaker_cache = {}

    def load_model(self, device: str) -> None:
        """Load OuteTTS model.

        Args:
            device: PyTorch device string.
        """
        import outetts

        self.model = outetts.InterfaceHF(
            model_version=outetts_settings.OUTETTS_MODEL_VERSION,
            backend=outetts_settings.OUTETTS_BACKEND,
        )
        logger.info(f"OuteTTSStrategy: Model loaded ({outetts_settings.OUTETTS_MODEL_VERSION})")

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech using OuteTTS.

        Args:
            text: Text to synthesize.
            reference_audio_path: Speaker reference audio path.
            output_path: Output WAV path.
            reference_text: Ignored by OuteTTS.
            seed: Optional random seed.

        Returns:
            Generation time in seconds.
        """
        start_time = time.time()

        ref_key = str(reference_audio_path)
        if ref_key not in self.speaker_cache:
            self.speaker_cache[ref_key] = self.model.load_speaker(str(reference_audio_path))

        output = self.model.generate(
            text=text,
            speaker=self.speaker_cache[ref_key],
            temperature=outetts_settings.OUTETTS_TEMPERATURE,
            repetition_penalty=outetts_settings.OUTETTS_REPETITION_PENALTY,
            max_length=outetts_settings.OUTETTS_MAX_LENGTH,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(str(output_path))

        return time.time() - start_time

    def cleanup(self) -> None:
        """Release model and clear GPU memory."""
        self.model = None
        self.speaker_cache.clear()
        torch.cuda.empty_cache()
        logger.info("OuteTTSStrategy: Cleanup complete.")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'OUTETTS' for protocol file entries.
        """
        return "OUTETTS"

    def needs_reference_transcript(self) -> bool:
        """OuteTTS does not need reference transcripts.

        Returns:
            False.
        """
        return False
