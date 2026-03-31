"""Chatterbox attack strategy for the Partial Spoof pipeline.

Wraps the ChatterboxMultilingualTTS local model for voice cloning.
Does not require a reference transcript; cloning is purely audio-based.
"""
import time
from pathlib import Path

import torch
import torchaudio
from loguru import logger

from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy
from app.pipeline.chatterbox_attack.settings import settings as chatterbox_settings
from app.pipeline.chatterbox_attack.utils.watermark_remover import NoOpWatermarker
from app.pipeline.chatterbox_attack.utils.speech_trimmer import trim_trailing_noise


class ChatterboxStrategy(AttackStrategy):
    """Voice cloning via ChatterboxMultilingualTTS.

    Attributes:
        model: Loaded Chatterbox model instance.
    """

    def __init__(self) -> None:
        """Initialize Chatterbox strategy."""
        self.model = None

    def load_model(self, device: str) -> None:
        """Load ChatterboxMultilingualTTS with watermark bypass.

        Args:
            device: PyTorch device string.
        """
        from app.pipeline.chatterbox_attack.utils.perth_patcher import ensure_patched  # noqa: F401
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self.model.watermarker = NoOpWatermarker()
        self._patch_sdpa_to_eager()
        logger.info(f"ChatterboxStrategy: Model loaded on {device}")

    def _patch_sdpa_to_eager(self) -> None:
        """Force eager attention for transformers >= 4.47 compatibility."""
        if self.model is None:
            return
        for attr_name in dir(self.model):
            try:
                attr = getattr(self.model, attr_name)
                if isinstance(attr, torch.nn.Module) and hasattr(attr, "config"):
                    if hasattr(attr.config, "_attn_implementation"):
                        attr.config._attn_implementation = "eager"
            except Exception:
                pass

    def generate(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path,
        reference_text: str = "",
        seed: int | None = None,
    ) -> float:
        """Generate cloned speech using Chatterbox.

        Args:
            text: Text to synthesize.
            reference_audio_path: Speaker reference audio path.
            output_path: Output WAV path.
            reference_text: Ignored by Chatterbox.
            seed: Optional random seed.

        Returns:
            Generation time in seconds.
        """
        start_time = time.time()

        wav = self.model.generate(
            text=text,
            language_id=chatterbox_settings.LANGUAGE_ID,
            audio_prompt_path=str(reference_audio_path),
            exaggeration=chatterbox_settings.EXAGGERATION,
            cfg_weight=chatterbox_settings.CFG_WEIGHT,
            temperature=chatterbox_settings.TEMPERATURE,
            repetition_penalty=chatterbox_settings.REPETITION_PENALTY,
        )

        wav_resampled = torchaudio.functional.resample(
            wav, self.model.sr, chatterbox_settings.SAMPLE_RATE
        )
        wav_trimmed = trim_trailing_noise(
            wav_resampled, chatterbox_settings.SAMPLE_RATE,
            margin_ms=chatterbox_settings.VAD_MARGIN_MS,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav_trimmed, chatterbox_settings.SAMPLE_RATE)

        return time.time() - start_time

    def cleanup(self) -> None:
        """Release model and clear GPU memory."""
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        logger.info("ChatterboxStrategy: Cleanup complete.")

    def name(self) -> str:
        """Return the system identifier.

        Returns:
            'CHATTERBOX' for protocol file entries.
        """
        return "CHATTERBOX"

    def needs_reference_transcript(self) -> bool:
        """Chatterbox does not need reference transcripts.

        Returns:
            False.
        """
        return False
