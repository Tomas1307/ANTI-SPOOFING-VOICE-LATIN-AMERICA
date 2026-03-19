"""
No-op watermark bypass for Chatterbox TTS research use.

Chatterbox embeds a neural steganography watermark (Perth) into every generated
audio tensor inside generate(). For anti-spoofing research this watermark must
be removed: it constitutes an undocumented signal that could give anti-spoofing
detectors an artificial advantage on synthetic samples, compromising the validity
of experiments.

This module provides a drop-in replacement for the internal PerthImplicitWatermarker
that leaves the audio unchanged.
"""
import numpy as np


class NoOpWatermarker:
    """No-operation replacement for Chatterbox's PerthImplicitWatermarker.

    Apply to a loaded ChatterboxMultilingualTTS instance after calling
    from_pretrained() to suppress watermark embedding during generate().

    Usage:
        model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        model.watermarker = NoOpWatermarker()

    Attributes:
        None
    """

    def apply_watermark(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the audio array unchanged, skipping watermark embedding.

        Args:
            wav: Audio signal as a NumPy float32 array.
            sample_rate: Sample rate of the audio in Hz (unused).

        Returns:
            The original ``wav`` array, unmodified.
        """
        return wav
