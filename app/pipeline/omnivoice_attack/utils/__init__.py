"""
Utility functions for OmniVoice Attack Pipeline.
"""
from app.pipeline.omnivoice_attack.utils.audio_concatenation import (
    concatenate_with_padding,
)

__all__ = [
    "concatenate_with_padding",
    "detect_silence",
    "transcribe_audio",
]


def __getattr__(name: str):
    """Lazily resolve exports that pull in heavy, step-specific SDKs.

    ``detect_silence`` needs librosa and ``transcribe_audio`` needs the
    NeMo/Parakeet ASR stack; both are only required by Step 4's quality
    validation, not by every consumer of this package (e.g. ``Cloner``,
    imported from ``app.pipeline.omnivoice_attack.utils.cloner``, needs
    neither). Deferring the import keeps lightweight consumers free of
    those dependencies, matching the venv-isolation pattern documented in
    ``app.pipeline.partial_spoof.utils.cloner_dispatcher``.
    """
    if name == "detect_silence":
        from app.pipeline.omnivoice_attack.utils.quality_metrics import detect_silence

        return detect_silence
    if name == "transcribe_audio":
        from app.pipeline.omnivoice_attack.utils.reference_transcriber import transcribe_audio

        return transcribe_audio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
