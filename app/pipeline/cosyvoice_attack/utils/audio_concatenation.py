"""
Compatibility shim - delegates to the canonical implementation at
``app.utils.audio_concatenation``.

CosyVoice was dropped from the active attack suite (no Spanish support)
but kept here for completeness in case the pipeline is revived.
"""
from app.utils.audio_concatenation import (
    clip_audio,
    concatenate_with_padding,
    find_nearest_silence_boundary,
    normalize_audio,
)


__all__ = [
    "clip_audio",
    "concatenate_with_padding",
    "find_nearest_silence_boundary",
    "normalize_audio",
]
