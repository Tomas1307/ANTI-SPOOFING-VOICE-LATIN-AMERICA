"""
Compatibility shim - delegates to the canonical implementation at
``app.utils.audio_concatenation``.

This pipeline previously re-exported from ``fishgram_attack``; both now
point at the same canonical implementation.
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
