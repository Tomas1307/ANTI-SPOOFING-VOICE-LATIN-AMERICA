"""
Compatibility shim - delegates to the canonical implementation at
``app.utils.audio_concatenation``.

OuteTTS benefits from 10-15 second reference clips for speaker profile
extraction. Callers pass ``target_duration`` explicitly via settings.
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
