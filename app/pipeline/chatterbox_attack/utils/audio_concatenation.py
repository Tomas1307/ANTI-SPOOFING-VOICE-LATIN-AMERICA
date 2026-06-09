"""
Compatibility shim - delegates to the canonical implementation at
``app.utils.audio_concatenation``.

Chatterbox internally clips reference audio to ~10 seconds for
conditioning. Callers in this pipeline pass ``target_duration=10.0``
explicitly via settings, so the shim default does not matter.
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
