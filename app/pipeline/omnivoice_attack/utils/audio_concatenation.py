"""
Compatibility shim - delegates to the canonical implementation at
``app.utils.audio_concatenation``.

The OmniVoice-specific silence-boundary reference fix originally lived
here (added 2026-05-06). It has since been promoted to the shared
``app/utils/audio_concatenation`` module so every attack pipeline and
the partial_spoof pipeline share a single implementation. Callers that
import from this path continue to work unchanged.
"""
from app.utils.audio_concatenation import (
    concatenate_with_padding,
    find_nearest_silence_boundary,
)


__all__ = [
    "concatenate_with_padding",
    "find_nearest_silence_boundary",
]
