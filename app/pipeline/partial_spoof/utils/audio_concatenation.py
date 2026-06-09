"""
Compatibility shim - delegates to the canonical implementation at
``app.utils.audio_concatenation``.

The partial_spoof pipeline previously kept its own copy of the audio
concatenation utility, which fell out of sync with the OmniVoice fix
landed in ``omnivoice_attack/utils/audio_concatenation.py`` on
2026-05-06 and caused OmniVoice clones to hallucinate. The canonical
implementation now lives in ``app/utils/audio_concatenation`` so every
attack pipeline plus partial_spoof share one source of truth.
"""
from app.utils.audio_concatenation import (
    concatenate_with_padding,
    find_nearest_silence_boundary,
)


__all__ = [
    "concatenate_with_padding",
    "find_nearest_silence_boundary",
]
