"""
Audio concatenation utilities for Qwen reference audio preparation.

Re-exports shared utilities from fishgram_attack to avoid code duplication.
Both pipelines use the same audio concatenation logic for reference preparation.
"""
from app.pipeline.fishgram_attack.utils.audio_concatenation import (
    concatenate_with_padding,
    normalize_audio,
    clip_audio,
)

__all__ = [
    "concatenate_with_padding",
    "normalize_audio",
    "clip_audio",
]
