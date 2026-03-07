"""
Utility functions for Qwen Attack Pipeline.

Shared helpers for audio processing, quality metrics, transcription,
and Qwen-specific artifact detection.
"""
from app.pipeline.qwen_attack.utils.audio_concatenation import (
    concatenate_with_padding,
    normalize_audio,
    clip_audio,
)
from app.pipeline.qwen_attack.utils.quality_metrics import (
    compute_dnsmos,
    compute_speaker_similarity,
    detect_silence,
)
from app.pipeline.qwen_attack.utils.reference_transcriber import transcribe_audio
from app.pipeline.qwen_attack.utils.artifact_detector import (
    detect_truncation,
    detect_low_energy,
    detect_duration_anomaly,
)

__all__ = [
    "concatenate_with_padding",
    "normalize_audio",
    "clip_audio",
    "compute_dnsmos",
    "compute_speaker_similarity",
    "detect_silence",
    "transcribe_audio",
    "detect_truncation",
    "detect_low_energy",
    "detect_duration_anomaly",
]
