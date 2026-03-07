"""
Quality metrics for Qwen synthetic speech validation.

Re-exports shared utilities from fishgram_attack to avoid code duplication.
Both pipelines use the same DNSMOS and speaker similarity computation.
"""
from app.pipeline.fishgram_attack.utils.quality_metrics import (
    compute_dnsmos,
    compute_speaker_similarity,
    detect_silence,
)

__all__ = [
    "compute_dnsmos",
    "compute_speaker_similarity",
    "detect_silence",
]
