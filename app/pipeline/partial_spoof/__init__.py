"""
Partial Spoof Pipeline.

Creates partially spoofed Latin American Spanish audio by replacing
individual words in bonafide HABLA utterances with voice-cloned versions.
Supports configurable attack strategies (Fish Speech, Qwen3-TTS, CosyVoice,
OuteTTS, Chatterbox, OpenVoice) and word replacement tiers (W1, W2, W3).
"""
from app.pipeline.partial_spoof.pipeline_facade import PartialSpoofPipeline

__all__ = ["PartialSpoofPipeline"]
