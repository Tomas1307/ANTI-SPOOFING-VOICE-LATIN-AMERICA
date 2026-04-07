"""
Attack strategy implementations for Partial Spoof Pipeline.

Each strategy wraps a specific voice cloning system (Fish Speech, Qwen3-TTS,
CosyVoice, OuteTTS, Chatterbox, OpenVoice) and exposes a uniform interface
for generating cloned speech from text and reference audio.
"""
from app.pipeline.partial_spoof.strategies.base_strategy import AttackStrategy

__all__ = ["AttackStrategy"]
