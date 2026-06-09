"""
CosyVoice Attack Pipeline.

Generates synthetic Spanish voice cloning attacks using CosyVoice 3.0
(Alibaba FunAudioLLM, Conditional Flow Matching, 1M hours training data).
Fifth codec architecture in the HABLA anti-spoofing attack suite.
"""
from app.pipeline.cosyvoice_attack.pipeline_facade import CosyVoiceAttackPipeline

__all__ = ["CosyVoiceAttackPipeline"]
