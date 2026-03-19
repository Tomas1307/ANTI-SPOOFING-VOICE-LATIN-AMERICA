"""
OpenVoice V2 Attack Pipeline.

Generates synthetic Spanish voice cloning attacks using OpenVoice V2
(MeloTTS + ToneColorConverter). Third codec architecture in the HABLA
anti-spoofing attack suite.
"""
from app.pipeline.openvoice_attack.pipeline_facade import OpenVoiceAttackPipeline

__all__ = ["OpenVoiceAttackPipeline"]
