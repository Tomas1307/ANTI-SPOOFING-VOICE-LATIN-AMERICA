"""
Chatterbox Attack Pipeline.

Generates synthetic Spanish voice cloning attacks using Chatterbox
Multilingual TTS (flow-matching, 500M). Fourth codec architecture in the
HABLA anti-spoofing attack suite.
"""
from app.pipeline.chatterbox_attack.pipeline_facade import ChatterboxAttackPipeline

__all__ = ["ChatterboxAttackPipeline"]
