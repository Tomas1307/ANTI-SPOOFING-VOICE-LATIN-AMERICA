"""
OuteTTS Attack Pipeline.

Generates synthetic Spanish voice cloning attacks using OuteTTS 0.6B
(Qwen-based LLM with DAC codec). Fifth codec architecture in the
HABLA anti-spoofing attack suite.
"""
from app.pipeline.outetts_attack.pipeline_facade import OuteTTSAttackPipeline

__all__ = ["OuteTTSAttackPipeline"]
