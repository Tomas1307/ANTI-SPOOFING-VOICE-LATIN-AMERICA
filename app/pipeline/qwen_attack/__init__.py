"""
Qwen3-TTS Attack Pipeline.

Generates synthetic voice cloning attacks using Qwen3-TTS (1.7B)
for anti-spoofing dataset augmentation. Secondary attack pipeline
providing codec architecture diversity alongside FishGram (Fish Speech).
"""
from app.pipeline.qwen_attack.pipeline_facade import QwenAttackPipeline
from app.pipeline.qwen_attack.schemas.pipeline_config import QwenPipelineConfig
from app.pipeline.qwen_attack.settings import settings

__all__ = [
    "QwenAttackPipeline",
    "QwenPipelineConfig",
    "settings",
]
