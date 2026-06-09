"""
OmniVoice Attack Pipeline

Voice cloning attack generation using OmniVoice (k2-fsa) for anti-spoofing
dataset augmentation. OmniVoice is a 646-language zero-shot TTS model based
on a diffusion language model architecture, with strong Spanish support
(27,559 hours of training data).

Usage:
    from app.pipeline.omnivoice_attack import OmniVoiceAttackPipeline, settings

    settings.VALIDATION_MODE = True
    settings.SAMPLES_PER_SPEAKER = 2

    pipeline = OmniVoiceAttackPipeline()
    output_dir = pipeline.run()
"""
from app.pipeline.omnivoice_attack.pipeline_facade import OmniVoiceAttackPipeline
from app.pipeline.omnivoice_attack.schemas.pipeline_config import OmniVoicePipelineConfig
from app.pipeline.omnivoice_attack.settings import settings

__all__ = [
    "OmniVoiceAttackPipeline",
    "OmniVoicePipelineConfig",
    "settings",
]
