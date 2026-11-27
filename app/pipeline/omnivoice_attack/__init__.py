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
from app.pipeline.omnivoice_attack.settings import settings

__all__ = [
    "OmniVoiceAttackPipeline",
    "OmniVoicePipelineConfig",
    "settings",
]


def __getattr__(name: str):
    """Lazily resolve the heavy facade/config exports on first access.

    ``OmniVoiceAttackPipeline`` and ``OmniVoicePipelineConfig`` pull in
    every pipeline step (including Step 4's NeMo/Parakeet-based quality
    validation), which is only installed in the venvs that actually run
    the full attack pipeline. Deferring the import here means a
    lightweight consumer that only needs ``settings`` or the ``Cloner``
    (e.g. a standalone demo script) can `import
    app.pipeline.omnivoice_attack` without those heavy dependencies being
    present, matching the lazy-import venv isolation already documented
    in ``app.pipeline.partial_spoof.utils.cloner_dispatcher``.
    """
    if name == "OmniVoiceAttackPipeline":
        from app.pipeline.omnivoice_attack.pipeline_facade import OmniVoiceAttackPipeline

        return OmniVoiceAttackPipeline
    if name == "OmniVoicePipelineConfig":
        from app.pipeline.omnivoice_attack.schemas.pipeline_config import OmniVoicePipelineConfig

        return OmniVoicePipelineConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
