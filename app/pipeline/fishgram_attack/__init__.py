"""
FishGram Attack Pipeline

Voice cloning attack generation using Fish Speech (4B parameters) for
anti-spoofing dataset augmentation.

Usage:
    from app.pipeline.fishgram_attack import FishGramAttackPipeline, settings

    # Validation mode (3 speakers, 6 samples)
    settings.VALIDATION_MODE = True
    settings.SAMPLES_PER_SPEAKER = 2

    pipeline = FishGramAttackPipeline()
    output_dir = pipeline.run()

    # Production mode (162 speakers, 810 samples)
    settings.VALIDATION_MODE = False
    settings.SAMPLES_PER_SPEAKER = 5

    pipeline = FishGramAttackPipeline()
    output_dir = pipeline.run()
"""
from app.pipeline.fishgram_attack.pipeline_facade import FishGramAttackPipeline
from app.pipeline.fishgram_attack.schemas.pipeline_config import FishGramPipelineConfig
from app.pipeline.fishgram_attack.settings import settings

__all__ = [
    "FishGramAttackPipeline",
    "FishGramPipelineConfig",
    "settings",
]
