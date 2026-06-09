"""
Mozilla Common Voice Speaker Selection Pipeline.

This pipeline selects acoustically diverse speakers from Common Voice to augment
the HABLA anti-spoofing dataset with Mexico and Spain accents.
"""
from app.pipeline.select_mozilla_speakers.pipeline_facade import (
    MozillaSpeakerSelectionPipeline
)
from app.pipeline.select_mozilla_speakers.schemas import (
    MozillaSpeakerPipelineConfig,
    EmbeddingResult,
)
from app.pipeline.select_mozilla_speakers.settings import settings

__all__ = [
    "MozillaSpeakerSelectionPipeline",
    "MozillaSpeakerPipelineConfig",
    "EmbeddingResult",
    "settings",
]
