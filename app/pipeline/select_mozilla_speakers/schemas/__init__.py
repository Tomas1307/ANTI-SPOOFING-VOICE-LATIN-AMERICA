"""
Schemas for Mozilla Common Voice speaker selection pipeline.

All data structures use Pydantic BaseModel as per CLAUDE.md guidelines.
"""
from app.pipeline.select_mozilla_speakers.schemas.pipeline_config import (
    MozillaSpeakerPipelineConfig
)
from app.pipeline.select_mozilla_speakers.schemas.embedding_result import (
    EmbeddingResult
)

__all__ = [
    "MozillaSpeakerPipelineConfig",
    "EmbeddingResult",
]
