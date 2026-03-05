"""
Schema for Fish Speech model loading result.
"""
from pydantic import BaseModel, Field
from typing import Any


class ModelLoadResult(BaseModel):
    """Result from Fish Speech model loading (Step 1).

    Attributes:
        model: Loaded Fish Speech model instance
        vram_usage_mb: Peak VRAM usage in megabytes
        warmup_rtf: Real-time factor from warmup inference
    """

    model: Any = Field(
        ...,
        description="Loaded Fish Speech model instance (stored for later steps)"
    )
    vram_usage_mb: int = Field(
        ...,
        description="Peak VRAM usage in megabytes during model loading"
    )
    warmup_rtf: float = Field(
        ...,
        description="Real-time factor from warmup inference (generation_time / audio_duration)"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False
        arbitrary_types_allowed = True  # Allow Any type for model instance
