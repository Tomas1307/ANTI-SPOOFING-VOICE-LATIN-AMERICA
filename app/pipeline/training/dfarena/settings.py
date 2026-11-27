"""
Backend-specific settings for the DF-Arena detector.

Only parameters that belong to this backend live here. Everything shared by
every detector, such as corpus paths, audit thresholds and optimisation
defaults, belongs in the training pipeline settings one level up.
"""
from pydantic import BaseModel, Field


class DFArenaBackendSettings(BaseModel):
    """Configuration for the DF-Arena detector backend.

    Attributes:
        MODEL_ID: Hugging Face repository identifier of the model.
        REQUIRED_SAMPLES: Exact input length the published model demands. Its
            feature extractor truncates longer clips and tiles shorter ones to
            this length, so the dataset must deliver it precisely. 64,600
            samples is 4.0375 seconds at 16 kHz, the ASVspoof convention.
    """

    MODEL_ID: str = Field(
        default="Speech-Arena-2025/DF_Arena_1B_V_1",
        description="Model repository identifier.",
    )
    REQUIRED_SAMPLES: int = Field(
        default=64600, description="Exact input length in samples."
    )


# Module-level singleton
settings = DFArenaBackendSettings()
