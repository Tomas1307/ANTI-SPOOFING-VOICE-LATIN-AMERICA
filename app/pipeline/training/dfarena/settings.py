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
        MODEL_ID: Hugging Face repository identifier of the backbone.
        CLASSIFIER_HIDDEN_DIM: Width of the classifier head hidden layer.
        CLASSIFIER_DROPOUT: Dropout applied inside the classifier head.
        NORMALIZE_INPUT: Whether each waveform is standardised to zero mean
            and unit variance over its unpadded samples, which is what
            wav2vec2-style large backbones expect at their input.
    """

    MODEL_ID: str = Field(
        default="Speech-Arena-2025/DF_Arena_1B_V_1",
        description="Backbone repository identifier.",
    )
    CLASSIFIER_HIDDEN_DIM: int = Field(
        default=256, description="Width of the classifier head hidden layer."
    )
    CLASSIFIER_DROPOUT: float = Field(
        default=0.1, description="Dropout inside the classifier head."
    )
    NORMALIZE_INPUT: bool = Field(
        default=True, description="Standardise each waveform before the backbone."
    )


# Module-level singleton
settings = DFArenaBackendSettings()
