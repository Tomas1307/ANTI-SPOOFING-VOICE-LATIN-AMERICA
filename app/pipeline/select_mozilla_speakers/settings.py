"""
Configuration settings for Mozilla Common Voice Speaker Selection Pipeline.

All pipeline-specific parameters are defined here as a Pydantic model.
Global application settings belong in app/config.py instead.
"""
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List


class MozillaSpeakerSelectionSettings(BaseModel):
    """Configuration for the Mozilla Common Voice speaker selection and integration pipeline.

    Attributes:
        HABLA_DIR: Directory containing original HABLA bonafide speakers
        OUTPUT_DIR: Directory for intermediate pipeline outputs
        CV_ARCHIVE: Path to Common Voice Spanish tar.gz archive
        CV_EXTRACTED_DIR: Directory where CV archive is extracted
        CV_CLIPS_DIR: Directory containing CV audio clips
        BONAFIDE_V2_DIR: Output directory for integrated HABLA v2 dataset

        SAMPLE_RATE: Target audio sample rate (Hz)
        DEVICE: Compute device for model inference

        SIMILARITY_THRESHOLD: Cosine similarity threshold for speaker filtering
        RANDOM_SEED: Random seed for reproducible sampling

        MAX_SAMPLES_PER_SPEAKER: Maximum samples to average for HABLA embeddings
        TRAIN_RATIO: Train split ratio for CV speakers
        VAL_RATIO: Validation split ratio for CV speakers
        TEST_RATIO: Test split ratio for CV speakers

        ACCENT_TARGETS: Fixed sample targets per accent for balanced sampling
        ACCENT_CODES: Mapping from accent name to directory code
        GENDER_CODES: Mapping from gender to directory code

        COLOMBIA_KEYWORDS: Keywords for identifying Colombian accent
        CHILE_KEYWORDS: Keywords for identifying Chilean accent
        VENEZUELA_KEYWORDS: Keywords for identifying Venezuelan accent
        MEXICO_KEYWORDS: Keywords for identifying Mexican accent
        SPAIN_KEYWORDS: Keywords for identifying Spanish accent
    """

    # === Directory Paths ===
    HABLA_DIR: Path = Field(
        default=Path("data/bonafide_dataset_by_speaker"),
        description="Directory containing original HABLA bonafide speakers"
    )
    OUTPUT_DIR: Path = Field(
        default=Path("data/mozilla_speaker_selection"),
        description="Directory for intermediate pipeline outputs (embeddings, metadata)"
    )
    CV_ARCHIVE: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05-es.tar.gz"),
        description="Path to Common Voice Spanish tar.gz archive"
    )
    CV_EXTRACTED_DIR: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05"),
        description="Directory where CV archive is extracted"
    )
    CV_CLIPS_DIR: Path = Field(
        default=Path("data/cv-corpus-24.0-2025-12-05/es/clips"),
        description="Directory containing CV audio clips"
    )
    BONAFIDE_V2_DIR: Path = Field(
        default=Path("data/bonafide_dataset_by_speaker_v2"),
        description="Output directory for integrated HABLA 2.0 dataset"
    )

    # === Audio Processing ===
    SAMPLE_RATE: int = Field(
        default=16000,
        description="Target audio sample rate in Hz"
    )
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Compute device for model inference (cuda or cpu)"
    )

    # === Model & Filtering ===
    MODEL_SOURCE: str = Field(
        default="speechbrain/spkrec-ecapa-voxceleb",
        description="SpeechBrain model identifier for ECAPA-TDNN embeddings"
    )
    MODEL_SAVE_DIR: Path = Field(
        default=Path("pretrained_models/spkrec-ecapa-voxceleb"),
        description="Local directory to save/load pre-trained model"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.75,
        description="Cosine similarity threshold for filtering speakers (max similarity to any HABLA speaker)"
    )
    MAX_SAMPLES_PER_SPEAKER: int = Field(
        default=20,
        description="Maximum samples to average per speaker for embedding extraction"
    )

    # === Sampling Configuration ===
    RANDOM_SEED: int = Field(
        default=42,
        description="Random seed for reproducible stratified sampling"
    )
    TRAIN_RATIO: float = Field(
        default=0.70,
        description="Train split ratio for CV speaker samples"
    )
    VAL_RATIO: float = Field(
        default=0.15,
        description="Validation split ratio for CV speaker samples"
    )
    TEST_RATIO: float = Field(
        default=0.15,
        description="Test split ratio for CV speaker samples"
    )

    # === Accent Targets ===
    ACCENT_TARGETS: Dict[str, int] = Field(
        default={
            "Colombia": 846,     # 4,604 + 846 = 5,450
            "Chile": 1361,       # 4,089 + 1,361 = 5,450
            "Venezuela": 2233,   # 3,217 + 2,233 = 5,450
            "Mexico": 5450,      # 0 + 5,450 = 5,450
            "Spain": 5450,       # 0 + 5,450 = 5,450
        },
        description="Fixed sample targets per accent to achieve balanced representation"
    )

    # === Speaker ID Generation ===
    ACCENT_CODES: Dict[str, str] = Field(
        default={
            "Colombia": "co",
            "Chile": "cl",
            "Venezuela": "ve",
            "Mexico": "mx",
            "Spain": "es",
        },
        description="Mapping from accent name to two-letter directory code"
    )
    GENDER_CODES: Dict[str, str] = Field(
        default={
            "male": "m",
            "female": "f",
            "other": "o",
        },
        description="Mapping from gender to single-letter directory code"
    )

    # === Accent Classification Keywords ===
    COLOMBIA_KEYWORDS: List[str] = Field(
        default=["colombia", "bogot", "colombiano"],
        description="Keywords for identifying Colombian accent in CV metadata"
    )
    CHILE_KEYWORDS: List[str] = Field(
        default=["chile", "chileno"],
        description="Keywords for identifying Chilean accent in CV metadata"
    )
    VENEZUELA_KEYWORDS: List[str] = Field(
        default=["venezuel"],
        description="Keywords for identifying Venezuelan accent in CV metadata"
    )
    MEXICO_KEYWORDS: List[str] = Field(
        default=["méxico", "mexico", "mexicano"],
        description="Keywords for identifying Mexican accent in CV metadata"
    )
    SPAIN_KEYWORDS: List[str] = Field(
        default=[
            "espa", "canaria", "peninsular", "madrid", "castilla",
            "andaluz", "cataluña", "valencia", "gallego", "vasco"
        ],
        description="Keywords for identifying Spanish accent in CV metadata"
    )

    class Config:
        """Pydantic model configuration."""
        frozen = False  # Allow modification if needed during runtime


# Module-level singleton
settings = MozillaSpeakerSelectionSettings()
