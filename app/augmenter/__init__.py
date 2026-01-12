"""
Voice Anti-Spoofing Data Augmentation Package
"""

from app.config.augmentation_config import (
    AugmentationConfigManager,
    AugmentationStrategy
)

from app.schema import (
    AugmentationType,
    RoomSize,
    NoiseSource,
    Distance
)

from app.augmenter.rir_augmenter import RIRAugmenter
from app.augmenter.codec_augmenter import CodecAugmenter
from app.augmenter.rawboost_augmenter import RawBoostAugmenter
from app.dataset_loader import DatasetLoader
from app.augmenter.base_augmenter import BaseAugmenter

__version__ = "1.0.0"

__all__ = [
    # Main components
    "AugmentationConfigManager",
    "DatasetLoader",
    
    # Augmenters
    "RIRAugmenter",
    "CodecAugmenter",
    "RawBoostAugmenter",
    "BaseAugmenter",
    
    # Configuration classes
    "AugmentationStrategy",
    "AugmentationType",
    "RoomSize",
    "NoiseSource",
    "Distance",
]