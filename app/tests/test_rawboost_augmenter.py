from app.config.augmentation_config import AugmentationConfigManager
from app.augmenter.rawboost_augmenter import RawBoostAugmenter
import numpy as np

def test_rawboost_augmenter():
    """Test RawBoost augmenter with sample audio."""
    
    
    config_manager = AugmentationConfigManager.get_instance()
    strategy = config_manager.get_strategy("3x")
    
    augmenter = RawBoostAugmenter(config=strategy.rawboost_config)
    
    test_audio = np.random.randn(16000 * 3)
    
    augmented, metadata = augmenter.augment(test_audio, 16000, return_metadata=True)
    
    print(f"\nRawBoost augmentation applied:")
    print(f"  Operations: {metadata['operations']}")
    print(f"  Count: {metadata['num_operations']}")
    print(f"  Label: {augmenter.get_augmentation_label(metadata['operations'])}")


if __name__ == "__main__":
    test_rawboost_augmenter()
