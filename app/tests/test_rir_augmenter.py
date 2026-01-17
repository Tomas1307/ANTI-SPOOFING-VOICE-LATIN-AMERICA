from app.config.augmentation_config import AugmentationConfigManager
from app.augmenter.rir_augmenter import RIRAugmenter
import numpy as np

def test_rir_augmenter():
    """Test RIR augmenter with sample audio."""
    
    
    config_manager = AugmentationConfigManager.get_instance()
    strategy = config_manager.get_strategy("3x")
    
    augmenter = RIRAugmenter(
        config=strategy.rir_noise_config,
        rir_root="data/noise_dataset/RIR",
        noise_root="data/noise_dataset/musan"
    )
    
    test_audio = np.random.randn(16000 * 3)
    
    augmented, metadata = augmenter.augment(test_audio, 16000, return_metadata=True)
    
    print(f"\nAugmentation applied:")
    print(f"  Room: {metadata['room_size']}")
    print(f"  Noise: {metadata['noise_source']}")
    print(f"  SNR: {metadata['snr_db']:.1f} dB")
    print(f"  Label: {augmenter.get_augmentation_label(**metadata)}")


if __name__ == "__main__":
    test_rir_augmenter()
