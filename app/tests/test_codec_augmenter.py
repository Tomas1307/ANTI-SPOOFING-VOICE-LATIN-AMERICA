from app.config.augmentation_config import AugmentationConfigManager
from app.augmenter.codec_augmenter import CodecAugmenter
import numpy as np

def test_codec_augmenter():
    """Test codec augmenter with sample audio."""
    
    config_manager = AugmentationConfigManager.get_instance()
    strategy = config_manager.get_strategy("3x")
    
    augmenter = CodecAugmenter(config=strategy.codec_config)
    
    test_audio = np.random.randn(16000 * 3)
    
    augmented, metadata = augmenter.augment(test_audio, 16000, return_metadata=True)
    
    print(f"\nCodec augmentation applied:")
    print(f"  Codec SR: {metadata['codec_sr']} Hz")
    print(f"  Packet loss: {metadata['packet_loss']*100:.1f}%")
    print(f"  Bandpass: {metadata['bandpass']}")
    print(f"  Label: {augmenter.get_augmentation_label(metadata['codec_sr'], metadata['packet_loss'])}")


if __name__ == "__main__":
    test_codec_augmenter()
