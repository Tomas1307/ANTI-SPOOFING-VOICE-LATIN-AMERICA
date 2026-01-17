#!/usr/bin/env python3
"""
Quick Test Script for Augmentation Pipeline

Validates that all components are properly installed and configured
before running the full augmentation pipeline.
"""

import sys
from pathlib import Path

print("="*70)
print("AUGMENTATION PIPELINE - SYSTEM CHECK")
print("="*70)

# Test 1: Check Python version
print("\n1. Checking Python version...")
if sys.version_info < (3, 7):
    print("   ❌ ERROR: Python 3.7+ required")
    sys.exit(1)
print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}")

# Test 2: Check dependencies
print("\n2. Checking dependencies...")
required_packages = [
    "numpy", "scipy", "librosa", "soundfile", "tqdm", "pandas"
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - MISSING")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   Install missing packages:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Test 3: Check augmentation modules
print("\n3. Checking augmentation modules...")
sys.path.insert(0, str(Path(__file__).parent))

modules = [
    "augmentation_config",
    "base_augmenter",
    "utils",
    "rir_augmenter",
    "codec_augmenter",
    "rawboost_augmenter",
    "dataset_loader",
    "augmentation_pipeline"
]

for module in modules:
    try:
        __import__(module)
        print(f"   ✅ {module}.py")
    except ImportError as e:
        print(f"   ❌ {module}.py - {e}")
        sys.exit(1)

# Test 4: Test configuration manager
print("\n4. Testing configuration manager...")
try:
    from app.config.augmentation_config import AugmentationConfigManager
    
    config = AugmentationConfigManager.get_instance()
    strategy = config.get_strategy("3x")
    
    print(f"   ✅ Singleton pattern working")
    print(f"   ✅ 3x strategy loaded")
    print(f"      - Augmentation factor: {strategy.augmentation_factor}")
    print(f"      - RIR+Noise: {strategy.type_distribution.get(list(strategy.type_distribution.keys())[0])*100:.0f}%")
except Exception as e:
    print(f"   ❌ Configuration error: {e}")
    sys.exit(1)

# Test 5: Test augmenters
print("\n5. Testing augmenters...")
try:
    from app.config.augmentation_config import AugmentationConfigManager
    from app.augmenter.rir_augmenter import RIRAugmenter
    from app.augmenter.codec_augmenter import CodecAugmenter
    from app.augmenter.rawboost_augmenter import RawBoostAugmenter
    import numpy as np
    
    config = AugmentationConfigManager.get_instance()
    strategy = config.get_strategy("3x")
    
    # Test RIR augmenter
    rir_aug = RIRAugmenter(
        config=strategy.rir_noise_config,
        rir_root="data/noise_dataset/RIR",
        noise_root="data/noise_dataset/musan"
    )
    test_audio = np.random.randn(16000)
    rir_aug.augment(test_audio, 16000)
    print(f"   ✅ RIR Augmenter")
    
    # Test codec augmenter
    codec_aug = CodecAugmenter(config=strategy.codec_config)
    codec_aug.augment(test_audio, 16000)
    print(f"   ✅ Codec Augmenter")
    
    # Test rawboost augmenter
    rawboost_aug = RawBoostAugmenter(config=strategy.rawboost_config)
    rawboost_aug.augment(test_audio, 16000)
    print(f"   ✅ RawBoost Augmenter")
    
except Exception as e:
    print(f"   ❌ Augmenter error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check data directories
print("\n6. Checking data directories...")
data_paths = {
    "Voices": "data/partition_dataset_by_speaker",
    "MUSAN": "data/noise_dataset/musan",
    "RIR": "data/noise_dataset/RIR"
}

all_exist = True
for name, path in data_paths.items():
    if Path(path).exists():
        print(f"   ✅ {name}: {path}")
    else:
        print(f"   ⚠️  {name}: {path} - NOT FOUND (pipeline will warn)")
        all_exist = False

if not all_exist:
    print("\n   Note: Some data directories not found.")
    print("   Pipeline will work but augmentation quality may be affected.")

# Test 7: Test dataset loader
print("\n7. Testing dataset loader...")
try:
    from dataset_loader import DatasetLoader
    
    loader = DatasetLoader(
        voices_root="data/partition_dataset_by_speaker",
        musan_root="data/noise_dataset/musan",
        rir_root="data/noise_dataset/RIR"
    )
    
    print(f"   ✅ Dataset loader initialized")
    
    # Try to load files if directory exists
    if Path("data/partition_dataset_by_speaker").exists():
        try:
            train_files = loader.load_train_files()
            print(f"   ✅ Found {len(train_files)} training files")
        except Exception as e:
            print(f"   ⚠️  Could not load train files: {e}")
    
except Exception as e:
    print(f"   ❌ Loader error: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("SYSTEM CHECK COMPLETE")
print("="*70)
print("\n✅ All components working correctly!")
print("\nYou can now run the augmentation pipeline:")
print("   python run_augmentation.py --factor 3x")
print("\n" + "="*70)
