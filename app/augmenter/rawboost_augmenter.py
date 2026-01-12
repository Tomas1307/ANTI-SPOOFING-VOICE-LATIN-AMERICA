"""
RawBoost Augmentation

Implements RawBoost data augmentation technique from Tak et al. (2022).
Applies raw waveform perturbations including linear/nonlinear convolutive noise
and signal-dependent additive noise.

Reference:
Tak, H., et al. (2022). "RawBoost: A Raw Data Boosting and Augmentation Method
applied to Automatic Speaker Verification Anti-Spoofing"
"""

import numpy as np
import random
from scipy import signal

from app.augmenter.base_augmenter import BaseAugmenter
from app.schema import RawBoostConfig


class RawBoostAugmenter(BaseAugmenter):
    """
    RawBoost augmentation implementation.
    
    Applies three types of perturbations:
    1. Linear convolutive noise (LFR - Linear Filtering with Random impulse)
    2. Nonlinear convolutive noise (nonlinear distortion)
    3. Signal-dependent additive noise (SDAN - clipping, overflow effects)
    
    Attributes:
        config: RawBoostConfig object with augmentation parameters.
    """
    
    def __init__(self, config: RawBoostConfig, sample_rate: int = 16000):
        """
        Initialize RawBoost augmenter.
        
        Args:
            config: Configuration object for RawBoost augmentation.
            sample_rate: Target sample rate for processing.
        """
        super().__init__(sample_rate)
        self.config = config
        
        print(f"RawBoostAugmenter initialized:")
        print(f"  - Linear filtering: {config.apply_linear_filtering}")
        print(f"  - Nonlinear filtering: {config.apply_nonlinear_filtering}")
        print(f"  - Additive noise: {config.apply_additive_noise}")
        print(f"  - Clipping threshold: {config.clipping_threshold}")
    
    def _apply_linear_filtering(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply linear convolutive noise using random FIR filter.
        
        Simulates linear distortions from recording devices and channels.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Filtered audio signal.
        """
        filter_length = random.randint(5, 25)
        
        filter_coeffs = np.random.randn(filter_length)
        filter_coeffs = filter_coeffs / np.sum(np.abs(filter_coeffs))
        
        filtered = signal.lfilter(filter_coeffs, [1.0], audio)
        
        return filtered
    
    def _apply_nonlinear_distortion(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply nonlinear distortion to simulate harmonic distortion.
        
        Uses polynomial nonlinearity to model amplifier distortion effects.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Distorted audio signal.
        """
        alpha = random.uniform(0.1, 0.5)
        
        distorted = np.tanh(alpha * audio)
        
        distorted = distorted / np.max(np.abs(distorted)) * 0.99
        
        return distorted
    
    def _apply_clipping(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply hard clipping to simulate ADC overflow.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Clipped audio signal.
        """
        threshold = self.config.clipping_threshold
        
        clipped = np.clip(audio, -threshold, threshold)
        
        return clipped
    
    def _apply_additive_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply signal-dependent additive noise.
        
        Simulates quantization noise and electronic interference that
        correlates with signal amplitude.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Audio with additive noise.
        """
        noise_level = random.uniform(0.001, 0.01)
        
        signal_dependent_noise = audio * np.random.randn(len(audio)) * noise_level
        
        noise = np.random.randn(len(audio)) * noise_level * 0.5
        
        noisy = audio + signal_dependent_noise + noise
        
        return noisy
    
    def _apply_random_gain(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random gain variation to simulate AGC effects.
        
        Args:
            audio: Input audio signal.
            
        Returns:
            Audio with gain variation.
        """
        gain = random.uniform(0.7, 1.3)
        
        return audio * gain
    
    def augment(
        self,
        audio: np.ndarray,
        sr: int,
        return_metadata: bool = False
    ) -> np.ndarray:
        """
        Apply RawBoost augmentation to audio.
        
        Randomly applies combination of:
        - Linear filtering
        - Nonlinear distortion
        - Additive noise
        - Clipping
        - Gain variation
        
        Args:
            audio: Input audio signal.
            sr: Sample rate of input audio.
            return_metadata: If True, returns tuple (audio, metadata).
            
        Returns:
            Augmented audio signal, or tuple (audio, metadata) if return_metadata=True.
        """
        audio, sr = self._ensure_sample_rate(audio, sr)
        
        augmented = audio.copy()
        
        applied_operations = []
        
        if self.config.apply_linear_filtering and random.random() < 0.5:
            augmented = self._apply_linear_filtering(augmented)
            applied_operations.append("linear_filter")
        
        if self.config.apply_nonlinear_filtering and random.random() < 0.3:
            augmented = self._apply_nonlinear_distortion(augmented)
            applied_operations.append("nonlinear_distortion")
        
        if self.config.apply_additive_noise and random.random() < 0.6:
            augmented = self._apply_additive_noise(augmented)
            applied_operations.append("additive_noise")
        
        if random.random() < 0.4:
            augmented = self._apply_random_gain(augmented)
            applied_operations.append("gain_variation")
        
        if random.random() < 0.2:
            augmented = self._apply_clipping(augmented)
            applied_operations.append("clipping")
        
        augmented = self._normalize_audio(augmented)
        augmented = self._clip_audio(augmented, max_val=0.99)
        
        if return_metadata:
            metadata = {
                "operations": applied_operations,
                "num_operations": len(applied_operations)
            }
            return augmented, metadata
        
        return augmented
    
    def get_augmentation_label(self, operations: list) -> str:
        """
        Generate descriptive label for augmentation applied.
        
        Args:
            operations: List of operations applied.
            
        Returns:
            Formatted augmentation label.
        """
        if not operations:
            return "RAWBOOST_NONE"
        
        op_short = {
            "linear_filter": "LF",
            "nonlinear_distortion": "NL",
            "additive_noise": "AN",
            "gain_variation": "GV",
            "clipping": "CL"
        }
        
        ops = "_".join([op_short.get(op, op[:2].upper()) for op in operations])
        
        return f"RAWBOOST_{ops}"

