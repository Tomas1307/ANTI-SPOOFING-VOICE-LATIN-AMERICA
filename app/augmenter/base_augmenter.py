"""
Base Augmenter Abstract Class

Provides the foundational interface for all audio augmentation implementations.
All concrete augmenters must inherit from this class and implement the augment method.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class BaseAugmenter(ABC):
    """
    Abstract base class for audio augmentation.
    
    All augmenters must implement the augment method which takes an audio
    signal and returns the augmented version.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize base augmenter.
        
        Args:
            sample_rate: Target sample rate for audio processing.
        """
        self.sample_rate = sample_rate
    
    @abstractmethod
    def augment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply augmentation to audio signal.
        
        Args:
            audio: Input audio signal as numpy array.
            sr: Sample rate of input audio.
            
        Returns:
            Augmented audio signal as numpy array.
        """
        pass
    
    def _normalize_audio(self, audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level.
        
        Args:
            audio: Input audio signal.
            target_level: Target RMS level in dB.
            
        Returns:
            Normalized audio signal.
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            scalar = 10 ** (target_level / 20) / rms
            audio = audio * scalar
        return audio
    
    def _ensure_sample_rate(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Ensure audio is at target sample rate.
        
        Args:
            audio: Input audio signal.
            sr: Current sample rate.
            
        Returns:
            Tuple of (resampled_audio, target_sample_rate).
        """
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        return audio, sr
    
    def _clip_audio(self, audio: np.ndarray, max_val: float = 1.0) -> np.ndarray:
        """
        Clip audio to prevent overflow.
        
        Args:
            audio: Input audio signal.
            max_val: Maximum absolute value.
            
        Returns:
            Clipped audio signal.
        """
        return np.clip(audio, -max_val, max_val)
