"""
Codec and Channel Degradation Augmentation

Simulates telephone and communication channel effects including downsampling,
codec compression, and packet loss. Based on ASVspoof channel augmentation
strategies.
"""

import numpy as np
import random
from typing import Optional, Tuple

from app.augmenter.base_augmenter import BaseAugmenter
from app.schema import CodecConfig
import app.utils.utils as utils


class CodecAugmenter(BaseAugmenter):
    """
    Channel and codec degradation augmentation.
    
    Simulates telecommunication channel effects by applying:
    - Sample rate conversion (e.g., 16kHz -> 8kHz -> 16kHz)
    - Packet loss simulation
    - Codec compression artifacts
    
    Attributes:
        config: CodecConfig object with degradation parameters.
    """
    
    def __init__(self, config: CodecConfig, sample_rate: int = 16000):
        """
        Initialize codec augmenter.
        
        Args:
            config: Configuration object for codec augmentation.
            sample_rate: Target sample rate for output.
        """
        super().__init__(sample_rate)
        self.config = config
        
        print(f"CodecAugmenter initialized:")
        print(f"  - Target sample rates: {config.target_sample_rates}")
        print(f"  - Packet loss range: {config.packet_loss_range[0]*100:.1f}%-{config.packet_loss_range[1]*100:.1f}%")
    
    def _apply_telephone_codec(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simulate telephone codec by downsampling and upsampling.
        
        This creates bandwidth limitation typical of telephone networks.
        
        Args:
            audio: Input audio signal.
            sr: Current sample rate.
            
        Returns:
            Audio with telephone codec simulation.
        """
        target_sr = random.choice(self.config.target_sample_rates)
        
        if target_sr == sr:
            return audio
        
        downsampled = utils.downsample_audio(audio, sr, target_sr)
        
        upsampled = utils.upsample_audio(downsampled, target_sr, sr)
        
        return upsampled
    
    def _apply_packet_loss(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simulate packet loss in VoIP transmission.
        
        Args:
            audio: Input audio signal.
            sr: Sample rate.
            
        Returns:
            Audio with simulated packet loss.
        """
        loss_rate = random.uniform(*self.config.packet_loss_range)
        
        return utils.simulate_packet_loss(audio, loss_rate, sr)
    
    def _apply_bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply bandpass filter to simulate channel bandwidth limitation.
        
        Typical telephone bandwidth: 300-3400 Hz
        
        Args:
            audio: Input audio signal.
            sr: Sample rate.
            
        Returns:
            Filtered audio signal.
        """
        from scipy import signal
        
        lowcut = 300.0
        highcut = 3400.0
        
        nyquist = sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if high >= 1.0:
            high = 0.99
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def _apply_quantization_noise(self, audio: np.ndarray, bits: int = 8) -> np.ndarray:
        """
        Simulate quantization noise from low-bitrate codec.
        
        Args:
            audio: Input audio signal.
            bits: Number of quantization bits.
            
        Returns:
            Quantized audio signal.
        """
        levels = 2 ** bits
        
        quantized = np.round(audio * (levels / 2)) / (levels / 2)
        
        quantized = np.clip(quantized, -1.0, 1.0)
        
        return quantized
    
    def augment(
        self,
        audio: np.ndarray,
        sr: int,
        return_metadata: bool = False
    ) -> np.ndarray:
        """
        Apply codec and channel degradation to audio.
        
        Process:
        1. Ensure audio is at target sample rate
        2. Apply telephone codec simulation (downsample/upsample)
        3. Apply bandpass filter for channel limitation
        4. Apply packet loss simulation
        5. Apply quantization noise
        6. Normalize and clip output
        
        Args:
            audio: Input audio signal.
            sr: Sample rate of input audio.
            return_metadata: If True, returns tuple (audio, metadata).
            
        Returns:
            Augmented audio signal, or tuple (audio, metadata) if return_metadata=True.
        """
        audio, sr = self._ensure_sample_rate(audio, sr)
        
        original_sr = sr
        
        augmented = self._apply_telephone_codec(audio, sr)
        
        if random.random() < 0.7:
            augmented = self._apply_bandpass_filter(augmented, sr)
        
        packet_loss_rate = 0.0
        if random.random() < 0.5:
            packet_loss_rate = random.uniform(*self.config.packet_loss_range)
            augmented = self._apply_packet_loss(augmented, sr)
        
        if random.random() < 0.3:
            bits = random.choice([8, 12])
            augmented = self._apply_quantization_noise(augmented, bits=bits)
        
        augmented = self._normalize_audio(augmented)
        augmented = self._clip_audio(augmented)
        
        if return_metadata:
            metadata = {
                "codec_sr": self.config.target_sample_rates[0] if len(self.config.target_sample_rates) > 0 else sr,
                "packet_loss": packet_loss_rate,
                "bandpass": True
            }
            return augmented, metadata
        
        return augmented
    
    def get_augmentation_label(
        self,
        codec_sr: int,
        packet_loss: float
    ) -> str:
        """
        Generate descriptive label for augmentation applied.
        
        Args:
            codec_sr: Sample rate used for codec simulation.
            packet_loss: Packet loss rate applied.
            
        Returns:
            Formatted augmentation label.
        """
        sr_khz = codec_sr // 1000
        loss_pct = int(packet_loss * 100)
        
        return f"CODEC_{sr_khz}K_LOSS{loss_pct}PCT"


