"""
RIR and Noise Augmentation

Implements Room Impulse Response convolution and noise addition with controlled
SNR levels. Based on research from Sánchez et al. (2024) and best practices
from ASVspoof challenge data augmentation strategies.
"""
import os
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from app.augmenter.base_augmenter import BaseAugmenter
from app.config.augmentation_config import AugmentationConfigManager
from app.schema import RIRNoiseConfig, RoomSize, NoiseSource, Distance
import app.utils.utils as utils


class RIRAugmenter(BaseAugmenter):
    """
    RIR and noise augmentation implementation.
    
    Applies room impulse response convolution followed by noise addition
    at controlled SNR levels to simulate various acoustic environments.
    
    Attributes:
        config: RIRNoiseConfig object with augmentation parameters.
        rir_files: Dictionary mapping room sizes to RIR file paths.
        noise_files: Dictionary mapping noise sources to file paths.
    """
    
    def __init__(
        self, 
        config: RIRNoiseConfig,
        rir_root: str = "data/RIR",
        noise_root: str = "data/noise_dataset/musan",
        sample_rate: int = 16000
    ):
        """
        Initialize RIR and noise augmenter.
        
        Args:
            config: Configuration object for RIR and noise augmentation.
            rir_root: Root directory containing RIR files.
            noise_root: Root directory containing MUSAN noise files.
            sample_rate: Target sample rate for processing.
        """
        super().__init__(sample_rate)
        
        self.config = config
        self.rir_root = Path(rir_root)
        self.noise_root = Path(noise_root)
        
        self.rir_files = self._load_rir_files()
        self.noise_files = self._load_noise_files()
        
        print(f"RIRAugmenter initialized:")
        print(f"  - RIR files loaded: {sum(len(v) for v in self.rir_files.values())}")
        print(f"  - Noise files loaded: {sum(len(v) for v in self.noise_files.values())}")
    
    def _load_rir_files(self) -> Dict[RoomSize, List[str]]:
        """
        Load RIR file paths organized by room size.
        
        Returns:
            Dictionary mapping RoomSize to list of RIR file paths.
        """
        rir_files = {room_size: [] for room_size in RoomSize}
        
        simulated_path = self.rir_root / "simulated_rirs"
        
        if not simulated_path.exists():
            print(f"Warning: RIR path not found: {simulated_path}")
            return rir_files
        
        for room_size in RoomSize:
            room_path = simulated_path / room_size.value
            
            if room_path.exists():
                files = utils.get_audio_files_recursive(str(room_path))
                rir_files[room_size] = files
                print(f"  Loaded {len(files)} RIRs for {room_size.value}")
        
        return rir_files
    
    def _load_noise_files(self) -> Dict[NoiseSource, List[str]]:
        """
        Load noise file paths organized by source type.
        
        Returns:
            Dictionary mapping NoiseSource to list of noise file paths.
        """
        noise_files = {source: [] for source in NoiseSource}
        
        if not self.noise_root.exists():
            print(f"Warning: MUSAN path not found: {self.noise_root}")
            return noise_files
        
        noise_path = self.noise_root / "noise"
        if noise_path.exists():
            files = utils.get_audio_files_recursive(str(noise_path))
            noise_files[NoiseSource.NOISE] = files
            print(f"  Loaded {len(files)} noise files")
        
        speech_path = self.noise_root / "speech"
        if speech_path.exists():
            files = utils.get_audio_files_recursive(str(speech_path))
            noise_files[NoiseSource.SPEECH] = files
            print(f"  Loaded {len(files)} speech files")
        
        music_path = self.noise_root / "music"
        if music_path.exists():
            files = utils.get_audio_files_recursive(str(music_path))
            noise_files[NoiseSource.MUSIC] = files
            print(f"  Loaded {len(files)} music files")
        
        return noise_files
    
    def _sample_room_size(self) -> RoomSize:
        """
        Sample room size according to configured distribution.
        
        Returns:
            Sampled RoomSize.
        """
        room_sizes = list(self.config.room_sizes.keys())
        probabilities = [self.config.room_sizes[rs] for rs in room_sizes]
        
        return random.choices(room_sizes, weights=probabilities, k=1)[0]
    
    def _sample_noise_source(self) -> NoiseSource:
        """
        Sample noise source according to configured distribution.
        
        Returns:
            Sampled NoiseSource.
        """
        sources = list(self.config.noise_sources.keys())
        probabilities = [self.config.noise_sources[ns] for ns in sources]
        
        return random.choices(sources, weights=probabilities, k=1)[0]
    
    def _sample_snr(self) -> float:
        """
        Sample SNR value according to configured distribution.
        
        Returns:
            Sampled SNR in dB.
        """
        rand = random.random()
        
        low_min, low_max, low_prob = self.config.snr_distribution.low_range
        mid_min, mid_max, mid_prob = self.config.snr_distribution.mid_range
        high_min, high_max, high_prob = self.config.snr_distribution.high_range
        
        if rand < low_prob:
            return random.uniform(low_min, low_max)
        elif rand < low_prob + mid_prob:
            return random.uniform(mid_min, mid_max)
        else:
            return random.uniform(high_min, high_max)
    
    def _load_random_rir(self, room_size: RoomSize) -> Optional[np.ndarray]:
        """
        Load random RIR file for given room size.
        
        Args:
            room_size: Room size category.
            
        Returns:
            RIR signal or None if no files available.
        """
        available_files = self.rir_files.get(room_size, [])
        
        if not available_files:
            print(f"Warning: No RIR files for {room_size.value}")
            return None
        
        rir_file = random.choice(available_files)
        rir, _ = utils.load_audio(rir_file, sr=self.sample_rate)
        
        return rir
    
    def _load_random_noise(self, noise_source: NoiseSource) -> Optional[np.ndarray]:
        """
        Load random noise file for given source type.
        
        Args:
            noise_source: Noise source category.
            
        Returns:
            Noise signal or None if no files available.
        """
        available_files = self.noise_files.get(noise_source, [])
        
        if not available_files:
            print(f"Warning: No noise files for {noise_source.value}")
            return None
        
        noise_file = random.choice(available_files)
        noise, _ = utils.load_audio(noise_file, sr=self.sample_rate)
        
        return noise
    
    def augment(
        self, 
        audio: np.ndarray, 
        sr: int,
        return_metadata: bool = False
    ) -> np.ndarray:
        """
        Apply RIR convolution and noise addition to audio.
        
        Process:
        1. Ensure audio is at target sample rate
        2. Sample and apply RIR (room acoustics)
        3. Sample and mix noise at target SNR
        4. Normalize and clip output
        
        Args:
            audio: Input audio signal.
            sr: Sample rate of input audio.
            return_metadata: If True, returns tuple (audio, metadata).
            
        Returns:
            Augmented audio signal, or tuple (audio, metadata) if return_metadata=True.
        """
        audio, sr = self._ensure_sample_rate(audio, sr)
        
        room_size = self._sample_room_size()
        noise_source = self._sample_noise_source()
        target_snr = self._sample_snr()
        
        rir = self._load_random_rir(room_size)
        
        if rir is not None:
            audio_with_rir = utils.convolve_with_rir(audio, rir)
        else:
            audio_with_rir = audio
        
        noise = self._load_random_noise(noise_source)
        
        if noise is not None:
            augmented = utils.mix_audio_with_snr(audio_with_rir, noise, target_snr)
        else:
            augmented = audio_with_rir
        
        augmented = self._clip_audio(augmented)
        
        if return_metadata:
            metadata = {
                "room_size": room_size.value,
                "noise_source": noise_source.value,
                "snr_db": target_snr
            }
            return augmented, metadata
        
        return augmented
    
    def get_augmentation_label(
        self,
        room_size: str,
        noise_source: str,
        snr_db: float
    ) -> str:
        """
        Generate descriptive label for augmentation applied.
        
        Args:
            room_size: Room size used.
            noise_source: Noise source used.
            snr_db: SNR level applied.
            
        Returns:
            Formatted augmentation label.
        """
        room_short = room_size.replace("room", "").upper()
        noise_short = noise_source.upper()[:3]
        snr_int = int(snr_db)
        
        return f"RIR_{room_short}_{noise_short}_SNR{snr_int}"


