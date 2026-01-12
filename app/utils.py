"""
Utility Functions for Audio Augmentation

Helper functions for audio processing, file I/O, and augmentation operations.
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Tuple, Optional, List
import random


def load_audio(filepath: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        filepath: Path to audio file.
        sr: Target sample rate.
        
    Returns:
        Tuple of (audio_data, sample_rate).
    """
    audio, sample_rate = librosa.load(filepath, sr=sr)
    return audio, sample_rate


def save_audio_flac(audio: np.ndarray, filepath: str, sr: int = 16000):
    """
    Save audio as FLAC file.
    
    Args:
        audio: Audio signal to save.
        filepath: Output file path.
        sr: Sample rate.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, audio, sr, format='FLAC')


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) of audio signal.
    
    Args:
        audio: Input audio signal.
        
    Returns:
        RMS value.
    """
    return np.sqrt(np.mean(audio ** 2))


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB.
    
    Args:
        signal: Clean signal.
        noise: Noise signal.
        
    Returns:
        SNR in decibels.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def mix_audio_with_snr(
    signal: np.ndarray, 
    noise: np.ndarray, 
    target_snr_db: float
) -> np.ndarray:
    """
    Mix signal with noise at specified SNR.
    
    Args:
        signal: Clean audio signal.
        noise: Noise signal (will be truncated or looped to match signal length).
        target_snr_db: Target Signal-to-Noise Ratio in dB.
        
    Returns:
        Mixed audio signal.
    """
    signal_length = len(signal)
    noise_length = len(noise)
    
    if noise_length < signal_length:
        repeats = int(np.ceil(signal_length / noise_length))
        noise = np.tile(noise, repeats)
    
    if len(noise) > signal_length:
        start_idx = random.randint(0, len(noise) - signal_length)
        noise = noise[start_idx:start_idx + signal_length]
    
    signal_rms = calculate_rms(signal)
    noise_rms = calculate_rms(noise)
    
    if signal_rms == 0 or noise_rms == 0:
        return signal
    
    snr_linear = 10 ** (target_snr_db / 20)
    
    scaling_factor = signal_rms / (noise_rms * snr_linear)
    noise_scaled = noise * scaling_factor
    
    mixed = signal + noise_scaled
    
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val * 0.99
    
    return mixed


def convolve_with_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Convolve audio with Room Impulse Response.
    
    Args:
        audio: Input audio signal.
        rir: Room Impulse Response.
        
    Returns:
        Convolved audio signal.
    """
    from scipy import signal as scipy_signal
    
    convolved = scipy_signal.fftconvolve(audio, rir, mode='same')
    
    max_val = np.max(np.abs(convolved))
    if max_val > 0:
        convolved = convolved / max_val * 0.99
    
    return convolved


def apply_clipping(audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """
    Apply hard clipping to audio signal.
    
    Args:
        audio: Input audio signal.
        threshold: Clipping threshold (0.0-1.0).
        
    Returns:
        Clipped audio signal.
    """
    return np.clip(audio, -threshold, threshold)


def downsample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Downsample audio to lower sample rate.
    
    Args:
        audio: Input audio signal.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Downsampled audio signal.
    """
    if orig_sr == target_sr:
        return audio
    
    downsampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return downsampled


def upsample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Upsample audio to higher sample rate.
    
    Args:
        audio: Input audio signal.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Upsampled audio signal.
    """
    if orig_sr == target_sr:
        return audio
    
    upsampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return upsampled


def simulate_packet_loss(audio: np.ndarray, loss_rate: float, sr: int) -> np.ndarray:
    """
    Simulate packet loss in audio transmission.
    
    Args:
        audio: Input audio signal.
        loss_rate: Packet loss rate (0.0-1.0).
        sr: Sample rate.
        
    Returns:
        Audio with simulated packet loss.
    """
    packet_size_ms = 20
    packet_size_samples = int(sr * packet_size_ms / 1000)
    
    num_packets = len(audio) // packet_size_samples
    
    audio_with_loss = audio.copy()
    
    for i in range(num_packets):
        if random.random() < loss_rate:
            start = i * packet_size_samples
            end = start + packet_size_samples
            audio_with_loss[start:end] = 0
    
    return audio_with_loss


def get_audio_files_recursive(directory: str, extensions: List[str] = ['.wav', '.flac']) -> List[str]:
    """
    Recursively get all audio files in directory.
    
    Args:
        directory: Root directory to search.
        extensions: List of audio file extensions to include.
        
    Returns:
        List of audio file paths.
    """
    audio_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    return sorted(audio_files)


def generate_audio_id(index: int, prefix: str = "LA_T") -> str:
    """
    Generate ASVspoof-style audio ID.
    
    Args:
        index: Sequential index.
        prefix: ID prefix.
        
    Returns:
        Formatted audio ID (e.g., "LA_T_0000001").
    """
    return f"{prefix}_{index:07d}"


def create_protocol_entry(
    speaker_id: str,
    audio_id: str,
    augmentation_type: str,
    key: str = "bonafide"
) -> str:
    """
    Create protocol file entry.
    
    Args:
        speaker_id: Speaker identifier.
        audio_id: Audio file identifier.
        augmentation_type: Type of augmentation applied.
        key: "bonafide" or "spoof".
        
    Returns:
        Protocol line string.
    """
    return f"{speaker_id} {audio_id} {augmentation_type} {key}"


def normalize_path(path: str) -> str:
    """
    Normalize file path for cross-platform compatibility.
    
    Args:
        path: Input path.
        
    Returns:
        Normalized path.
    """
    return str(Path(path).resolve())


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path.
    """
    os.makedirs(directory, exist_ok=True)
