"""
Audio concatenation utilities for FishGram reference audio preparation.

Reuses patterns from app/augmenter/base_augmenter.py for audio processing.
"""
import numpy as np
import librosa
from pathlib import Path
from typing import List


def concatenate_with_padding(
    audio_files: List[Path],
    target_duration: float = 15.0,
    sample_rate: int = 16000,
    silence_padding: float = 0.1
) -> np.ndarray:
    """Concatenate audio files with silence padding to reach target duration.

    Args:
        audio_files: List of audio file paths to concatenate
        target_duration: Target duration in seconds (default: 15.0)
        sample_rate: Target sample rate (default: 16000)
        silence_padding: Silence duration between files in seconds (default: 0.1)

    Returns:
        Concatenated audio array with shape (samples,)

    Raises:
        ValueError: If audio_files is empty or target_duration cannot be reached
    """
    if not audio_files:
        raise ValueError("audio_files cannot be empty")

    audios = []
    cumulative_duration = 0.0

    for file in sorted(audio_files):
        audio, sr = librosa.load(file, sr=sample_rate)
        duration = len(audio) / sr

        if cumulative_duration + duration >= target_duration:
            # Trim last file to exactly reach target
            needed_duration = target_duration - cumulative_duration
            needed_samples = int(needed_duration * sample_rate)
            audios.append(audio[:needed_samples])
            break
        else:
            # Add full audio + silence padding
            audios.append(audio)
            silence_samples = int(silence_padding * sample_rate)
            audios.append(np.zeros(silence_samples))
            cumulative_duration += duration + silence_padding

    if not audios:
        raise ValueError(f"Could not concatenate to target duration {target_duration}s")

    return np.concatenate(audios)


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level.

    Reused from app/augmenter/base_augmenter.py pattern.

    Args:
        audio: Input audio signal
        target_level: Target RMS level in dB (default: -20.0)

    Returns:
        Normalized audio signal
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        scalar = 10 ** (target_level / 20) / rms
        audio = audio * scalar
    return audio


def clip_audio(audio: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Clip audio to prevent overflow.

    Reused from app/augmenter/base_augmenter.py pattern.

    Args:
        audio: Input audio signal
        max_val: Maximum absolute value (default: 1.0)

    Returns:
        Clipped audio signal
    """
    return np.clip(audio, -max_val, max_val)
