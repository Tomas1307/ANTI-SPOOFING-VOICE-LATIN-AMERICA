"""
Audio concatenation utilities for Partial Spoof reference audio preparation.

Generic utility for concatenating speaker audio files to a target duration.
Kept local to avoid importing from pipeline packages whose __init__.py may
trigger heavy or unavailable dependencies (e.g., chatterbox_attack imports perth).
"""
import numpy as np
import librosa
from pathlib import Path
from typing import List


def concatenate_with_padding(
    audio_files: List[Path],
    target_duration: float = 15.0,
    sample_rate: int = 16000,
    silence_padding: float = 0.1,
) -> np.ndarray:
    """Concatenate audio files with silence padding to reach target duration.

    Args:
        audio_files: List of audio file paths to concatenate.
        target_duration: Target duration in seconds (default: 15.0).
        sample_rate: Target sample rate in Hz (default: 16000).
        silence_padding: Silence duration between files in seconds (default: 0.1).

    Returns:
        Concatenated audio array with shape (samples,).

    Raises:
        ValueError: If audio_files is empty.
    """
    if not audio_files:
        raise ValueError("audio_files cannot be empty")

    audios = []
    cumulative_duration = 0.0

    for file in sorted(audio_files):
        audio, sr = librosa.load(file, sr=sample_rate)
        duration = len(audio) / sr

        if cumulative_duration + duration >= target_duration:
            needed_duration = target_duration - cumulative_duration
            needed_samples = int(needed_duration * sample_rate)
            audios.append(audio[:needed_samples])
            break
        else:
            audios.append(audio)
            silence_samples = int(silence_padding * sample_rate)
            audios.append(np.zeros(silence_samples))
            cumulative_duration += duration + silence_padding

    if not audios:
        raise ValueError(f"Could not concatenate to target duration {target_duration}s")

    return np.concatenate(audios)
