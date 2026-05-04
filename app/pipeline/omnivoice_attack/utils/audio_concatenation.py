"""
Audio concatenation utilities for OmniVoice reference audio preparation.
"""
import numpy as np
import librosa
from pathlib import Path
from typing import List


def concatenate_with_padding(
    audio_files: List[Path],
    target_duration: float = 10.0,
    sample_rate: int = 16000,
    silence_padding: float = 0.1
) -> np.ndarray:
    """Concatenate audio files with silence padding to reach target duration.

    OmniVoice recommends a 3-10 second reference clip; longer references
    degrade cloning quality (per upstream docs).

    Args:
        audio_files: List of audio file paths to concatenate.
        target_duration: Target duration in seconds.
        sample_rate: Target sample rate.
        silence_padding: Silence duration between files in seconds.

    Returns:
        Concatenated audio array with shape (samples,).

    Raises:
        ValueError: If audio_files is empty or target_duration cannot be reached.
    """
    if not audio_files:
        raise ValueError("audio_files cannot be empty")

    audios = []
    cumulative_duration = 0.0

    for file in sorted(audio_files):
        audio, _ = librosa.load(file, sr=sample_rate)
        duration = len(audio) / sample_rate

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
