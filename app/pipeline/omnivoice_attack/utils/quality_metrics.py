"""
Quality metrics utilities for OmniVoice synthetic speech validation.
"""
import librosa
import numpy as np
from loguru import logger


def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_duration: float = 1.0,
    sample_rate: int = 16000,
) -> bool:
    """Detect if audio contains excessive consecutive silence.

    Args:
        audio: Audio signal as a 1-D float32 array.
        threshold: RMS amplitude below which a frame is considered silent.
        min_duration: Minimum consecutive silent duration in seconds that
            triggers the flag.
        sample_rate: Audio sample rate in Hz.

    Returns:
        True if the longest consecutive silent region meets or exceeds
        ``min_duration``, False otherwise.
    """
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    silent_frames = rms < threshold

    max_consecutive = 0
    current = 0
    for is_silent in silent_frames:
        if is_silent:
            current += 1
        else:
            max_consecutive = max(max_consecutive, current)
            current = 0
    max_consecutive = max(max_consecutive, current)

    silence_duration = (max_consecutive * hop_length) / sample_rate
    if silence_duration >= min_duration:
        logger.debug(f"Silence detected: {silence_duration:.2f}s consecutive silent region")
        return True
    return False
