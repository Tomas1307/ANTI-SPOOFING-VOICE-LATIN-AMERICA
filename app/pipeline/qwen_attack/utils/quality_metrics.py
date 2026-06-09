"""
Quality metrics for Qwen synthetic speech validation.

Silence detection utility used in Step 4 artifact checks.
WER/CER computation has moved to app/utils/wer_cer.py.
Parakeet transcription has moved to app/utils/parakeet_transcriber.py.
"""
import numpy as np
import librosa
from loguru import logger


def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_duration: float = 1.0,
    sample_rate: int = 16000,
) -> bool:
    """Detect if audio contains excessive consecutive silence.

    Args:
        audio: Audio signal as a numpy array.
        threshold: RMS energy threshold below which a frame is considered silent.
        min_duration: Minimum consecutive silence duration in seconds to flag.
        sample_rate: Audio sample rate in Hz.

    Returns:
        True if consecutive silence >= min_duration is detected, False otherwise.
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

    silence_seconds = (max_consecutive * hop_length) / sample_rate
    if silence_seconds >= min_duration:
        logger.warning(f"Excessive silence detected: {silence_seconds:.2f}s consecutive")
        return True
    return False
