"""
Qwen-specific artifact detection utilities.

Detects known Qwen3-TTS generation artifacts including truncation,
near-silent outputs, and duration anomalies. These checks supplement
the standard DNSMOS and speaker similarity validation.
"""
import numpy as np
from loguru import logger


def detect_truncation(
    audio: np.ndarray,
    text: str,
    sample_rate: int = 16000,
    min_words_per_second: float = 1.5
) -> bool:
    """Detect if generated audio is truncated relative to text length.

    Qwen3-TTS silently truncates on long texts without raising errors.
    This function checks if the audio duration is suspiciously short
    for the number of words in the input text.

    Args:
        audio: Audio signal as numpy array.
        text: Input text that was synthesized.
        sample_rate: Audio sample rate in Hz.
        min_words_per_second: Minimum expected speaking rate. Spanish
            averages 2-3 words/second; 1.5 is a conservative floor.

    Returns:
        True if truncation is detected, False otherwise.
    """
    audio_duration = len(audio) / sample_rate
    word_count = len(text.split())

    if audio_duration <= 0 or word_count <= 0:
        return True

    expected_min_duration = word_count / max(min_words_per_second, 0.1)
    # Allow generous margin: audio should be at least 30% of expected minimum
    threshold = expected_min_duration * 0.3

    is_truncated = audio_duration < threshold

    if is_truncated:
        logger.warning(
            f"Truncation detected: {audio_duration:.1f}s audio for "
            f"{word_count} words (expected >= {threshold:.1f}s)"
        )

    return is_truncated


def detect_low_energy(audio: np.ndarray, threshold: float = 0.001) -> bool:
    """Detect if audio is near-silent (garbled or failed generation).

    Qwen3-TTS can produce silent or extremely low-energy output
    without raising errors. This catches those cases.

    Args:
        audio: Audio signal as numpy array.
        threshold: RMS energy threshold below which audio is
            considered near-silent.

    Returns:
        True if audio energy is below threshold, False otherwise.
    """
    if len(audio) == 0:
        return True

    rms = np.sqrt(np.mean(audio ** 2))
    is_low = rms < threshold

    if is_low:
        logger.warning(
            f"Low energy detected: RMS={rms:.6f} < threshold={threshold}"
        )

    return is_low


def detect_duration_anomaly(
    audio_duration: float,
    min_duration: float = 0.5,
    max_duration: float = 30.0
) -> bool:
    """Detect if audio duration is outside acceptable bounds.

    Args:
        audio_duration: Duration of the generated audio in seconds.
        min_duration: Minimum acceptable duration in seconds.
        max_duration: Maximum acceptable duration in seconds.

    Returns:
        True if duration is anomalous (outside bounds), False otherwise.
    """
    is_anomalous = audio_duration < min_duration or audio_duration > max_duration

    if is_anomalous:
        logger.warning(
            f"Duration anomaly: {audio_duration:.1f}s "
            f"(expected {min_duration}-{max_duration}s)"
        )

    return is_anomalous
