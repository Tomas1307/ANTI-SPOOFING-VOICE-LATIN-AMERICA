"""
Shared audio concatenation utilities for reference audio preparation.

Single source of truth replacing the duplicated `audio_concatenation.py`
files that previously lived in each per-attack pipeline. The implementation
incorporates the silence-boundary fix originally landed in
`omnivoice_attack/utils/audio_concatenation.py` on 2026-05-06: reference
clips must end on a clean silence boundary, never mid-phrase, because
diffusion-based TTS (OmniVoice) attempts to "complete" abrupt mid-word
cuts in the reference, manifesting as hallucinated content at the start
of generation.

Three rules govern reference construction:

    1. Concatenate complete files only -- never slice mid-file unless
       the first file alone exceeds the target duration.
    2. When the first file alone overflows, snap the cut to the nearest
       silent frame within +/- 1 s of the target.
    3. Always append a 200 ms trailing silence so the reference ends on
       a clear silence pattern regardless of where the last file ends.

Autoregressive TTS models (Chatterbox, Qwen, OuteTTS, etc.) tokenize the
reference and effectively ignore the trailing edge, so applying the
silence boundary rule is harmless for them; it is only strictly required
by OmniVoice's diffusion conditioning. Centralising the implementation
means a fix to any rule applies to every attack pipeline at once.

Dependencies are intentionally venv-agnostic (`numpy`, `librosa`) so this
module can be imported safely from every per-attack venv without
triggering pipeline-specific dependency loads.
"""
import numpy as np
import librosa
from pathlib import Path
from typing import List


def find_nearest_silence_boundary(
    audio: np.ndarray,
    sample_rate: int,
    target_seconds: float,
    search_window_seconds: float = 1.0,
    silence_threshold: float = 0.01,
) -> int:
    """Find the sample index of the nearest silent frame to a target time.

    Searches within +/- search_window_seconds of the target time for frames
    whose RMS energy falls below silence_threshold and returns the sample
    index of the closest one. Falls back to the exact target sample index
    if no silent frame is found within the window.

    Args:
        audio: 1-D audio signal in [-1.0, 1.0] range.
        sample_rate: Audio sample rate in Hz.
        target_seconds: Target time in seconds at which to cut.
        search_window_seconds: Half-width of the search window around target.
        silence_threshold: RMS amplitude below which a frame is silent.

    Returns:
        Sample index at which to slice. Always within
        [(target - search_window) * sample_rate, len(audio)].
    """
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    target_frame = int(target_seconds * sample_rate / hop_length)
    window_frames = int(search_window_seconds * sample_rate / hop_length)

    start_frame = max(0, target_frame - window_frames)
    end_frame = min(len(rms), target_frame + window_frames + 1)

    silent_frame_indices = [
        i for i in range(start_frame, end_frame) if rms[i] < silence_threshold
    ]

    if not silent_frame_indices:
        return min(int(target_seconds * sample_rate), len(audio))

    nearest = min(silent_frame_indices, key=lambda i: abs(i - target_frame))
    return min(nearest * hop_length, len(audio))


def concatenate_with_padding(
    audio_files: List[Path],
    target_duration: float = 15.0,
    sample_rate: int = 16000,
    silence_padding: float = 0.1,
    trailing_silence: float = 0.2,
) -> np.ndarray:
    """Concatenate audio files into a reference clip ending in silence.

    Builds a reference clip by stacking complete files separated by
    silence_padding seconds of silence. Files that would push the cumulative
    duration past target_duration are skipped, so the result lies in
    [3 s, target_duration + trailing_silence] under normal conditions.

    Edge case: if the first file alone is longer than target_duration, it is
    trimmed at the nearest silent frame within +/- 1 s of target_duration so
    the cut never lands mid-word. The 200 ms trailing silence is always
    appended regardless of how the body ended, guaranteeing the reference
    ends on a clean silence boundary -- which is required by OmniVoice's
    diffusion conditioning and harmless for other strategies.

    Args:
        audio_files: Audio file paths to concatenate, processed in sorted
            order.
        target_duration: Target duration of the speech body in seconds.
            Callers should pass their pipeline-specific value; the default
            15.0 matches most pipelines (OmniVoice and Chatterbox prefer
            10.0 and pass it explicitly).
        sample_rate: Target sample rate.
        silence_padding: Silence duration inserted between consecutive files.
        trailing_silence: Silence duration appended after the last file.

    Returns:
        Concatenated audio array with shape (samples,). Always ends with at
        least trailing_silence seconds of silence.

    Raises:
        ValueError: If audio_files is empty or no usable audio could be
            loaded.
    """
    if not audio_files:
        raise ValueError("audio_files cannot be empty")

    audios: List[np.ndarray] = []
    cumulative_duration = 0.0

    for file in sorted(audio_files):
        audio, _ = librosa.load(file, sr=sample_rate)
        duration = len(audio) / sample_rate

        if cumulative_duration + duration > target_duration:
            if not audios:
                trim_idx = find_nearest_silence_boundary(
                    audio,
                    sample_rate=sample_rate,
                    target_seconds=target_duration,
                    search_window_seconds=1.0,
                )
                audios.append(audio[:trim_idx])
            break

        audios.append(audio)
        silence_samples = int(silence_padding * sample_rate)
        audios.append(np.zeros(silence_samples))
        cumulative_duration += duration + silence_padding

    if not audios:
        raise ValueError(
            f"No usable audio for target duration {target_duration}s"
        )

    trailing_samples = int(trailing_silence * sample_rate)
    audios.append(np.zeros(trailing_samples))

    return np.concatenate(audios)


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """Normalize audio to a target RMS dB level.

    Args:
        audio: Input audio signal as a 1-D float32 array.
        target_level: Target RMS level in dB (default: -20.0).

    Returns:
        Normalized audio signal.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        scalar = 10 ** (target_level / 20) / rms
        audio = audio * scalar
    return audio


def clip_audio(audio: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Clip audio to prevent overflow.

    Args:
        audio: Input audio signal as a 1-D float32 array.
        max_val: Maximum absolute value (default: 1.0).

    Returns:
        Clipped audio signal.
    """
    return np.clip(audio, -max_val, max_val)
