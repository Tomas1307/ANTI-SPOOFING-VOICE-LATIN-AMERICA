"""
Gap-detection speech endpoint trimmer for Chatterbox output.

Chatterbox Multilingual TTS frequently generates noise artifacts after the
actual speech content ends (triggered by the ``long_tail`` EOS forcing
mechanism). These artifacts are NOT silence — they can reach 8-20% of peak
energy, defeating simple threshold-based trimming.

However, a consistent pattern exists: speech ends, energy drops to near zero
for 100-300 ms, then the noise artifact begins. This module exploits that
gap by:
  1. Computing a smoothed RMS energy envelope in short frames.
  2. Scanning backwards from the end to find a sustained silence gap (energy
     below 3% of peak for at least ``min_gap_ms`` milliseconds).
  3. Trimming at the START of that gap (with a small margin to preserve the
     last syllable's tail).

If no trailing gap is found, the audio is returned unchanged — it is either
clean or the noise is too merged with speech to safely remove.

No external models or downloads required — only numpy.
"""
import numpy as np
import torch
from loguru import logger


def trim_trailing_noise(
    wav: torch.Tensor,
    sample_rate: int,
    margin_ms: int = 150,
    frame_ms: int = 25,
    smooth_ms: int = 100,
    silence_threshold_ratio: float = 0.03,
    min_gap_ms: int = 150,
) -> torch.Tensor:
    """Trim trailing non-speech noise by detecting the post-speech silence gap.

    Chatterbox noise artifacts are preceded by a brief silence gap after the
    real speech ends. This function finds that gap and trims at its start.

    Args:
        wav: Audio tensor of shape (1, num_samples) or (num_samples,).
        sample_rate: Sample rate of the waveform in Hz.
        margin_ms: Milliseconds to keep after the last speech frame before
            the gap (preserves tail of final syllable).
        frame_ms: RMS frame length in milliseconds.
        smooth_ms: Rolling average window size in milliseconds for smoothing
            the RMS envelope.
        silence_threshold_ratio: Frames with smoothed RMS below this fraction
            of peak are considered silence. 0.03 = 3% of peak.
        min_gap_ms: Minimum duration in milliseconds for a silence region to
            qualify as the post-speech gap. Must be long enough to distinguish
            from inter-syllable pauses (~50-80 ms) but short enough to catch
            the Chatterbox artifact gap (~100-300 ms).

    Returns:
        Trimmed waveform tensor with the same number of dimensions as input.
    """
    was_2d = wav.dim() == 2
    audio = wav.squeeze(0).cpu().numpy().astype(np.float32)

    frame_len = int(frame_ms * sample_rate / 1000)
    hop_len = frame_len // 2
    num_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_len
        frame = audio[start : start + frame_len]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    smooth_frames = max(1, int(smooth_ms / frame_ms))
    kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
    smoothed = np.convolve(rms, kernel, mode="same")

    peak_rms = smoothed.max()
    if peak_rms < 1e-8:
        return wav

    silence_threshold = peak_rms * silence_threshold_ratio
    is_silent = smoothed < silence_threshold

    min_gap_frames = max(1, int(min_gap_ms / (hop_len / sample_rate * 1000)))

    # Scan backwards from the end looking for a silence gap that is
    # followed by non-silence (the noise artifact). The gap must be at
    # least min_gap_frames long and must NOT extend to the very end of
    # the audio (that would be trailing silence, not a gap before noise).
    #
    # Walk from the end:
    #   1. Skip any trailing non-silence (the noise artifact itself).
    #   2. Count consecutive silence frames (the gap).
    #   3. If the gap is long enough, we found it. Trim at its start.

    i = num_frames - 1

    # Step 1: skip trailing non-silence (noise artifact)
    while i >= 0 and not is_silent[i]:
        i -= 1

    if i < 0:
        return wav

    noise_start_frame = i + 1

    # Step 2: count consecutive silence frames (the gap)
    gap_end = i
    while i >= 0 and is_silent[i]:
        i -= 1

    gap_start = i + 1
    gap_length = gap_end - gap_start + 1

    if gap_length < min_gap_frames:
        return wav

    # Step 3: verify there is actual speech BEFORE the gap
    # (at least 500ms of audio before gap_start to be meaningful)
    min_speech_frames = int(500 / (hop_len / sample_rate * 1000))
    if gap_start < min_speech_frames:
        return wav

    # Trim at gap_start (last speech frame before the gap) + margin
    trim_frame = gap_start
    trim_sample = trim_frame * hop_len + frame_len
    margin_samples = int(margin_ms * sample_rate / 1000)
    trim_point = min(trim_sample + margin_samples, len(audio))

    # Safety: don't trim more than 40% of the audio
    if trim_point < len(audio) * 0.6:
        logger.warning(
            f"Gap-based trim would remove >{40}% of audio "
            f"({len(audio) / sample_rate:.2f}s -> {trim_point / sample_rate:.2f}s). "
            f"Skipping trim for safety."
        )
        return wav

    original_dur = len(audio) / sample_rate
    trimmed_dur = trim_point / sample_rate
    logger.debug(
        f"Trimmed trailing noise: {original_dur:.2f}s -> {trimmed_dur:.2f}s "
        f"(gap at {gap_start * hop_len / sample_rate:.2f}s, "
        f"noise at {noise_start_frame * hop_len / sample_rate:.2f}s)"
    )

    trimmed = torch.from_numpy(audio[:trim_point])
    if was_2d:
        trimmed = trimmed.unsqueeze(0)

    return trimmed
