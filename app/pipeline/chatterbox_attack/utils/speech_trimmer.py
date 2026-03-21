"""
Speech endpoint trimmer for Chatterbox trailing noise artifacts.

Chatterbox Multilingual TTS frequently generates loud noise artifacts after
the actual speech ends. The pattern is consistent:
  Speech (high, variable energy) -> Silence gap -> Noise (moderate, flat energy)

The noise can reach 15-40% of peak energy, defeating simple threshold trims.
However, the silence gap between speech and noise is always the LONGEST
silence gap in the audio — longer than any inter-word or inter-syllable pause.

Algorithm:
  1. Compute smoothed RMS energy envelope.
  2. Identify all silence gaps (runs of frames below 3% of peak).
  3. Select the longest gap that occurs in the second half of the audio.
  4. Verify by comparing mean energy before vs after the gap: if after-gap
     energy is lower than before-gap, it is noise.
  5. Trim at the start of the gap (plus a small margin).

No external models required — only numpy.
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
    """Trim trailing non-speech noise by finding the longest silence gap.

    Args:
        wav: Audio tensor of shape (1, num_samples) or (num_samples,).
        sample_rate: Sample rate of the waveform in Hz.
        margin_ms: Milliseconds to keep after the last speech frame before
            the gap (preserves tail of final syllable).
        frame_ms: RMS frame length in milliseconds.
        smooth_ms: Rolling average window for smoothing the RMS envelope.
        silence_threshold_ratio: Frames below this fraction of peak smoothed
            RMS are considered silent. 0.03 = 3% of peak.
        min_gap_ms: Minimum gap duration to consider as a speech endpoint.

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

    gaps = _find_silence_gaps(is_silent, hop_len, sample_rate, min_gap_ms)

    if not gaps:
        return wav

    # Find the longest gap in the second half of the audio
    midpoint = num_frames // 2
    candidate_gaps = [g for g in gaps if g["start"] >= midpoint]

    if not candidate_gaps:
        return wav

    longest_gap = max(candidate_gaps, key=lambda g: g["length"])

    # Verify: mean energy after gap should be lower than before gap
    before_mean = smoothed[:longest_gap["start"]].mean()
    after_start = longest_gap["end"]
    if after_start < num_frames:
        after_mean = smoothed[after_start:].mean()
        if after_mean >= before_mean:
            # After-gap is louder than before-gap — not a noise artifact
            return wav

    # Trim at the start of the gap + margin
    trim_frame = longest_gap["start"]
    trim_sample = trim_frame * hop_len + frame_len
    margin_samples = int(margin_ms * sample_rate / 1000)
    trim_point = min(trim_sample + margin_samples, len(audio))

    # Safety: don't trim more than 50% of the audio
    if trim_point < len(audio) * 0.5:
        logger.warning(
            f"Trim would remove >{50}% of audio "
            f"({len(audio) / sample_rate:.2f}s -> {trim_point / sample_rate:.2f}s). "
            f"Skipping for safety."
        )
        return wav

    original_dur = len(audio) / sample_rate
    trimmed_dur = trim_point / sample_rate
    logger.debug(
        f"Trimmed trailing noise: {original_dur:.2f}s -> {trimmed_dur:.2f}s "
        f"(removed {original_dur - trimmed_dur:.2f}s, "
        f"gap at {longest_gap['start'] * hop_len / sample_rate:.2f}s, "
        f"gap duration {longest_gap['length'] * hop_len / sample_rate * 1000:.0f}ms)"
    )

    trimmed = torch.from_numpy(audio[:trim_point])
    if was_2d:
        trimmed = trimmed.unsqueeze(0)

    return trimmed


def _find_silence_gaps(
    is_silent: np.ndarray,
    hop_len: int,
    sample_rate: int,
    min_gap_ms: int,
) -> list:
    """Find all silence gaps that exceed the minimum duration.

    Args:
        is_silent: Boolean array where True means the frame is below the
            silence threshold.
        hop_len: Hop length in samples between frames.
        sample_rate: Audio sample rate in Hz.
        min_gap_ms: Minimum gap duration in milliseconds.

    Returns:
        List of dicts with 'start', 'end', and 'length' (in frames) for
        each qualifying silence gap.
    """
    min_gap_frames = max(1, int(min_gap_ms / (hop_len / sample_rate * 1000)))
    gaps = []
    i = 0
    n = len(is_silent)

    while i < n:
        if is_silent[i]:
            gap_start = i
            while i < n and is_silent[i]:
                i += 1
            gap_length = i - gap_start
            if gap_length >= min_gap_frames:
                gaps.append({
                    "start": gap_start,
                    "end": i,
                    "length": gap_length,
                })
        else:
            i += 1

    return gaps
