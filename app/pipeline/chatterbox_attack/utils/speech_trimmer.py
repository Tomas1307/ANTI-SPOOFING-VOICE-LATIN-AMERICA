"""
Two-stage speech endpoint trimmer for Chatterbox output.

Chatterbox Multilingual TTS frequently generates noise artifacts after the
actual speech content ends (triggered by the ``long_tail`` EOS forcing
mechanism). These artifacts are NOT silence — they can be loud, speech-like
noise that both energy-based trimming and VAD misclassify as speech.

This module uses a two-stage approach:
  1. **Duration ceiling**: Estimates expected speech duration from the input
     text word count and a conservative speaking rate. Any audio beyond
     ``max_duration_factor * expected_duration`` is unconditionally removed.
  2. **Silero VAD refinement**: Within the ceiling, detects the end of the
     last contiguous speech block (ignoring short isolated segments near the
     end that are likely noise misdetected as speech) and trims there.

The Silero VAD model is loaded once via torch.hub and reused across calls.
"""
import torch
import torchaudio
from loguru import logger


_vad_model = None
_vad_utils = None

SPEECH_GAP_THRESHOLD_MS = 500
MIN_TRAILING_SEGMENT_MS = 300


def _load_vad():
    """Load Silero VAD model from torch.hub (cached after first call).

    Returns:
        Tuple of (model, utils) where utils contains get_speech_timestamps
        and other helper functions.
    """
    global _vad_model, _vad_utils
    if _vad_model is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
    return _vad_model, _vad_utils


def _find_main_speech_end(speech_timestamps: list) -> int:
    """Find the end sample of the main speech block, ignoring trailing noise.

    Chatterbox noise artifacts can fool VAD into detecting short spurious
    speech segments after the real content. This function identifies the
    main contiguous speech block and ignores any short, isolated segments
    that appear after a gap.

    Strategy:
      - Walk the speech segments from the end.
      - If the last segment is short (< MIN_TRAILING_SEGMENT_MS) and there
        is a gap (> SPEECH_GAP_THRESHOLD_MS) before it, discard it and use
        the previous segment's end.
      - Repeat until a substantial segment or the main block is reached.

    Args:
        speech_timestamps: List of dicts with 'start' and 'end' keys
            (sample indices at 16 kHz) from Silero VAD.

    Returns:
        End sample index (at 16 kHz) of the main speech content.
    """
    if len(speech_timestamps) <= 1:
        return speech_timestamps[-1]["end"]

    gap_threshold_samples = int(SPEECH_GAP_THRESHOLD_MS * 16)
    min_segment_samples = int(MIN_TRAILING_SEGMENT_MS * 16)

    end_idx = len(speech_timestamps) - 1

    while end_idx > 0:
        current = speech_timestamps[end_idx]
        previous = speech_timestamps[end_idx - 1]

        segment_duration = current["end"] - current["start"]
        gap_before = current["start"] - previous["end"]

        if gap_before > gap_threshold_samples and segment_duration < min_segment_samples:
            logger.debug(
                f"Discarding trailing VAD segment [{current['start']}-{current['end']}] "
                f"(gap={gap_before / 16:.0f}ms, duration={segment_duration / 16:.0f}ms)"
            )
            end_idx -= 1
        else:
            break

    return speech_timestamps[end_idx]["end"]


def trim_trailing_noise(
    wav: torch.Tensor,
    sample_rate: int,
    margin_ms: int = 150,
    text: str = "",
    max_duration_factor: float = 1.5,
    min_words_per_second: float = 2.0,
) -> torch.Tensor:
    """Trim trailing non-speech noise from a waveform.

    Uses a two-stage approach for robust trimming:
      1. Apply a text-based duration ceiling (if text is provided).
      2. Use Silero VAD to find the actual speech end within the ceiling.

    Args:
        wav: Audio tensor of shape (1, num_samples) or (num_samples,).
        sample_rate: Sample rate of the waveform in Hz.
        margin_ms: Extra milliseconds to keep after last detected speech end.
        text: The text that was synthesised. Used to estimate expected duration
            and apply a hard ceiling. Pass empty string to skip ceiling.
        max_duration_factor: Multiplier for expected duration ceiling.
        min_words_per_second: Conservative speaking rate for ceiling estimation.
            2.0 words/sec is very slow Spanish speech — ensures we never clip
            legitimate content.

    Returns:
        Trimmed waveform tensor with the same number of dimensions as input.
    """
    was_2d = wav.dim() == 2
    wav_1d = wav.squeeze(0) if was_2d else wav

    if text:
        word_count = len(text.split())
        expected_duration = word_count / min_words_per_second
        max_duration = expected_duration * max_duration_factor
        max_samples = int(max_duration * sample_rate)

        if wav_1d.shape[-1] > max_samples:
            original_dur = wav_1d.shape[-1] / sample_rate
            logger.debug(
                f"Duration ceiling: {original_dur:.1f}s -> {max_duration:.1f}s "
                f"({word_count} words, {min_words_per_second} w/s * {max_duration_factor}x)"
            )
            wav_1d = wav_1d[:max_samples]

    model, utils = _load_vad()
    get_speech_timestamps = utils[0]

    if sample_rate != 16000:
        wav_16k = torchaudio.functional.resample(wav_1d, sample_rate, 16000)
    else:
        wav_16k = wav_1d

    speech_timestamps = get_speech_timestamps(
        wav_16k, model, sampling_rate=16000
    )

    if not speech_timestamps:
        if was_2d:
            wav_1d = wav_1d.unsqueeze(0)
        return wav_1d

    main_speech_end_16k = _find_main_speech_end(speech_timestamps)

    scale = sample_rate / 16000
    main_speech_end = int(main_speech_end_16k * scale)

    margin_samples = int(margin_ms * sample_rate / 1000)
    trim_point = min(main_speech_end + margin_samples, wav_1d.shape[-1])

    trimmed = wav_1d[:trim_point]

    if was_2d:
        trimmed = trimmed.unsqueeze(0)

    return trimmed
