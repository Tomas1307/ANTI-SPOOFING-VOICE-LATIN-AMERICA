"""
VAD-based speech endpoint trimmer for Chatterbox output.

Chatterbox Multilingual TTS frequently generates noise artifacts after the
actual speech content ends (triggered by the ``long_tail`` EOS forcing
mechanism). These artifacts are NOT silence — they are loud enough that
simple energy-threshold trimming (e.g. librosa.effects.trim) misses them.

This module uses Silero VAD to detect the last speech frame and trims
everything after it, with a small configurable margin.

The Silero VAD model is loaded once via torch.hub and reused across calls.
"""
import torch
import torchaudio


_vad_model = None
_vad_utils = None


def _load_vad():
    """Load Silero VAD model from torch.hub (cached after first call).

    Returns:
        Tuple of (model, get_speech_timestamps, read_audio, collect_chunks).
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


def trim_trailing_noise(
    wav: torch.Tensor,
    sample_rate: int,
    margin_ms: int = 150,
) -> torch.Tensor:
    """Trim trailing non-speech noise from a waveform using Silero VAD.

    Detects all speech segments in the waveform, finds the end of the last
    one, and trims everything after it (plus a configurable margin to avoid
    cutting off the last syllable).

    If VAD finds no speech at all, the original waveform is returned unchanged
    to avoid data loss.

    Args:
        wav: Audio tensor of shape (1, num_samples) or (num_samples,).
        sample_rate: Sample rate of the waveform in Hz.
        margin_ms: Extra milliseconds to keep after last detected speech end.

    Returns:
        Trimmed waveform tensor with the same number of dimensions as input.
    """
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]

    was_2d = wav.dim() == 2
    wav_1d = wav.squeeze(0) if was_2d else wav

    if sample_rate != 16000:
        wav_16k = torchaudio.functional.resample(wav_1d, sample_rate, 16000)
    else:
        wav_16k = wav_1d

    speech_timestamps = get_speech_timestamps(
        wav_16k, model, sampling_rate=16000
    )

    if not speech_timestamps:
        return wav

    last_speech_end_16k = speech_timestamps[-1]["end"]

    scale = sample_rate / 16000
    last_speech_end = int(last_speech_end_16k * scale)

    margin_samples = int(margin_ms * sample_rate / 1000)
    trim_point = min(last_speech_end + margin_samples, wav_1d.shape[-1])

    trimmed = wav_1d[:trim_point]

    if was_2d:
        trimmed = trimmed.unsqueeze(0)

    return trimmed
