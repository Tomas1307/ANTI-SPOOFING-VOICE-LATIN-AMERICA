"""
Real codec degradation backend (torchaudio AudioEffector / ffmpeg).

This module isolates the torch/torchaudio dependency for codec round-trips so
that the rest of the augmentation pipeline stays numpy-based. It exposes the
canonical codec registry (the single source of truth for codec specs), a runtime
capability probe (codecs missing from the host ffmpeg build are disabled), and a
single ``apply_codec`` entry point that encodes then decodes an audio array.

``torchaudio.functional.apply_codec`` was removed in torchaudio >= 2.1, so this
uses the modern ``torchaudio.io.AudioEffector`` (ffmpeg-backed). The ffmpeg
binary must be discoverable by torchaudio at runtime.
"""
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio
from torchaudio.io import AudioEffector, CodecConfig

from app.augmenter.schemas.codec_rawboost_config import CodecSpec

SAMPLE_RATE = 16000

# Single source of truth for codec specifications. Encoder/format names are
# matched to standard ffmpeg builds; any not present in the host build are
# disabled by probe_available_codecs() at runtime.
DEFAULT_CODEC_REGISTRY: Dict[str, CodecSpec] = {
    "g711_ulaw": CodecSpec(
        codec_format="wav", encoder="pcm_mulaw", sample_rate=8000,
        bitrates=None, narrowband=True,
    ),
    "g711_alaw": CodecSpec(
        codec_format="wav", encoder="pcm_alaw", sample_rate=8000,
        bitrates=None, narrowband=True,
    ),
    "amr_nb": CodecSpec(
        codec_format="amr", encoder="amr_nb", sample_rate=8000,
        bitrates=[4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200], narrowband=True,
    ),
    "ilbc": CodecSpec(
        codec_format="ipod", encoder="ilbc", sample_rate=8000,
        bitrates=None, narrowband=True,
    ),
    "opus": CodecSpec(
        codec_format="ogg", encoder="libopus", sample_rate=48000,
        bitrates=[8000, 12000, 16000, 24000, 32000], narrowband=False,
    ),
    "aac": CodecSpec(
        codec_format="adts", encoder="aac", sample_rate=16000,
        bitrates=[24000, 32000, 48000, 64000], narrowband=False,
    ),
}

_PROBE_CACHE: Optional[Dict[str, bool]] = None


def _to_time_channels(audio: np.ndarray) -> torch.Tensor:
    """Convert a 1-D numpy array to a (time, 1) float32 tensor."""
    arr = np.ascontiguousarray(audio, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(1)


def apply_codec(
    audio: np.ndarray,
    spec: CodecSpec,
    bitrate: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Encode then decode a mono 16 kHz signal through a real codec.

    The signal is resampled to the codec's operating rate, round-tripped through
    the ffmpeg encoder/decoder, resampled back to 16 kHz, and trimmed/padded to
    the original length. All work is done on CPU so it runs without a GPU.

    Args:
        audio: Mono float32 audio at 16 kHz.
        spec: Codec specification.
        bitrate: Bitrate in bps, or None to use the encoder default.

    Returns:
        The degraded audio as float32 at 16 kHz, or None if the codec round-trip
        fails (e.g. the encoder is unavailable in the host ffmpeg build).
    """
    target_len = len(audio)
    try:
        wav = _to_time_channels(audio)  # (T, 1)

        if spec.sample_rate != SAMPLE_RATE:
            wav = torchaudio.functional.resample(
                wav.transpose(0, 1), SAMPLE_RATE, spec.sample_rate
            ).transpose(0, 1)

        codec_config = CodecConfig(bit_rate=bitrate) if bitrate else None
        effector = AudioEffector(
            format=spec.codec_format,
            encoder=spec.encoder,
            codec_config=codec_config,
        )
        out = effector.apply(wav, spec.sample_rate)  # (T2, C)

        out = out[:, 0:1]  # keep mono as (T2, 1)
        if spec.sample_rate != SAMPLE_RATE:
            out = torchaudio.functional.resample(
                out.transpose(0, 1), spec.sample_rate, SAMPLE_RATE
            ).transpose(0, 1)

        result = out[:, 0].cpu().numpy().astype(np.float32)
    except Exception:
        return None

    if len(result) >= target_len:
        return result[:target_len]
    return np.pad(result, (0, target_len - len(result)))


def probe_available_codecs(
    registry: Optional[Dict[str, CodecSpec]] = None,
    force: bool = False,
) -> Dict[str, bool]:
    """
    Determine which registry codecs the host ffmpeg build can actually run.

    Each codec is exercised once on a short noise signal; any that raises is
    marked unavailable. The result is cached for the process.

    Args:
        registry: Codec registry to probe (defaults to DEFAULT_CODEC_REGISTRY).
        force: Re-run the probe even if a cached result exists.

    Returns:
        Mapping of codec name to availability (True/False).
    """
    global _PROBE_CACHE
    if _PROBE_CACHE is not None and not force and registry is None:
        return _PROBE_CACHE

    registry = registry if registry is not None else DEFAULT_CODEC_REGISTRY
    probe_signal = (np.random.randn(SAMPLE_RATE) * 0.05).astype(np.float32)

    available: Dict[str, bool] = {}
    for name, spec in registry.items():
        bitrate = spec.bitrates[0] if spec.bitrates else None
        available[name] = apply_codec(probe_signal, spec, bitrate) is not None

    if registry is DEFAULT_CODEC_REGISTRY:
        _PROBE_CACHE = available
    return available
