"""
Real codec degradation backend (torchaudio AudioEffector / external ffmpeg).

This module isolates the torch/torchaudio dependency for codec round-trips so
that the rest of the augmentation pipeline stays numpy-based. It exposes the
canonical codec registry (the single source of truth for codec specs), a runtime
capability probe, and a single ``apply_codec`` entry point that encodes then
decodes an audio array.

Two execution backends exist:

* ``torio`` runs in-process via ``torchaudio.io.AudioEffector``, which links the
  system ffmpeg libraries. ``torchaudio.functional.apply_codec`` was removed in
  torchaudio >= 2.1, hence AudioEffector.
* ``ffmpeg_bin`` shells out to an external ffmpeg binary. This exists because
  Debian/Ubuntu build ffmpeg without patent-encumbered codecs such as AMR-NB,
  so the system libavcodec cannot encode them at any container setting. The
  binary is selected with the ``MARSA_FFMPEG_BINARY`` environment variable; when
  it is left unset the affected codecs are simply reported unavailable by the
  probe and excluded from sampling.

Failures are recorded per codec in ``_CODEC_ERRORS`` and exposed through
``get_codec_errors()``. An earlier revision swallowed every exception and
returned ``None``, which made a wrong encoder name indistinguishable from a
genuinely missing codec and allowed a silently degraded codec set to survive a
full production run.
"""
import os
import subprocess
import tempfile
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.io import AudioEffector, CodecConfig

from app.augmenter.schemas.codec_rawboost_config import CodecSpec

SAMPLE_RATE = 16000

# External ffmpeg binary used by the "ffmpeg_bin" backend. Defaults to whatever
# "ffmpeg" resolves to on PATH; point it at a build that carries the extra
# codecs (for example a static release) to enable them.
FFMPEG_BINARY = os.environ.get("MARSA_FFMPEG_BINARY", "ffmpeg")

# Single source of truth for codec specifications.
#
# Encoder names are ffmpeg ENCODER names, not codec IDs: ffmpeg exposes AMR-NB
# encoding as "libopencore_amrnb" and Speex as "libspeex". Container names must
# be valid as both muxer and demuxer, since the round-trip re-reads what it
# writes; "adts" and "ipod" are write-only and fail on read-back.
#
# Speex occupies the legacy-VoIP slot originally assigned to iLBC. iLBC is
# absent from the stock Debian/Ubuntu ffmpeg and from the common static
# releases, whereas Speex is the pre-Opus open VoIP codec, fills the same role,
# and ships in the stock build.
DEFAULT_CODEC_REGISTRY: Dict[str, CodecSpec] = {
    "g711_ulaw": CodecSpec(
        codec_format="wav", encoder="pcm_mulaw", sample_rate=8000,
        bitrates=None, narrowband=True, backend="torio",
    ),
    "g711_alaw": CodecSpec(
        codec_format="wav", encoder="pcm_alaw", sample_rate=8000,
        bitrates=None, narrowband=True, backend="torio",
    ),
    "amr_nb": CodecSpec(
        codec_format="amr", encoder="libopencore_amrnb", sample_rate=8000,
        bitrates=[4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200],
        narrowband=True, backend="ffmpeg_bin",
    ),
    "speex": CodecSpec(
        codec_format="ogg", encoder="libspeex", sample_rate=8000,
        bitrates=None, narrowband=True, backend="torio",
    ),
    "opus": CodecSpec(
        codec_format="ogg", encoder="libopus", sample_rate=48000,
        bitrates=[8000, 12000, 16000, 24000, 32000],
        narrowband=False, backend="torio",
    ),
    "aac": CodecSpec(
        codec_format="mpegts", encoder="aac", sample_rate=16000,
        bitrates=[24000, 32000, 48000, 64000],
        narrowband=False, backend="torio",
    ),
}

_PROBE_CACHE: Optional[Dict[str, bool]] = None
_CODEC_ERRORS: Dict[str, str] = {}


def get_codec_errors() -> Dict[str, str]:
    """
    Return the first recorded failure message for each codec that failed.

    Returns:
        Mapping of codec encoder name to the exception text observed.
    """
    return dict(_CODEC_ERRORS)


def _to_time_channels(audio: np.ndarray) -> torch.Tensor:
    """Convert a 1-D numpy array to a (time, 1) float32 tensor."""
    arr = np.ascontiguousarray(audio, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(1)


def _round_trip_torio(
    audio: np.ndarray,
    spec: CodecSpec,
    bitrate: Optional[int],
) -> Optional[np.ndarray]:
    """
    Encode and decode in-process through torchaudio's AudioEffector.

    Args:
        audio: Mono float32 audio at SAMPLE_RATE.
        spec: Codec specification.
        bitrate: Bitrate in bps, or None for the encoder default.

    Returns:
        Degraded audio at SAMPLE_RATE, or None if the round-trip failed.
    """
    try:
        wav = _to_time_channels(audio)

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
        out = effector.apply(wav, spec.sample_rate)

        out = out[:, 0:1]
        if spec.sample_rate != SAMPLE_RATE:
            out = torchaudio.functional.resample(
                out.transpose(0, 1), spec.sample_rate, SAMPLE_RATE
            ).transpose(0, 1)

        return out[:, 0].cpu().numpy().astype(np.float32)
    except Exception as error:
        _CODEC_ERRORS.setdefault(spec.encoder, f"{type(error).__name__}: {error}")
        return None


def _round_trip_ffmpeg_bin(
    audio: np.ndarray,
    spec: CodecSpec,
    bitrate: Optional[int],
) -> Optional[np.ndarray]:
    """
    Encode and decode by shelling out to the external ffmpeg binary.

    Resampling in both directions is delegated to ffmpeg via ``-ar``, so the
    caller always supplies and receives audio at SAMPLE_RATE.

    Args:
        audio: Mono float32 audio at SAMPLE_RATE.
        spec: Codec specification.
        bitrate: Bitrate in bps, or None for the encoder default.

    Returns:
        Degraded audio at SAMPLE_RATE, or None if the round-trip failed.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="marsa_codec_") as tmp:
            src = os.path.join(tmp, "src.wav")
            enc = os.path.join(tmp, f"enc.{spec.codec_format}")
            dec = os.path.join(tmp, "dec.wav")

            sf.write(src, audio, SAMPLE_RATE, subtype="PCM_16")

            encode_cmd = [
                FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
                "-i", src, "-ar", str(spec.sample_rate), "-ac", "1",
                "-c:a", spec.encoder,
            ]
            if bitrate:
                encode_cmd += ["-b:a", str(bitrate)]
            encode_cmd.append(enc)

            decode_cmd = [
                FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
                "-i", enc, "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-f", "wav", dec,
            ]

            for cmd in (encode_cmd, decode_cmd):
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                if proc.returncode != 0:
                    _CODEC_ERRORS.setdefault(
                        spec.encoder,
                        f"ffmpeg exit {proc.returncode}: "
                        f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no stderr'}"
                    )
                    return None

            result, _ = sf.read(dec, dtype="float32")
            if result.ndim > 1:
                result = result[:, 0]
            return result.astype(np.float32)
    except Exception as error:
        _CODEC_ERRORS.setdefault(spec.encoder, f"{type(error).__name__}: {error}")
        return None


def apply_codec(
    audio: np.ndarray,
    spec: CodecSpec,
    bitrate: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Encode then decode a mono 16 kHz signal through a real codec.

    The signal is round-tripped through the backend named by ``spec.backend``
    and trimmed or zero-padded back to the original length.

    Args:
        audio: Mono float32 audio at 16 kHz.
        spec: Codec specification.
        bitrate: Bitrate in bps, or None to use the encoder default.

    Returns:
        The degraded audio as float32 at 16 kHz, or None if the round-trip
        failed (see ``get_codec_errors()`` for the reason).
    """
    target_len = len(audio)

    if spec.backend == "ffmpeg_bin":
        result = _round_trip_ffmpeg_bin(audio, spec, bitrate)
    else:
        result = _round_trip_torio(audio, spec, bitrate)

    if result is None:
        return None

    if len(result) >= target_len:
        return result[:target_len]
    return np.pad(result, (0, target_len - len(result)))


def probe_available_codecs(
    registry: Optional[Dict[str, CodecSpec]] = None,
    force: bool = False,
) -> Dict[str, bool]:
    """
    Determine which registry codecs the host can actually run.

    Each codec is exercised once on a short noise signal; any that fails is
    marked unavailable and its error recorded. The result is cached for the
    process.

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
