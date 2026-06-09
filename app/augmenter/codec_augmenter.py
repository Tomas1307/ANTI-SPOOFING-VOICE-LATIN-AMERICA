"""
Codec and Channel Degradation Augmentation.

Applies REAL codec degradation (G.711 mu-law/A-law, AMR-NB, iLBC, Opus, AAC) via
the ffmpeg-backed ``torchaudio.io.AudioEffector`` round-trip implemented in
``app.augmenter.codec_backend``. Optionally simulates packet loss on top. Codecs
unavailable in the host ffmpeg build are detected once at construction time and
skipped. Loudness normalization is NOT done here; it is applied uniformly by the
orchestrator on write so loudness cannot leak augmentation type.
"""
import random
from typing import List, Optional

import numpy as np

import app.augmenter.codec_backend as codec_backend
from app.augmenter.base_augmenter import BaseAugmenter
from app.augmenter.codec_backend import DEFAULT_CODEC_REGISTRY
from app.augmenter.schemas.codec_rawboost_config import CodecConfigV2
import app.utils.utils as utils


class CodecAugmenter(BaseAugmenter):
    """
    Real codec/channel degradation augmentation.

    Encodes and decodes each clip through a randomly chosen real codec (drawn
    from the configured, ffmpeg-available set) and optionally simulates packet
    loss, modelling telephony and VoIP transmission.

    Attributes:
        config: CodecConfigV2 selecting the enabled codecs and packet-loss params.
        registry: Codec specification registry.
        available_codecs: Configured codec names confirmed available at runtime.
    """

    def __init__(self, config: CodecConfigV2, sample_rate: int = 16000):
        """
        Initialize codec augmenter and probe codec availability.

        Args:
            config: Configuration selecting codecs and packet-loss behaviour.
            sample_rate: Target sample rate for output (16 kHz).
        """
        super().__init__(sample_rate)
        self.config = config
        self.registry = DEFAULT_CODEC_REGISTRY

        probe = codec_backend.probe_available_codecs(self.registry)
        self.available_codecs: List[str] = [
            name for name in config.codec_set if probe.get(name, False)
        ]
        skipped = [name for name in config.codec_set if not probe.get(name, False)]

        print("CodecAugmenter initialized:")
        print(f"  - Enabled codecs: {self.available_codecs}")
        if skipped:
            print(f"  - Skipped (unavailable in ffmpeg build): {skipped}")

    def augment(
        self,
        audio: np.ndarray,
        sr: int,
        return_metadata: bool = False
    ) -> np.ndarray:
        """
        Apply a real codec round-trip (and optional packet loss) to audio.

        Args:
            audio: Input audio signal.
            sr: Sample rate of input audio.
            return_metadata: If True, returns tuple (audio, metadata).

        Returns:
            Augmented audio signal, or tuple (audio, metadata) if return_metadata.
        """
        audio, sr = self._ensure_sample_rate(audio, sr)
        augmented = audio

        codec_name: Optional[str] = None
        codec_sr = self.sample_rate
        bandpass = False
        bitrate = 0
        skipped = False
        fallback = False

        apply_codec = (
            self.available_codecs and random.random() < self.config.apply_probability
        )
        if apply_codec:
            codec_name = random.choices(
                self.available_codecs,
                weights=[self.config.codec_weights.get(c, 1.0) for c in self.available_codecs],
                k=1,
            )[0]
            spec = self.registry[codec_name]
            bitrate = random.choice(spec.bitrates) if spec.bitrates else 0
            degraded = codec_backend.apply_codec(
                augmented, spec, bitrate if bitrate else None
            )
            if degraded is None:
                # Codec failed at runtime: fall back to passthrough rather than crash.
                fallback = True
                codec_name = None
            else:
                augmented = degraded
                codec_sr = spec.sample_rate
                bandpass = spec.narrowband
        else:
            skipped = True

        packet_loss_rate = 0.0
        if random.random() < self.config.apply_packet_loss_prob:
            packet_loss_rate = random.uniform(*self.config.packet_loss_range)
            augmented = utils.simulate_packet_loss(augmented, packet_loss_rate, sr)

        augmented = self._clip_audio(augmented)

        if return_metadata:
            metadata = {
                "codec_sr": codec_sr,
                "packet_loss": packet_loss_rate,
                "bandpass": bandpass,
                "quantization_bits": 0,
                "codec": codec_name,
                "bitrate": bitrate,
                "skipped": skipped,
                "fallback": fallback,
            }
            return augmented, metadata

        return augmented

    def get_augmentation_label(
        self,
        codec_sr: int,
        packet_loss: float,
        bandpass: bool,
        quantization_bits: int,
        codec_name: Optional[str] = None
    ) -> str:
        """
        Generate a descriptive label for the codec augmentation applied.

        Args:
            codec_sr: Codec operating sample rate.
            packet_loss: Packet loss rate applied.
            bandpass: Whether the codec is narrowband (telephony-band).
            quantization_bits: Retained for metadata-contract compatibility (0).
            codec_name: Real codec name; None for a passthrough/skip.

        Returns:
            Formatted augmentation label.
        """
        if not codec_name:
            return "CODEC_SKIP_PASSTHROUGH"

        sr_khz = codec_sr // 1000
        loss_pct = int(packet_loss * 100)
        label = f"CODEC_{codec_name.upper()}_{sr_khz}K_LOSS{loss_pct}PCT"

        if bandpass:
            label += "_BP"

        return label
