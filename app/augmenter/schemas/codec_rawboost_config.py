"""
Pydantic configuration models for the rebuilt RawBoost and codec augmenters.

These replace the legacy ``@dataclass`` ``RawBoostConfig`` / ``CodecConfig`` in
``app/schema.py``. They live here (not in ``app/schema.py``) so the augmenters do
not have to import that module, which pulls in torch/transformers and runs
``multiprocessing.set_start_method('spawn')`` at import time.
"""
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class CodecSpec(BaseModel):
    """
    Technical specification for one real codec applied via ffmpeg.

    Attributes:
        codec_format: ffmpeg container/muxer name (e.g. "wav", "amr", "ogg").
        encoder: ffmpeg encoder name (e.g. "pcm_mulaw", "libopus", "aac").
        sample_rate: Operating sample rate the codec runs at (Hz).
        bitrates: Candidate bitrates (bps) to sample from, or None for codecs
            whose rate is fixed by the encoder (e.g. G.711 companding).
        narrowband: True for telephony-band codecs (used for the metadata
            ``bandpass`` flag and label).
    """

    codec_format: str
    encoder: str
    sample_rate: int
    bitrates: Optional[List[int]] = None
    narrowband: bool = False


class RawBoostParams(BaseModel):
    """
    Parameter ranges for the official RawBoost algorithm (Tak et al., 2022).

    The field names intentionally match the reference implementation's argparse
    namespace, so an instance of this model can be passed directly as the
    ``args`` object expected by ``process_Rawboost_feature`` (attribute access on
    ``args.nBands``, ``args.P``, ``args.SNRmin``, ... resolves to these fields).
    """

    # --- LnL: linear and non-linear convolutive noise (multiband notch + Hammerstein) ---
    N_f: int = Field(default=5, description="Number of notch filters in the multiband LnL filter")
    nBands: int = Field(default=5, description="Number of notch frequency bands")
    minF: int = Field(default=20, description="Minimum notch centre frequency in Hz")
    maxF: int = Field(default=8000, description="Maximum notch centre frequency in Hz")
    minBW: int = Field(default=100, description="Minimum notch bandwidth in Hz")
    maxBW: int = Field(default=1000, description="Maximum notch bandwidth in Hz")
    minCoeff: int = Field(default=10, description="Minimum number of FIR coefficients")
    maxCoeff: int = Field(default=100, description="Maximum number of FIR coefficients")
    minG: int = Field(default=0, description="Minimum gain (dB) applied per notch")
    maxG: int = Field(default=0, description="Maximum gain (dB) applied per notch")
    minBiasLinNonLin: int = Field(
        default=5, description="Minimum bias separating the linear and non-linear branches (dB)"
    )
    maxBiasLinNonLin: int = Field(
        default=20, description="Maximum bias separating the linear and non-linear branches (dB)"
    )

    # --- ISD: impulsive signal-dependent additive noise ---
    P: int = Field(default=10, description="Maximum percentage of samples receiving impulsive noise")
    g_sd: int = Field(default=2, description="Gain parameter for the signal-dependent impulsive noise")

    # --- SSI: stationary signal-independent additive noise (FIR-colored white noise) ---
    SNRmin: int = Field(default=10, description="Minimum SNR (dB) for the stationary additive noise")
    SNRmax: int = Field(default=40, description="Maximum SNR (dB) for the stationary additive noise")


class RawBoostConfigV2(BaseModel):
    """
    Configuration for the RawBoost augmenter using the real LnL/ISD/SSI algorithm.

    Attributes:
        algo: Fixed RawBoost algorithm id (1-7), or 0 to draw one at random per
            clip from ``algo_choices``. Reference algos: 1=LnL, 2=ISD, 3=SSI,
            4=series(LnL->ISD->SSI), 5=series(LnL->ISD), 6=series(LnL->SSI),
            7=parallel(LnL || ISD).
        algo_choices: Algorithms sampled when ``algo == 0``.
        params: Parameter ranges passed through to the reference implementation.
    """

    algo: int = Field(default=0, description="Fixed RawBoost algo 1-7, or 0 to sample per clip")
    algo_choices: List[int] = Field(
        default_factory=lambda: [4, 5, 7],
        description="RawBoost algorithms sampled per clip when algo == 0",
    )
    algo_weights: Dict[int, float] = Field(
        default_factory=lambda: {4: 0.5, 5: 0.3, 7: 0.2},
        description="Relative sampling weight per algorithm (favours the full "
                    "LnL->ISD->SSI pipeline). Missing entries default to 1.0.",
    )
    params: RawBoostParams = Field(default_factory=RawBoostParams)


class CodecConfigV2(BaseModel):
    """
    Configuration for the real codec augmenter (torchaudio AudioEffector / ffmpeg).

    The per-codec technical specs (container format, ffmpeg encoder, operating
    sample rate, bitrate options, narrowband flag) live in
    ``app.augmenter.codec_backend.DEFAULT_CODEC_REGISTRY`` as the single source
    of truth. This model only selects which codec names are enabled and controls
    optional packet-loss simulation.

    Attributes:
        codec_set: Codec names to draw from per clip. Default covers the full
            "all threats" range: narrowband telephony (G.711 mu-law/A-law,
            AMR-NB, iLBC) plus broadband (Opus, AAC). Names unavailable in the
            host ffmpeg build are disabled by the backend probe at runtime.
        apply_packet_loss_prob: Probability of also simulating packet loss.
        packet_loss_range: (min, max) packet-loss fraction.
        apply_probability: Probability the codec augmentation is applied at all.
    """

    codec_set: List[str] = Field(
        default_factory=lambda: ["g711_ulaw", "g711_alaw", "amr_nb", "ilbc", "opus", "aac"],
        description="Enabled codec names (subject to ffmpeg availability probe)",
    )
    codec_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "g711_ulaw": 0.25,
            "g711_alaw": 0.15,
            "amr_nb": 0.20,
            "ilbc": 0.05,
            "opus": 0.25,
            "aac": 0.10,
        },
        description="Relative sampling weight per codec (favours the most common "
                    "deployment channels). Missing entries default to 1.0; weights "
                    "are renormalized over the codecs actually available at runtime.",
    )
    apply_packet_loss_prob: float = Field(
        default=0.3, description="Probability of simulating packet loss on top of the codec"
    )
    packet_loss_range: Tuple[float, float] = Field(
        default=(0.0, 0.05), description="(min, max) packet-loss fraction"
    )
    apply_probability: float = Field(
        default=1.0, description="Probability the codec degradation is applied"
    )
