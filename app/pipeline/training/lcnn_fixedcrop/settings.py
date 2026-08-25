"""
Backend-specific settings for the fixed-750-frame-truncation LCNN detector.
"""
from pydantic import BaseModel, Field


class LCNNFixedCropBackendSettings(BaseModel):
    """Configuration for Xin Wang's "fixed" LCNN backend.

    Attributes:
        CHECKPOINT_PATH: Path to the published .pt state dict. Named
            ``trained_network_att.pt`` on ml-server03 despite implementing the
            architecture from the ``lfcc-lcnn-fixed-p2s`` directory -- verified
            by matching parameter shapes before this backend was written; do
            not "fix" this path by pointing it at ``trained_network_fix.pt``,
            which holds a different architecture entirely.
        FRAME_LENGTH: LFCC analysis frame length, in waveform samples.
        FRAME_SHIFT: LFCC analysis frame shift, in waveform samples.
        FFT_N: FFT length.
        FILTER_NUM: Triangular filters in the LFCC filter bank.
        WITH_ENERGY: Whether the first LFCC coefficient is replaced by log
            energy.
        WITH_DELTA: Whether delta and delta-delta coefficients are appended.
        EMBEDDING_DIM: Width of the pooled embedding feeding the P2SGrad head.
        NUM_CLASSES: Output classes. Always 2: spoof (index 0), bonafide
            (index 1).
    """

    CHECKPOINT_PATH: str = Field(
        default=(
            "/home/jahurtado905/notebooks/anti-spoofing/anti-spoof-eval/"
            "03-asvspoof-mega/trained_network_att.pt"
        ),
        description="Path to the published fixed-crop checkpoint (misnamed 'att').",
    )
    FRAME_LENGTH: int = Field(default=320, description="LFCC frame length, samples.")
    FRAME_SHIFT: int = Field(default=160, description="LFCC frame shift, samples.")
    FFT_N: int = Field(default=512, description="FFT length.")
    FILTER_NUM: int = Field(default=20, description="LFCC filter bank size.")
    WITH_ENERGY: bool = Field(default=True, description="Replace coef 0 with energy.")
    WITH_DELTA: bool = Field(default=True, description="Append delta/delta-delta.")
    EMBEDDING_DIM: int = Field(default=64, description="Pooled embedding width.")
    NUM_CLASSES: int = Field(default=2, description="Output classes.")


# Module-level singleton
settings = LCNNFixedCropBackendSettings()
