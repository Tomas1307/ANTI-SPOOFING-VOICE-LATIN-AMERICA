"""
Backend-specific settings for the self-weighted-pooling LCNN detector.
"""
from pydantic import BaseModel, Field


class LCNNSelfAttnBackendSettings(BaseModel):
    """Configuration for Xin Wang's "attention" LCNN backend.

    This is the architecture reported as the overall best configuration in
    "Accent-Based Evaluation of Speech Anti-spoofing Countermeasures Across
    Multiple Languages" (Hurtado Romero, Acosta Bernal, Manrique; ICAI 2026)
    for Spanish-trained models: LFCC + SelfWeightedPooling + P2SGrad,
    0.14% EER on the paper's own matched Spanish test set.

    Attributes:
        CHECKPOINT_PATH: Path to the published .pt state dict,
            ``trained_network_att.pt`` on ml-server03. Confirmed by the same
            unambiguous fingerprint used for the other two LCNN backends
            (2026-08-25): its 46-tensor state dict, with
            ``m_pooling.0.mm_weights`` and a plain ``m_output_act.0.weight``,
            is unique to this architecture.
        FRAME_LENGTH: LFCC analysis frame length, in waveform samples.
        FRAME_SHIFT: LFCC analysis frame shift, in waveform samples.
        FFT_N: FFT length.
        FILTER_NUM: Triangular filters in the LFCC filter bank.
        WITH_ENERGY: Whether the first LFCC coefficient is replaced by log
            energy.
        WITH_DELTA: Whether delta and delta-delta coefficients are appended.
        POOLING_NUM_HEAD: Attention heads in the self-weighted pooling layer.
        POOLING_MEAN_ONLY: Whether pooling outputs only the mean (True) or
            mean and standard deviation concatenated (False). This checkpoint
            uses False, confirmed by its Linear layer's 192 = 96*2 input
            width.
        EMBEDDING_DIM: Width of the pooled embedding feeding the P2SGrad head.
        NUM_CLASSES: Output classes. Always 2: spoof (index 0), bonafide
            (index 1).
    """

    CHECKPOINT_PATH: str = Field(
        default=(
            "/home/jahurtado905/notebooks/anti-spoofing/anti-spoof-eval/"
            "03-asvspoof-mega/trained_network_att.pt"
        ),
        description="Path to the published self-weighted-pooling checkpoint.",
    )
    FRAME_LENGTH: int = Field(default=320, description="LFCC frame length, samples.")
    FRAME_SHIFT: int = Field(default=160, description="LFCC frame shift, samples.")
    FFT_N: int = Field(default=512, description="FFT length.")
    FILTER_NUM: int = Field(default=20, description="LFCC filter bank size.")
    WITH_ENERGY: bool = Field(default=True, description="Replace coef 0 with energy.")
    WITH_DELTA: bool = Field(default=True, description="Append delta/delta-delta.")
    POOLING_NUM_HEAD: int = Field(default=1, description="Attention heads.")
    POOLING_MEAN_ONLY: bool = Field(
        default=False, description="Mean only, or mean and std concatenated."
    )
    EMBEDDING_DIM: int = Field(default=64, description="Pooled embedding width.")
    NUM_CLASSES: int = Field(default=2, description="Output classes.")


# Module-level singleton
settings = LCNNSelfAttnBackendSettings()
