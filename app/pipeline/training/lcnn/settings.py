"""
Backend-specific settings for Jaime Hurtado's LFCC-LCNN-BLSTM-P2SGrad detector.

Only parameters that belong to this backend live here. Corpus paths, audit
thresholds and shared optimisation defaults belong in the training pipeline
settings one level up.
"""
from pydantic import BaseModel, Field


class LCNNBackendSettings(BaseModel):
    """Configuration for the LFCC-LCNN-BLSTM-P2SGrad detector backend.

    Reproduces the checkpoints trained for "Accent-Based Evaluation of Speech
    Anti-spoofing Countermeasures Across Multiple Languages" (Hurtado Romero,
    Acosta Bernal, Manrique; ICAI 2026), on ASVspoof2019 English plus HABLA v1
    Spanish. Architecture and hyperparameters are ported from
    project-NN-Pytorch-scripts (Xin Wang, NII); see
    ``app/pipeline/training/lcnn/lcnn_detector.py`` for the module-by-module
    mapping.

    Attributes:
        CHECKPOINT_PATH: Path to the published .pt state dict on ml-server03.
            The LSTM-sum backend is the one whose architecture this class
            reproduces; the attention and fixed-pooling backends use a
            different m_before_pooling and are not compatible with it.
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
            "03-asvspoof-mega/trained_network_lstm.pt"
        ),
        description="Path to the published LSTM-sum checkpoint.",
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
settings = LCNNBackendSettings()
