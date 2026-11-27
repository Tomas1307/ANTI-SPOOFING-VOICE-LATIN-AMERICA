"""
LFCC-LCNN-BLSTM-P2SGrad detector: Jaime Hurtado's ASVspoof2019+HABLA v1 baseline.
"""
from pathlib import Path
from typing import Any, Dict, List

import torch
from loguru import logger
from torch import nn

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector
from app.pipeline.training.lcnn.modules.blstm_layer import BLSTMLayer
from app.pipeline.training.lcnn.modules.lfcc_frontend import LFCC
from app.pipeline.training.lcnn.modules.max_feature_map import MaxFeatureMap2D
from app.pipeline.training.lcnn.modules.p2s_activation import P2SActivationLayer
from app.pipeline.training.lcnn.settings import settings


class LCNNDetector(BaseSpoofDetector):
    """Adapter over Jaime Hurtado's LFCC-LCNN-BLSTM-P2SGrad checkpoint.

    Reproduces "Accent-Based Evaluation of Speech Anti-spoofing
    Countermeasures Across Multiple Languages" (ICAI 2026), the LSTM-sum
    pooling backend, trained on ASVspoof2019 English plus HABLA v1 Spanish.
    Ported module-for-module from project-NN-Pytorch-scripts (Xin Wang, NII)
    so a published checkpoint loads into this class without a key mismatch:
    LFCC front end, an eight-block Max-Feature-Map convolutional stack, two
    stacked bidirectional LSTMs, a linear embedding layer, and a P2SGrad
    cosine-similarity output.

    The module tree below (``m_frontend``, ``m_transform``,
    ``m_before_pooling``, ``m_output_act``, ``m_angle``, each a one-element
    ``ModuleList`` mirroring the original's multi-front-end design collapsed
    to a single front end) exists purely so the checkpoint's parameter names
    match exactly; nothing here is an architectural choice of ours.

    Unlike DF-Arena, this backbone has no fixed input-length contract, and its
    frame-pooling sum (``(hidden_features_lstm + hidden_features).sum(1)``) is
    unmasked in the source it was ported from: it is only correct when every
    clip in a batch shares the same length, since a padded clip's silent
    frames would otherwise be summed in as if they were signal. Rather than
    derive a frame-count formula through four pooling layers to build a mask
    -- fragile, and a wrong derivation would silently corrupt every score --
    ``forward`` asserts uniform lengths and raises if the batch is padded.
    Run this backend with a fixed, non-zero crop so every clip is the same
    length; batches of naturally variable-length full clips are not supported.

    His score convention already matches this project's: index 1 is bonafide,
    index 0 is spoof, so no relabelling is applied.

    Attributes:
        m_frontend: Single-element ModuleList holding the LFCC front end.
        m_transform: Single-element ModuleList holding the Max-Feature-Map
            convolutional stack.
        m_before_pooling: Single-element ModuleList holding the two stacked
            BLSTM layers.
        m_output_act: Single-element ModuleList holding the embedding
            projection.
        m_angle: Single-element ModuleList holding the P2SGrad output layer.
    """

    def __init__(self, checkpoint_path: str = None) -> None:
        """Initialize the detector and load the published checkpoint.

        Args:
            checkpoint_path: Path to the .pt state dict. Defaults to the
                LSTM-sum checkpoint path in the backend settings.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        super().__init__()

        lfcc_dim = settings.FILTER_NUM * (3 if settings.WITH_DELTA else 1)
        pooled_dim = (lfcc_dim // 16) * 32

        self.m_frontend = nn.ModuleList(
            [
                LFCC(
                    frame_length=settings.FRAME_LENGTH,
                    frame_shift=settings.FRAME_SHIFT,
                    fft_n=settings.FFT_N,
                    sample_rate=16000,
                    filter_num=settings.FILTER_NUM,
                    with_energy=settings.WITH_ENERGY,
                    with_delta=settings.WITH_DELTA,
                )
            ]
        )
        self.m_transform = nn.ModuleList([self._build_conv_stack()])
        self.m_before_pooling = nn.ModuleList(
            [
                nn.Sequential(
                    BLSTMLayer(pooled_dim, pooled_dim),
                    BLSTMLayer(pooled_dim, pooled_dim),
                )
            ]
        )
        self.m_output_act = nn.ModuleList(
            [nn.Linear(pooled_dim, settings.EMBEDDING_DIM)]
        )
        self.m_angle = nn.ModuleList(
            [P2SActivationLayer(settings.EMBEDDING_DIM, settings.NUM_CLASSES)]
        )

        # Unused legacy normalisation buffers, present only so the checkpoint's
        # keys load without a mismatch. Neither this class nor the ported
        # source calls normalize_input/normalize_output anywhere in the
        # scoring path.
        self.input_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(1), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(1), requires_grad=False)

        self._load_checkpoint(checkpoint_path or settings.CHECKPOINT_PATH)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LCNNDetector ready: {checkpoint_path or settings.CHECKPOINT_PATH}, "
            f"trainable {trainable:,}/{total:,} parameters"
        )

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Score a batch of equal-length waveforms.

        Args:
            waveform: Waveforms of shape (batch, samples). Every item must
                carry the same length.
            lengths: True sample count per item, used only to verify the
                batch is unpadded.

        Returns:
            Cosine-similarity scores of shape (batch, 2), spoof at index 0,
            bonafide at index 1. Bounded in [-1, 1] rather than being
            conventional logits, but a valid ranking score for EER.

        Raises:
            ValueError: If the batch mixes clip lengths, which would mean
                padded frames are present with no mask to exclude them.
        """
        self._assert_uniform_lengths(lengths)

        features = self.m_frontend[0](waveform)
        hidden = self.m_transform[0](features.unsqueeze(1))
        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        batch_size, frame_num = hidden.shape[0], hidden.shape[1]
        hidden = hidden.view(batch_size, frame_num, -1)

        hidden_lstm = self.m_before_pooling[0](hidden)
        embedding = self.m_output_act[0]((hidden_lstm + hidden).sum(1))
        return self.m_angle[0](embedding)

    def parameter_groups(
        self, head_learning_rate: float, backbone_learning_rate: float
    ) -> List[Dict[str, Any]]:
        """Build optimiser parameter groups.

        The P2SGrad output layer and the embedding projection are the closest
        thing this architecture has to a head; the LFCC front end contributes
        no trainable weights (its filter bank and DCT matrix are frozen by
        construction), so the convolutional stack and the two BLSTM layers are
        the backbone.

        Args:
            head_learning_rate: Peak learning rate for the output layer and
                embedding projection.
            backbone_learning_rate: Peak learning rate for the convolutional
                stack and BLSTM layers.

        Returns:
            Two parameter groups, head and backbone.
        """
        head_params = list(self.m_output_act.parameters()) + list(
            self.m_angle.parameters()
        )
        backbone_params = list(self.m_transform.parameters()) + list(
            self.m_before_pooling.parameters()
        )
        return [
            {"params": head_params, "lr": head_learning_rate, "name": "lcnn_head"},
            {
                "params": backbone_params,
                "lr": backbone_learning_rate,
                "name": "lcnn_backbone",
            },
        ]

    @staticmethod
    def _build_conv_stack() -> nn.Sequential:
        """Build the Max-Feature-Map convolutional stack.

        The layer order and channel counts are ported exactly from the
        published architecture; every index below was verified against the
        checkpoint's own parameter shapes before this class was written.

        Returns:
            The 29-module Sequential, indices 0-28.
        """
        return nn.Sequential(
            nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
            MaxFeatureMap2D(),
            nn.MaxPool2d([2, 2], [2, 2]),
            nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),
            nn.MaxPool2d([2, 2], [2, 2]),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(48, affine=False),
            nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),
            nn.MaxPool2d([2, 2], [2, 2]),
            nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(64, affine=False),
            nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
            MaxFeatureMap2D(),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
            MaxFeatureMap2D(),
            nn.MaxPool2d([2, 2], [2, 2]),
            nn.Dropout(0.7),
        )

    @staticmethod
    def _assert_uniform_lengths(lengths: torch.Tensor) -> None:
        """Verify every clip in the batch has the same true length.

        Args:
            lengths: True sample count per item.

        Raises:
            ValueError: If the batch mixes lengths, meaning shorter clips
                carry padding this forward pass does not mask out.
        """
        if lengths.numel() > 1 and (lengths != lengths[0]).any():
            raise ValueError(
                "LCNNDetector requires every clip in a batch to share the "
                "same length; its pooling step has no padding mask. Pass a "
                "fixed, non-zero --crop-seconds/--eval-crop-seconds so the "
                "dataset crops every clip uniformly, rather than scoring "
                "variable-length full clips."
            )

    def _load_checkpoint(self, path: str) -> None:
        """Load the published state dict.

        Args:
            path: Checkpoint path.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        checkpoint_file = Path(path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"LCNN checkpoint not found: {path}")

        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded {len(state_dict)} tensors from {path}")
