"""
LFCC-LCNN detector with self-weighted attention pooling (Xin Wang's "attention" backend).
"""
from pathlib import Path
from typing import Any, Dict, List

import torch
from loguru import logger
from torch import nn

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector
from app.pipeline.training.lcnn.modules.lfcc_frontend import LFCC
from app.pipeline.training.lcnn.modules.max_feature_map import MaxFeatureMap2D
from app.pipeline.training.lcnn.modules.p2s_activation import P2SActivationLayer
from app.pipeline.training.lcnn.modules.self_weighted_pooling import (
    SelfWeightedPooling,
)
from app.pipeline.training.lcnn_selfattn.settings import settings


class LCNNSelfAttnDetector(BaseSpoofDetector):
    """Adapter over Xin Wang's "attention" LCNN backend: LFCC-LCNN with
    self-weighted attention pooling instead of BLSTM-sum or fixed truncation.

    Its checkpoint is ``trained_network_att.pt`` on ml-server03, confirmed by
    the same unambiguous, single-process fingerprint used for the other two
    LCNN backends: this architecture's 46-tensor state dict, with
    ``m_pooling.0.mm_weights`` and a plain (non-Sequential)
    ``m_output_act.0.weight``, is unique to it.

    This is the configuration reported as the paper's overall best for
    Spanish-trained models (0.14% EER on the authors' own matched Spanish
    test set), which the corpus this project builds is evaluated against.

    Like the LSTM-sum backend, this architecture places no fixed-length
    requirement on its input -- the front end and pooling both operate on
    whatever frame count the waveform produces. Also like LSTM-sum, its
    pooling has no padding mask: ``SelfWeightedPooling``'s softmax attention
    would assign some weight to padded frames as though they were signal.
    ``forward`` therefore asserts every clip in a batch shares the same
    length and raises rather than silently scoring a padded batch.

    Attributes:
        m_frontend: The LFCC front end.
        m_transform: The Max-Feature-Map convolutional stack.
        m_pooling: The self-weighted attention pooling layer.
        m_output_act: The linear embedding projection.
        m_angle: The P2SGrad output layer.
    """

    def __init__(self, checkpoint_path: str = None) -> None:
        """Initialize the detector and load the published checkpoint.

        Args:
            checkpoint_path: Path to the .pt state dict. Defaults to the
                backend settings' checkpoint path.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        super().__init__()

        lfcc_dim = settings.FILTER_NUM * (3 if settings.WITH_DELTA else 1)
        pooled_dim = (lfcc_dim // 16) * 32
        pooling_output_dim = pooled_dim * settings.POOLING_NUM_HEAD * (
            1 if settings.POOLING_MEAN_ONLY else 2
        )

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
        self.m_pooling = nn.ModuleList(
            [
                SelfWeightedPooling(
                    feature_dim=pooled_dim,
                    num_head=settings.POOLING_NUM_HEAD,
                    mean_only=settings.POOLING_MEAN_ONLY,
                )
            ]
        )
        self.m_output_act = nn.ModuleList(
            [nn.Linear(pooling_output_dim, settings.EMBEDDING_DIM)]
        )
        self.m_angle = nn.ModuleList(
            [P2SActivationLayer(settings.EMBEDDING_DIM, settings.NUM_CLASSES)]
        )

        self.input_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(1), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(1), requires_grad=False)

        self._load_checkpoint(checkpoint_path or settings.CHECKPOINT_PATH)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LCNNSelfAttnDetector ready: {checkpoint_path or settings.CHECKPOINT_PATH}, "
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
            bonafide at index 1.

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

        pooled = self.m_pooling[0](hidden)
        embedding = self.m_output_act[0](pooled)
        return self.m_angle[0](embedding)

    def parameter_groups(
        self, head_learning_rate: float, backbone_learning_rate: float
    ) -> List[Dict[str, Any]]:
        """Build optimiser parameter groups.

        Args:
            head_learning_rate: Peak learning rate for the pooling layer,
                embedding projection and output layer.
            backbone_learning_rate: Peak learning rate for the convolutional
                stack.

        Returns:
            Two parameter groups, head and backbone.
        """
        head_params = (
            list(self.m_pooling.parameters())
            + list(self.m_output_act.parameters())
            + list(self.m_angle.parameters())
        )
        backbone_params = list(self.m_transform.parameters())
        return [
            {"params": head_params, "lr": head_learning_rate, "name": "lcnn_sa_head"},
            {
                "params": backbone_params,
                "lr": backbone_learning_rate,
                "name": "lcnn_sa_backbone",
            },
        ]

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
                "LCNNSelfAttnDetector requires every clip in a batch to share "
                "the same length; its attention pooling has no padding mask. "
                "Pass a fixed, non-zero --crop-seconds/--eval-crop-seconds so "
                "the dataset crops every clip uniformly, rather than scoring "
                "variable-length full clips."
            )

    @staticmethod
    def _build_conv_stack() -> nn.Sequential:
        """Build the Max-Feature-Map convolutional stack.

        Shares every convolutional layer with the LSTM-sum and fixed-crop
        backends -- verified against this checkpoint's own parameter shapes
        before this class was written. Ends in a trailing Dropout(0.7),
        matching the LSTM-sum backend's stack rather than the fixed-crop
        backend's (which places its Dropout inside the head instead).
        Dropout carries no parameters, so this has no effect on the
        checkpoint load or on eval-mode scoring; it only matters if this
        backend is fine-tuned.

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

    def _load_checkpoint(self, path: str) -> None:
        """Load the published state dict.

        Args:
            path: Checkpoint path.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        checkpoint_file = Path(path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded {len(state_dict)} tensors from {path}")
