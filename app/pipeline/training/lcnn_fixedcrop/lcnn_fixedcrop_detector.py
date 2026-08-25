"""
LFCC-LCNN detector with fixed-length frame truncation (Xin Wang's "fixed" backend).
"""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from loguru import logger
from torch import nn

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector
from app.pipeline.training.lcnn.modules.lfcc_frontend import LFCC
from app.pipeline.training.lcnn.modules.max_feature_map import MaxFeatureMap2D
from app.pipeline.training.lcnn.modules.p2s_activation import P2SActivationLayer
from app.pipeline.training.lcnn_fixedcrop.settings import settings


class LCNNFixedCropDetector(BaseSpoofDetector):
    """Adapter over Xin Wang's "fixed" LCNN backend: LFCC-LCNN with a fixed
    750-frame truncation instead of BLSTM-sum or self-weighted pooling.

    Its checkpoint file is confusingly named ``trained_network_att.pt`` on
    ml-server03. That name was verified wrong before this class was written:
    the directory Xin Wang's own code calls ``lfcc-lcnn-fixed-p2s`` matches
    this checkpoint's parameter shapes exactly
    (``Linear((750//16)*(60//16)*32, 160) = Linear(4416, 160)``), while the
    directory called ``lfcc-lcnn-attention-p2s`` matches a different
    checkpoint, ``trained_network_fix.pt``. This class is named for the
    architecture it implements, not for the misleading checkpoint filename;
    the default checkpoint path documents the mismatch explicitly.

    Unlike the LSTM-sum backend, this architecture has a genuine fixed-length
    contract, built into the network rather than merely chosen for evaluation
    convenience: the LFCC frontend runs on the full waveform, and the
    resulting frame sequence is then truncated or tiled to exactly 750 frames
    before the flatten-and-classify head, because that head's first Linear
    layer has 750-frame width baked into its trained weight matrix.

    The published truncation is a RANDOM crop position for clips longer than
    750 frames (``pos = torch.rand(...)``), applied even at evaluation time in
    the source this is ported from. This class preserves that behaviour
    faithfully rather than substituting a deterministic centre crop, since a
    centre crop would not reproduce the published model's evaluation
    protocol. Reproducibility across runs comes from the pipeline's global
    seeding, not from a per-module generator.

    Attributes:
        m_frontend: The LFCC front end (unchanged from the LSTM-sum backend).
        m_transform: The Max-Feature-Map convolutional stack.
        m_output_act: The flatten-and-classify head.
        m_angle: The P2SGrad output layer.
    """

    TRUNCATE_FRAMES = 750

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
        flatten_dim = (self.TRUNCATE_FRAMES // 16) * pooled_dim

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
        self.m_output_act = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(0.7),
                    nn.Linear(flatten_dim, 160),
                    MaxFeatureMap2D(),
                    nn.Linear(80, settings.EMBEDDING_DIM),
                )
            ]
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
            f"LCNNFixedCropDetector ready: {checkpoint_path or settings.CHECKPOINT_PATH}, "
            f"trainable {trainable:,}/{total:,} parameters"
        )

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Score a batch of waveforms.

        Args:
            waveform: Waveforms of shape (batch, samples). May be padded;
                ``lengths`` gives the true sample count the frame-domain
                truncation is computed from.
            lengths: True sample count per item.

        Returns:
            Cosine-similarity scores of shape (batch, 2), spoof at index 0,
            bonafide at index 1.
        """
        features = self.m_frontend[0](waveform)
        cropped = self._truncate_frames(features, lengths)

        hidden = self.m_transform[0](cropped.unsqueeze(1))
        embedding = self.m_output_act[0](torch.flatten(hidden, 1))
        return self.m_angle[0](embedding)

    def parameter_groups(
        self, head_learning_rate: float, backbone_learning_rate: float
    ) -> List[Dict[str, Any]]:
        """Build optimiser parameter groups.

        Args:
            head_learning_rate: Peak learning rate for the classify head and
                output layer.
            backbone_learning_rate: Peak learning rate for the convolutional
                stack.

        Returns:
            Two parameter groups, head and backbone.
        """
        head_params = list(self.m_output_act.parameters()) + list(
            self.m_angle.parameters()
        )
        backbone_params = list(self.m_transform.parameters())
        return [
            {"params": head_params, "lr": head_learning_rate, "name": "lcnn_fc_head"},
            {
                "params": backbone_params,
                "lr": backbone_learning_rate,
                "name": "lcnn_fc_backbone",
            },
        ]

    def _truncate_frames(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Crop or tile the LFCC frame sequence to exactly TRUNCATE_FRAMES.

        Ported from the published ``_front_end`` override: the true frame
        count is estimated from the waveform sample length divided by the
        frame shift, independent of the STFT's own frame count. Clips with
        more frames than the target get a random crop window; shorter clips
        are tiled to fill it. Both branches, and the random crop position for
        clips longer than the target, mirror the source exactly.

        Args:
            features: LFCC features of shape (batch, frame_num, dim).
            lengths: True sample count per item.

        Returns:
            Features of shape (batch, TRUNCATE_FRAMES, dim).
        """
        batch_size, _frame_num, dim = features.shape
        target = self.TRUNCATE_FRAMES
        cropped = torch.zeros(
            batch_size, target, dim, dtype=features.dtype, device=features.device
        )

        for item in range(batch_size):
            true_frame_num = int(lengths[item].item()) // self.m_frontend[0].frame_shift
            true_frame_num = max(true_frame_num, 1)

            if true_frame_num > target:
                position = torch.rand([1]).item() * (true_frame_num - target)
                start = int(np.floor(position))
                cropped[item] = features[item, start : start + target, :]
            else:
                repeats = int(np.ceil(target / true_frame_num))
                tiled = features[item, :true_frame_num, :].repeat(repeats, 1)
                cropped[item] = tiled[:target, :]

        return cropped

    @staticmethod
    def _build_conv_stack() -> nn.Sequential:
        """Build the Max-Feature-Map convolutional stack.

        Identical to the LSTM-sum backend's stack, except this variant has no
        final Dropout inside the Sequential; the published source places its
        Dropout at the start of ``m_output_act`` instead.

        Returns:
            The 28-module Sequential, indices 0-27.
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
