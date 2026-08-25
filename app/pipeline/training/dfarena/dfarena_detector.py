"""
DF-Arena detector: a self-supervised speech backbone with a pooled classifier.
"""
from typing import Any, Dict, List

import torch
from loguru import logger
from torch import nn
from transformers import AutoConfig, AutoModel

from app.pipeline.training.base_spoof_detector import BaseSpoofDetector


class DFArenaDetector(BaseSpoofDetector):
    """Self-supervised backbone, masked mean pooling and an MLP head.

    The backbone is loaded from the Hugging Face hub and consumes raw
    waveforms. Its frame-level hidden states are averaged over the unpadded
    frames only, then classified. Masking matters: padded frames would
    otherwise pull the pooled representation of short clips toward zero, and
    duration correlates with class in any corpus built from mixed sources.

    Attributes:
        backbone: The pretrained self-supervised encoder.
        classifier: The trainable classification head.
        normalize_input: Whether waveforms are standardised per utterance.
        frozen: Whether backbone parameters are excluded from optimisation.
    """

    def __init__(
        self,
        model_id: str,
        hidden_dim: int,
        dropout: float,
        freeze_backbone: bool,
        normalize_input: bool = True,
    ) -> None:
        """Initialize the detector.

        Args:
            model_id: Hugging Face repository identifier of the backbone.
            hidden_dim: Width of the classifier hidden layer.
            dropout: Dropout applied inside the classifier head.
            freeze_backbone: Whether to freeze the backbone weights.
            normalize_input: Whether to standardise each waveform to zero mean
                and unit variance, which is what wav2vec2-style large models
                expect.
        """
        super().__init__()

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_id, config=config, trust_remote_code=True
        )
        feature_dim = getattr(config, "hidden_size", None)
        if feature_dim is None:
            feature_dim = getattr(config, "output_hidden_size", 1024)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.normalize_input = normalize_input
        self.frozen = freeze_backbone

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DFArenaDetector ready: backbone={model_id}, feature_dim={feature_dim}, "
            f"trainable={trainable:,}/{total:,} parameters, frozen={freeze_backbone}"
        )

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Score a batch of waveforms.

        Args:
            waveform: Padded waveforms of shape (batch, samples).
            lengths: True sample count per item, of shape (batch,).

        Returns:
            Class logits of shape (batch, 2).
        """
        if self.normalize_input:
            waveform = self._standardise(waveform, lengths)

        attention_mask = self._sample_mask(waveform, lengths)
        outputs = self.backbone(
            input_values=waveform, attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state

        pooled = self._masked_mean(hidden, lengths, waveform.shape[1])
        return self.classifier(pooled)

    def parameter_groups(
        self, head_learning_rate: float, backbone_learning_rate: float
    ) -> List[Dict[str, Any]]:
        """Build optimiser parameter groups.

        Args:
            head_learning_rate: Peak learning rate for the classifier head.
            backbone_learning_rate: Peak learning rate for backbone weights.

        Returns:
            One group for the head, plus a backbone group when it is trainable.
        """
        groups: List[Dict[str, Any]] = [
            {
                "params": list(self.classifier.parameters()),
                "lr": head_learning_rate,
                "name": "classifier",
            }
        ]
        if not self.frozen:
            groups.append(
                {
                    "params": [
                        p for p in self.backbone.parameters() if p.requires_grad
                    ],
                    "lr": backbone_learning_rate,
                    "name": "backbone",
                }
            )
        return groups

    @staticmethod
    def _standardise(waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Standardise each waveform over its unpadded samples.

        Args:
            waveform: Padded waveforms of shape (batch, samples).
            lengths: True sample count per item.

        Returns:
            Standardised waveforms of the same shape.
        """
        mask = DFArenaDetector._sample_mask(waveform, lengths)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (waveform * mask).sum(dim=1, keepdim=True) / counts
        variance = (((waveform - mean) * mask) ** 2).sum(dim=1, keepdim=True) / counts
        return (waveform - mean) / torch.sqrt(variance + 1e-7) * mask

    @staticmethod
    def _sample_mask(waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Build a sample-level padding mask.

        Args:
            waveform: Padded waveforms of shape (batch, samples).
            lengths: True sample count per item.

        Returns:
            A float mask of shape (batch, samples), one for real samples.
        """
        positions = torch.arange(waveform.shape[1], device=waveform.device)
        return (positions.unsqueeze(0) < lengths.unsqueeze(1).to(waveform.device)).to(
            waveform.dtype
        )

    @staticmethod
    def _masked_mean(
        hidden: torch.Tensor, lengths: torch.Tensor, input_samples: int
    ) -> torch.Tensor:
        """Average frame-level features over the unpadded frames only.

        The backbone downsamples by a factor this class does not hard-code;
        the frame count per item is derived proportionally from the sample
        count, which holds for any convolutional front end and avoids
        depending on architecture-specific helper methods.

        Args:
            hidden: Frame-level features of shape (batch, frames, dim).
            lengths: True sample count per item.
            input_samples: Padded sample count the backbone consumed.

        Returns:
            Pooled features of shape (batch, dim).
        """
        frames = hidden.shape[1]
        ratio = frames / max(input_samples, 1)
        frame_lengths = torch.clamp(
            (lengths.to(hidden.device).float() * ratio).ceil().long(), min=1, max=frames
        )
        positions = torch.arange(frames, device=hidden.device)
        mask = (positions.unsqueeze(0) < frame_lengths.unsqueeze(1)).to(hidden.dtype)
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        return summed / mask.sum(dim=1, keepdim=True).clamp(min=1)
