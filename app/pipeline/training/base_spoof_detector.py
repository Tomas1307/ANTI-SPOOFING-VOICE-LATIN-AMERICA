"""
Abstract interface every anti-spoofing detector backend implements.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from torch import nn


class BaseSpoofDetector(nn.Module, ABC):
    """Common contract for detector backends.

    Backends differ in their front end and backbone but agree on this
    interface, so the training and evaluation steps never learn which model
    they are driving. Adding Nes2Net or HoliAntiSpoof later means adding a
    subclass and a factory entry, and touching nothing else.

    Attributes:
        required_samples: Exact input length the backend demands, in samples,
            or None when it accepts variable-length input. A backend that
            publishes a fixed contract is authoritative about it: the dataset
            crops to this length rather than to whatever the run configuration
            asked for, and the steps log the override. DF-Arena, for instance,
            requires exactly 64,600 samples.
    """

    required_samples: Optional[int] = None

    @abstractmethod
    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Score a batch of waveforms.

        Args:
            waveform: Padded waveforms of shape (batch, samples).
            lengths: True sample count per item, of shape (batch,).

        Returns:
            Class logits of shape (batch, 2), spoof at index 0 and bonafide at
            index 1.
        """
        raise NotImplementedError

    @abstractmethod
    def parameter_groups(
        self, head_learning_rate: float, backbone_learning_rate: float
    ) -> List[Dict[str, Any]]:
        """Build the optimiser parameter groups for this backend.

        A freshly initialised head needs a far larger learning rate than a
        pretrained backbone, so the two are always separated.

        Args:
            head_learning_rate: Peak learning rate for the classifier head.
            backbone_learning_rate: Peak learning rate for backbone weights.

        Returns:
            Parameter groups ready to hand to an optimiser.
        """
        raise NotImplementedError

    @staticmethod
    def score_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """Reduce class logits to a single countermeasure score.

        The score is the bonafide log-likelihood ratio, which is the quantity
        the ASVspoof evaluation convention expects: higher means more genuine.

        Args:
            logits: Class logits of shape (batch, 2).

        Returns:
            Scores of shape (batch,).
        """
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        return log_probabilities[:, 1] - log_probabilities[:, 0]
