"""
Self-weighted attention pooling, faithfully ported from Xin Wang's LCNN baseline.
"""
import torch
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init
from torch import nn


class SelfWeightedPooling(nn.Module):
    """Attention-based pooling over the time axis of a frame-level sequence.

    Learns a shared query vector per attention head, scores every frame
    against it, and takes a softmax over time to obtain per-frame weights.
    The output concatenates the attention-weighted mean and standard
    deviation of the frames, giving twice the input feature width.

    Ported without modification from ``sandbox/block_nn.py`` in
    project-NN-Pytorch-scripts (Xin Wang, NII), so that a checkpoint trained
    against that code loads here without a key mismatch. Reference
    implementation: https://github.com/joaomonteirof/e2e_antispoofing.

    A small amount of Gaussian noise (std 1e-5) is added before computing the
    standard deviation, unconditionally, including at evaluation time. This
    is preserved as published rather than disabled in eval mode: at this
    magnitude it is negligible relative to real feature variance, and the
    project's reproducibility guarantee comes from the pipeline's global
    seeding, not from suppressing every source of run-to-run noise in a
    ported module.

    Like LSTM-sum, this pooling has no fixed-length input requirement, but its
    softmax attention has no padding mask: a padded batch would let attention
    assign weight to silent frames as though they were signal. The owning
    detector is responsible for only calling this on uniform-length batches.

    Attributes:
        feature_dim: Width of each input frame's feature vector.
        num_head: Number of attention heads.
        mean_only: Whether to output only the weighted mean (True) or the
            mean and standard deviation concatenated (False).
        mm_weights: Learnable query vectors, shape (num_head, feature_dim).
    """

    def __init__(
        self, feature_dim: int, num_head: int = 1, mean_only: bool = False
    ) -> None:
        """Initialize the pooling layer.

        Args:
            feature_dim: Width of each input frame's feature vector.
            num_head: Number of attention heads.
            mean_only: Whether to output only the mean, or mean and std.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.mean_only = mean_only
        self.noise_std = 1e-5
        self.num_head = num_head

        self.mm_weights = nn.Parameter(
            torch.Tensor(num_head, feature_dim), requires_grad=True
        )
        torch_init.kaiming_uniform_(self.mm_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool a frame sequence into a fixed-size representation.

        Args:
            inputs: Tensor of shape (batch, length, feature_dim).

        Returns:
            Tensor of shape (batch, feature_dim * num_head), when
            ``mean_only`` is True, or (batch, feature_dim * num_head * 2)
            otherwise.
        """
        representations, _attentions = self._forward_with_attention(inputs)
        return representations

    def _forward_with_attention(self, inputs: torch.Tensor):
        """Pool a frame sequence and also return the attention weights.

        Args:
            inputs: Tensor of shape (batch, length, feature_dim).

        Returns:
            A tuple of (pooled representation, attention weights of shape
            (batch, length, num_head)).
        """
        batch_size = inputs.size(0)
        feat_dim = inputs.size(2)

        weights = torch.bmm(
            inputs,
            self.mm_weights.permute(1, 0).contiguous().unsqueeze(0).repeat(
                batch_size, 1, 1
            ),
        )
        attentions = torch_nn_func.softmax(torch.tanh(weights), dim=1)

        if self.num_head == 1:
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            weighted = torch.bmm(
                inputs.reshape(-1, feat_dim, 1),
                attentions.reshape(-1, 1, self.num_head),
            )
            weighted = weighted.reshape(batch_size, -1, feat_dim * self.num_head)

        if self.mean_only:
            representations = weighted.sum(1)
        else:
            noise = self.noise_std * torch.randn(
                weighted.size(), dtype=weighted.dtype, device=weighted.device
            )
            avg_repr, std_repr = weighted.sum(1), (weighted + noise).std(1)
            representations = torch.cat((avg_repr, std_repr), 1)

        return representations, attentions
